"""Benchmark one saved Task 2 baseline model under a common environment.

Run this script once for each model on the same Kaggle accelerator.  It times
only tokenisation + forward inference; model download/loading is excluded.
Rows are replaced by ``model_key`` so interrupted reruns are safe.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAMES = {
    "roberta": "RoBERTa-base",
    "deberta_v3": "DeBERTa-v3-base",
    "xlnet": "XLNet-base-cased",
}


class TextDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.questions = frame["question"].fillna("").astype(str).tolist()
        self.answers = frame["interview_answer"].fillna("").astype(str).tolist()

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, index: int) -> tuple[str, str]:
        return self.questions[index], self.answers[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure inference speed and peak CUDA memory consistently."
    )
    parser.add_argument("--model-key", choices=sorted(MODEL_NAMES), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permit CPU measurement. For the official comparison use one GPU.",
    )
    return parser.parse_args()


def collate_factory(tokenizer, max_length: int):
    def collate(batch: list[tuple[str, str]]) -> dict[str, torch.Tensor]:
        questions, answers = zip(*batch)
        return tokenizer(
            list(questions),
            list(answers),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    return collate


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_inference(model, loader: DataLoader, device: torch.device) -> tuple[float, int]:
    processed = 0
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(**batch).logits
            else:
                logits = model(**batch).logits
            if logits.ndim != 2 or logits.shape[1] != 9:
                raise ValueError(f"Expected [batch, 9] logits, received {tuple(logits.shape)}")
            processed += int(logits.shape[0])
    synchronize(device)
    return time.perf_counter() - started, processed


def upsert_result(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        if "model_key" not in old.columns:
            raise ValueError(f"Existing benchmark lacks model_key: {path}")
        old = old.loc[old["model_key"] != row["model_key"]]
        new = pd.concat([old, new], ignore_index=True)
    order = {key: index for index, key in enumerate(MODEL_NAMES)}
    new["_order"] = new["model_key"].map(order)
    new = new.sort_values("_order").drop(columns="_order")
    new.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    if args.repeats < 1 or args.batch_size < 1:
        raise ValueError("--repeats and --batch-size must be positive")
    if not args.model_dir.exists():
        raise FileNotFoundError(args.model_dir)
    if not args.test_file.exists():
        raise FileNotFoundError(args.test_file)

    cuda = torch.cuda.is_available()
    if not cuda and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable. Enable a Kaggle GPU or pass --allow-cpu.")
    device = torch.device("cuda" if cuda else "cpu")

    frame = pd.read_csv(args.test_file)
    required = {"sample_id", "question", "interview_answer"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Test file is missing columns: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError("Test sample_id values must be unique")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if args.model_key == "xlnet":
        tokenizer.padding_side = "left"
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    if int(model.config.num_labels) != 9:
        raise ValueError(f"Saved model has {model.config.num_labels} labels, expected 9")
    if args.model_key == "xlnet":
        model.config.use_mems_train = False
        model.config.use_mems_eval = False
    model = model.float().to(device).eval()

    loader = DataLoader(
        TextDataset(frame),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=cuda,
        collate_fn=collate_factory(tokenizer, args.max_length),
    )

    # Warm up a bounded number of batches without including them in the timing.
    warmup_loader = []
    for index, batch in enumerate(loader):
        warmup_loader.append(batch)
        if index + 1 >= args.warmup_batches:
            break
    with torch.inference_mode():
        for batch in warmup_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            if cuda:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    model(**batch)
            else:
                model(**batch)
    synchronize(device)

    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    durations: list[float] = []
    for _ in range(args.repeats):
        duration, processed = run_inference(model, loader, device)
        if processed != len(frame):
            raise RuntimeError(f"Processed {processed} rows, expected {len(frame)}")
        durations.append(duration)

    mean_seconds = statistics.fmean(durations)
    std_seconds = statistics.stdev(durations) if len(durations) > 1 else 0.0
    peak_gib = (
        torch.cuda.max_memory_allocated(device) / 1024**3 if cuda else float("nan")
    )
    row = {
        "model_key": args.model_key,
        "model": MODEL_NAMES[args.model_key],
        "model_dir": str(args.model_dir),
        "num_samples": len(frame),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "repeats": args.repeats,
        "mean_inference_seconds": mean_seconds,
        "std_inference_seconds": std_seconds,
        "samples_per_second": len(frame) / mean_seconds,
        "milliseconds_per_sample": mean_seconds * 1000 / len(frame),
        "peak_gpu_memory_gib": peak_gib,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if cuda else "CPU",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "run_seconds_json": json.dumps(durations),
    }
    upsert_result(args.output_csv, row)
    print(pd.DataFrame([row]).to_string(index=False))
    print(f"Saved benchmark: {args.output_csv}")


if __name__ == "__main__":
    main()
