"""Train and evaluate the RoBERTa-base Task 2 nine-class baseline.

This script is the command-line version of
``roberta-task2-qevasion-recovered.ipynb``. It consumes the processed data and
evaluation interface delivered on the ``feature/A-data`` branch.

Examples
--------
Smoke test only::

    python train_roberta.py --mode smoke \
        --project-dir /kaggle/working/Response_Clarity_Project \
        --output-dir /kaggle/working/roberta_task2_outputs

Full training, prediction, and evaluation::

    python train_roberta.py --mode train \
        --project-dir /kaggle/working/Response_Clarity_Project \
        --output-dir /kaggle/working/roberta_task2_outputs

The mode argument is required so that running the script without arguments
cannot accidentally start a full GPU training job.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


TASK2_LABELS = [
    "Explicit",
    "Implicit",
    "Dodging",
    "General",
    "Deflection",
    "Declining to answer",
    "Claims ignorance",
    "Clarification",
    "Partial/half-answer",
]

LABEL_TO_ID = {label: index for index, label in enumerate(TASK2_LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

PROBABILITY_COLUMNS = [
    "p_Explicit",
    "p_Implicit",
    "p_Dodging",
    "p_General",
    "p_Deflection",
    "p_Declining_to_answer",
    "p_Claims_ignorance",
    "p_Clarification",
    "p_Partial_half_answer",
]

LOGIT_COLUMNS = [
    "logit_Explicit",
    "logit_Implicit",
    "logit_Dodging",
    "logit_General",
    "logit_Deflection",
    "logit_Declining_to_answer",
    "logit_Claims_ignorance",
    "logit_Clarification",
    "logit_Partial_half_answer",
]

EXPECTED_SPLIT_SIZES = {
    "train": 2700,
    "validation": 748,
    "test": 308,
}


class QAPairDataset(Dataset):
    """Tokenize question and interview answer as a RoBERTa sequence pair."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: Any,
        max_length: int,
        include_labels: bool = True,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.include_labels = include_labels

        self.encodings = tokenizer(
            self.dataframe["question"].tolist(),
            self.dataframe["interview_answer"].tolist(),
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        self.labels = (
            self.dataframe["label_id"].tolist() if include_labels else None
        )

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {
            key: values[index]
            for key, values in self.encodings.items()
        }
        if self.include_labels:
            item["labels"] = int(self.labels[index])
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train FacebookAI/roberta-base for Task 2 nine-class response "
            "classification using the processed A-group data."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "train", "all"),
        required=True,
        help=(
            "smoke: small pipeline test only; train: full experiment only; "
            "all: smoke test followed by the full experiment."
        ),
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        required=True,
        help="Root of the cloned Response_Clarity_Project repository.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory for checkpoints, predictions, results, and logs.",
    )
    parser.add_argument(
        "--model-id",
        default="FacebookAI/roberta-base",
        help="Hugging Face model identifier.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--dataloader-workers", type=int, default=2)
    parser.add_argument("--smoke-train-samples", type=int, default=64)
    parser.add_argument("--smoke-validation-samples", type=int, default=32)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help=(
            "Allow execution without CUDA. Full training on CPU is not "
            "recommended and fp16 is automatically disabled."
        ),
    )
    return parser.parse_args()


def configure_reproducibility(seed: int) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_compute_device(allow_cpu: bool) -> bool:
    has_cuda = torch.cuda.is_available()
    if not has_cuda and not allow_cpu:
        raise RuntimeError(
            "CUDA GPU is not available. Enable a Kaggle GPU or pass "
            "--allow-cpu for a slow CPU-only check."
        )

    if has_cuda:
        print("PyTorch:", torch.__version__)
        print("Visible GPU count:", torch.cuda.device_count())
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "GPU memory GiB:",
            round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3,
                2,
            ),
        )
    else:
        print("WARNING: running without CUDA; fp16 is disabled.")
    return has_cuda


def prepare_output_directories(output_root: Path) -> dict[str, Path]:
    directories = {
        "root": output_root,
        "smoke": output_root / "smoke_test",
        "formal": output_root / "formal_training",
        "best_model": output_root / "best_model",
        "predictions": output_root / "predictions",
        "results": output_root / "results",
        "logs": output_root / "logs",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def validate_required_files(project_dir: Path) -> dict[str, Path]:
    data_dir = project_dir / "01_Data" / "data_splits"
    paths = {
        "train": data_dir / "train.csv",
        "validation": data_dir / "validation.csv",
        "test": data_dir / "test.csv",
        "mapping": project_dir / "01_Data" / "label_mapping.json",
        "evaluation": project_dir / "05_Evaluation" / "evaluation.py",
        "handover": project_dir / "A_data_handover.md",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required A-group files are missing:\n- " + "\n- ".join(missing)
        )
    return paths


def load_and_validate_data(
    paths: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(paths["train"])
    validation_df = pd.read_csv(paths["validation"])
    test_df = pd.read_csv(paths["test"])

    labelled_required = {
        "sample_id",
        "question",
        "interview_answer",
        "evasion_label",
    }
    test_required = {
        "sample_id",
        "question",
        "interview_answer",
        "annotator1",
        "annotator2",
        "annotator3",
        "evasion_label_majority",
        "majority_vote_count",
    }

    for split_name, dataframe, required in (
        ("train", train_df, labelled_required),
        ("validation", validation_df, labelled_required),
        ("test", test_df, test_required),
    ):
        missing_columns = sorted(required - set(dataframe.columns))
        if missing_columns:
            raise ValueError(
                f"{split_name} is missing columns: {missing_columns}"
            )
        expected_size = EXPECTED_SPLIT_SIZES[split_name]
        if len(dataframe) != expected_size:
            raise ValueError(
                f"{split_name} has {len(dataframe)} rows; "
                f"expected {expected_size}."
            )

    all_ids = pd.concat(
        [
            train_df["sample_id"],
            validation_df["sample_id"],
            test_df["sample_id"],
        ],
        ignore_index=True,
    )
    if not all_ids.notna().all() or all_ids.nunique() != len(all_ids):
        raise ValueError("sample_id values must be non-null and globally unique.")

    for dataframe in (train_df, validation_df, test_df):
        for column in ("question", "interview_answer"):
            dataframe[column] = dataframe[column].fillna("").astype(str)

    for split_name, dataframe in (
        ("train", train_df),
        ("validation", validation_df),
    ):
        labels = set(dataframe["evasion_label"].dropna().unique())
        if labels != set(TASK2_LABELS):
            raise ValueError(
                f"{split_name} label set does not match Task 2 labels: {labels}"
            )
        dataframe["label_id"] = dataframe["evasion_label"].map(LABEL_TO_ID)
        if dataframe["label_id"].isna().any():
            raise ValueError(f"{split_name} contains unmapped labels.")
        dataframe["label_id"] = dataframe["label_id"].astype(int)

    print("Train:", train_df.shape)
    print("Validation:", validation_df.shape)
    print("Test:", test_df.shape)
    print("Label mapping:", LABEL_TO_ID)
    return train_df, validation_df, test_df


def compute_metrics(eval_prediction: Any) -> dict[str, float]:
    logits, label_ids = eval_prediction
    if isinstance(logits, tuple):
        logits = logits[0]
    predicted_ids = np.argmax(logits, axis=1)
    return {
        "accuracy": float(accuracy_score(label_ids, predicted_ids)),
        "macro_f1": float(
            f1_score(
                label_ids,
                predicted_ids,
                labels=list(range(len(TASK2_LABELS))),
                average="macro",
                zero_division=0,
            )
        ),
    }


def create_fresh_model(model_id: str) -> Any:
    return AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=len(TASK2_LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )


def build_trainer(
    model: Any,
    training_args: TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer: Any,
    data_collator: DataCollatorWithPadding,
) -> Trainer:
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )


def run_smoke_test(
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    tokenizer: Any,
    data_collator: DataCollatorWithPadding,
    directories: dict[str, Path],
    has_cuda: bool,
) -> None:
    smoke_train_df, _ = train_test_split(
        train_df,
        train_size=args.smoke_train_samples,
        random_state=args.seed,
        stratify=train_df["evasion_label"],
    )
    smoke_validation_df, _ = train_test_split(
        validation_df,
        train_size=args.smoke_validation_samples,
        random_state=args.seed,
        stratify=validation_df["evasion_label"],
    )

    smoke_train_dataset = QAPairDataset(
        smoke_train_df,
        tokenizer,
        max_length=min(args.max_length, 128),
        include_labels=True,
    )
    smoke_validation_dataset = QAPairDataset(
        smoke_validation_df,
        tokenizer,
        max_length=min(args.max_length, 128),
        include_labels=True,
    )

    smoke_args = TrainingArguments(
        output_dir=str(directories["smoke"]),
        num_train_epochs=1,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        weight_decay=args.weight_decay,
        warmup_steps=0,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        fp16=has_cuda,
        optim="adamw_torch",
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=args.dataloader_workers,
    )

    smoke_trainer = build_trainer(
        create_fresh_model(args.model_id),
        smoke_args,
        smoke_train_dataset,
        smoke_validation_dataset,
        tokenizer,
        data_collator,
    )
    start = time.perf_counter()
    train_result = smoke_trainer.train()
    metrics = smoke_trainer.evaluate()
    prediction = smoke_trainer.predict(smoke_validation_dataset)
    logits = prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = np.asarray(logits)

    expected_shape = (args.smoke_validation_samples, len(TASK2_LABELS))
    if logits.shape != expected_shape or not np.isfinite(logits).all():
        raise ValueError(
            f"Invalid smoke-test logits: {logits.shape}; "
            f"expected {expected_shape}."
        )
    probabilities = stable_softmax(logits)
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Smoke-test probability rows do not sum to 1.")

    print("Smoke training metrics:", train_result.metrics)
    print("Smoke validation metrics:", metrics)
    print("Smoke runtime seconds:", round(time.perf_counter() - start, 2))
    print("SMOKE TEST PASSED")

    del smoke_trainer, prediction, logits, probabilities
    gc.collect()
    if has_cuda:
        torch.cuda.empty_cache()


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def get_logits(trainer: Trainer, dataset: Dataset) -> np.ndarray:
    prediction_output = trainer.predict(dataset)
    logits = prediction_output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    return np.asarray(logits)


def save_predictions(
    trainer: Trainer,
    validation_dataset: Dataset,
    test_dataset: Dataset,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    prediction_dir: Path,
) -> dict[str, Path]:
    validation_logits = get_logits(trainer, validation_dataset)
    test_logits = get_logits(trainer, test_dataset)

    if validation_logits.shape != (EXPECTED_SPLIT_SIZES["validation"], 9):
        raise ValueError(f"Unexpected validation logits: {validation_logits.shape}")
    if test_logits.shape != (EXPECTED_SPLIT_SIZES["test"], 9):
        raise ValueError(f"Unexpected test logits: {test_logits.shape}")

    validation_probabilities = stable_softmax(validation_logits)
    test_probabilities = stable_softmax(test_logits)
    if not np.allclose(validation_probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Validation probability rows do not sum to 1.")
    if not np.allclose(test_probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Test probability rows do not sum to 1.")

    validation_labels = [
        ID_TO_LABEL[index]
        for index in np.argmax(validation_probabilities, axis=1)
    ]
    test_labels = [
        ID_TO_LABEL[index]
        for index in np.argmax(test_probabilities, axis=1)
    ]

    validation_predictions = pd.DataFrame(
        {
            "sample_id": validation_df["sample_id"],
            "gold_label": validation_df["evasion_label"],
            "predicted_label": validation_labels,
        }
    )
    validation_probability_table = pd.DataFrame(
        validation_probabilities,
        columns=PROBABILITY_COLUMNS,
    )
    validation_probability_table.insert(
        0, "sample_id", validation_df["sample_id"].to_numpy()
    )
    validation_logit_table = pd.DataFrame(
        validation_logits,
        columns=LOGIT_COLUMNS,
    )
    validation_logit_table.insert(
        0, "sample_id", validation_df["sample_id"].to_numpy()
    )
    test_predictions = pd.DataFrame(
        {
            "sample_id": test_df["sample_id"],
            "predicted_label": test_labels,
        }
    )
    test_probability_table = pd.DataFrame(
        test_probabilities,
        columns=PROBABILITY_COLUMNS,
    )
    test_probability_table.insert(
        0, "sample_id", test_df["sample_id"].to_numpy()
    )

    paths = {
        "validation_predictions": (
            prediction_dir / "roberta_validation_predictions.csv"
        ),
        "validation_probabilities": (
            prediction_dir / "roberta_validation_probabilities.csv"
        ),
        "validation_logits": (
            prediction_dir / "roberta_validation_logits.csv"
        ),
        "test_predictions": prediction_dir / "roberta_predictions.csv",
        "test_probabilities": prediction_dir / "roberta_probabilities.csv",
    }
    validation_predictions.to_csv(paths["validation_predictions"], index=False)
    validation_probability_table.to_csv(
        paths["validation_probabilities"], index=False
    )
    validation_logit_table.to_csv(paths["validation_logits"], index=False)
    test_predictions.to_csv(paths["test_predictions"], index=False)
    test_probability_table.to_csv(paths["test_probabilities"], index=False)

    if not test_predictions["sample_id"].is_unique:
        raise ValueError("Test prediction sample_id values are not unique.")
    print("Prediction files saved to:", prediction_dir)
    return paths


def load_evaluation_module(evaluation_file: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "group_evaluation", evaluation_file
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load evaluation module: {evaluation_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_group_evaluation(
    project_dir: Path,
    evaluation_file: Path,
    prediction_paths: dict[str, Path],
    result_dir: Path,
) -> None:
    # The notebook ran after `%cd` into the repository. Mirroring that working
    # directory preserves any relative-path behavior in A's evaluation module.
    previous_cwd = Path.cwd()
    os.chdir(project_dir)
    try:
        group_evaluation = load_evaluation_module(evaluation_file)
        shared_test_df = group_evaluation.load_test_data()

        summaries, class_tables = group_evaluation.evaluate_task2_annotators(
            shared_test_df,
            prediction_paths["test_probabilities"],
            model_name="RoBERTa-base",
            prediction_file=prediction_paths["test_predictions"],
        )
        group_evaluation.save_results(
            summaries,
            class_tables,
            prefix="roberta_task2",
        )

        majority_summary, majority_class_table = (
            group_evaluation.evaluate_task2_majority(
                shared_test_df,
                prediction_paths["test_probabilities"],
                model_name="RoBERTa-base",
                prediction_file=prediction_paths["test_predictions"],
            )
        )
        group_evaluation.save_results(
            [majority_summary],
            [majority_class_table],
            prefix="roberta_task2_majority",
        )
    finally:
        os.chdir(previous_cwd)

    evaluation_output_dir = project_dir / "05_Evaluation"
    filenames = [
        "roberta_task2_summary.csv",
        "roberta_task2_per_class.csv",
        "roberta_task2_majority_summary.csv",
        "roberta_task2_majority_per_class.csv",
    ]
    for filename in filenames:
        source = evaluation_output_dir / filename
        if not source.exists():
            raise FileNotFoundError(f"Missing evaluation output: {source}")
        shutil.copy2(source, result_dir / filename)
    print("Evaluation files copied to:", result_dir)


def get_git_commit(project_dir: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(project_dir), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_experiment_record(
    args: argparse.Namespace,
    directories: dict[str, Path],
    trainer: Trainer,
    training_seconds: float,
    warmup_steps: int,
    final_validation_metrics: dict[str, Any],
    train_result_metrics: dict[str, Any],
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    has_cuda: bool,
) -> Path:
    hardware = {
        "device": "cuda" if has_cuda else "cpu",
        "visible_gpu_count": torch.cuda.device_count() if has_cuda else 0,
    }
    if has_cuda:
        hardware.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "vram_gib": round(
                    torch.cuda.get_device_properties(0).total_memory / 1024**3,
                    2,
                ),
            }
        )

    record = {
        "experiment": "RoBERTa-base Task 2 nine-class baseline",
        "model_id": args.model_id,
        "task": "Task 2",
        "num_labels": len(TASK2_LABELS),
        "labels": TASK2_LABELS,
        "input_fields": ["question", "interview_answer"],
        "train_samples": len(train_df),
        "validation_samples": len(validation_df),
        "test_samples": len(test_df),
        "random_seed": args.seed,
        "max_length": args.max_length,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": 1,
        "weight_decay": args.weight_decay,
        "warmup_steps": warmup_steps,
        "fp16": has_cuda,
        "model_selection_metric": "validation_macro_f1",
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_macro_f1": trainer.state.best_metric,
        "final_validation_metrics": final_validation_metrics,
        "training_metrics": train_result_metrics,
        "formal_training_seconds": training_seconds,
        "formal_training_minutes": training_seconds / 60,
        "python_environment": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "hardware": hardware,
        "data_source": {
            "repository": "qwerty2345911/Response_Clarity_Project",
            "branch": "feature/A-data",
            "commit": get_git_commit(args.project_dir),
        },
    }

    path = directories["logs"] / "roberta_experiment_config.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(json_safe(record), file, indent=2, ensure_ascii=False)
    print("Experiment record:", path)
    return path


def create_delivery_zip(output_root: Path) -> Path:
    delivery_zip = output_root.parent / "roberta_task2_results.zip"
    files = (
        list((output_root / "predictions").glob("*.csv"))
        + list((output_root / "results").glob("*.csv"))
        + list((output_root / "logs").glob("*.csv"))
        + list((output_root / "logs").glob("*.json"))
    )
    with zipfile.ZipFile(
        delivery_zip,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in files:
            archive.write(path, arcname=path.relative_to(output_root))
    print("Delivery package:", delivery_zip)
    for path in files:
        print("-", path.relative_to(output_root))
    return delivery_zip


def run_formal_experiment(
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: Any,
    data_collator: DataCollatorWithPadding,
    directories: dict[str, Path],
    evaluation_file: Path,
    has_cuda: bool,
) -> None:
    train_dataset = QAPairDataset(
        train_df, tokenizer, args.max_length, include_labels=True
    )
    validation_dataset = QAPairDataset(
        validation_df, tokenizer, args.max_length, include_labels=True
    )
    test_dataset = QAPairDataset(
        test_df, tokenizer, args.max_length, include_labels=False
    )

    steps_per_epoch = math.ceil(len(train_dataset) / args.train_batch_size)
    total_steps = math.ceil(steps_per_epoch * args.epochs)
    warmup_steps = round(total_steps * args.warmup_ratio)
    print("Estimated steps per epoch:", steps_per_epoch)
    print("Estimated total steps:", total_steps)
    print("Warmup steps:", warmup_steps)

    training_args = TrainingArguments(
        output_dir=str(directories["formal"]),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=1,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=has_cuda,
        optim="adamw_torch",
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=args.dataloader_workers,
        eval_accumulation_steps=8,
    )

    trainer = build_trainer(
        create_fresh_model(args.model_id),
        training_args,
        train_dataset,
        validation_dataset,
        tokenizer,
        data_collator,
    )
    start = time.perf_counter()
    train_result = trainer.train()
    training_seconds = time.perf_counter() - start
    final_validation_metrics = trainer.evaluate()

    print("Formal training metrics:", train_result.metrics)
    print("Best checkpoint:", trainer.state.best_model_checkpoint)
    print("Best validation Macro-F1:", trainer.state.best_metric)
    print("Loaded-best-model validation metrics:", final_validation_metrics)
    print("Formal training time minutes:", round(training_seconds / 60, 2))

    trainer.save_model(str(directories["best_model"]))
    tokenizer.save_pretrained(str(directories["best_model"]))
    pd.DataFrame(trainer.state.log_history).to_csv(
        directories["logs"] / "roberta_training_log.csv",
        index=False,
    )

    prediction_paths = save_predictions(
        trainer,
        validation_dataset,
        test_dataset,
        validation_df,
        test_df,
        directories["predictions"],
    )
    run_group_evaluation(
        args.project_dir,
        evaluation_file,
        prediction_paths,
        directories["results"],
    )
    save_experiment_record(
        args,
        directories,
        trainer,
        training_seconds,
        warmup_steps,
        final_validation_metrics,
        train_result.metrics,
        train_df,
        validation_df,
        test_df,
        has_cuda,
    )
    create_delivery_zip(args.output_dir)


def main() -> None:
    args = parse_args()
    args.project_dir = args.project_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    configure_reproducibility(args.seed)
    has_cuda = check_compute_device(args.allow_cpu)
    paths = validate_required_files(args.project_dir)
    directories = prepare_output_directories(args.output_dir)
    train_df, validation_df, test_df = load_and_validate_data(paths)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    if args.mode in ("smoke", "all"):
        run_smoke_test(
            args,
            train_df,
            validation_df,
            tokenizer,
            data_collator,
            directories,
            has_cuda,
        )

    if args.mode in ("train", "all"):
        run_formal_experiment(
            args,
            train_df,
            validation_df,
            test_df,
            tokenizer,
            data_collator,
            directories,
            paths["evaluation"],
            has_cuda,
        )


if __name__ == "__main__":
    main()
