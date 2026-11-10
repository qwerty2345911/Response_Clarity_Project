"""Audit and compare the three Task 2 nine-class baseline models.

The default mode is intentionally strict: all handover artifacts must be
present and internally consistent before final comparison files are written.
An input audit CSV is always produced, including when validation fails.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = [
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
LOGIT_COLUMNS = [column.replace("p_", "logit_", 1) for column in PROBABILITY_COLUMNS]
EXPECTED_ROWS = {"validation": 748, "test": 308}

MODEL_SPECS = {
    "roberta": {
        "display": "RoBERTa-base",
        "prefix": "roberta",
        "advantages": "训练与部署成熟，通常可作为稳定的 Transformer 基准。",
        "limitations": "对长上下文与细粒度回避类别的区分仍可能不足。",
    },
    "deberta_v3": {
        "display": "DeBERTa-v3-base",
        "prefix": "deberta_v3",
        "advantages": "解耦注意力适合细粒度语义分类。",
        "limitations": "参数量和显存需求高于 RoBERTa，训练速度可能更慢。",
    },
    "xlnet": {
        "display": "XLNet-base-cased",
        "prefix": "xlnet",
        "advantages": "排列语言建模提供了不同于掩码语言模型的归纳偏置。",
        "limitations": "模型较旧且资源效率通常较低，需特别控制 padding 与 memory 设置。",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate baseline artifacts and create CSV/PNG/Markdown comparison."
    )
    parser.add_argument("--inputs-root", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark-file", type=Path)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Generate a partial comparison from available models (draft only).",
    )
    return parser.parse_args()


def expected_filenames(prefix: str) -> dict[str, str]:
    return {
        "test_predictions": f"{prefix}_predictions.csv",
        "test_probabilities": f"{prefix}_probabilities.csv",
        "validation_predictions": f"{prefix}_validation_predictions.csv",
        "validation_probabilities": f"{prefix}_validation_probabilities.csv",
        "validation_logits": f"{prefix}_validation_logits.csv",
        "task2_summary": f"{prefix}_task2_summary.csv",
        "task2_per_class": f"{prefix}_task2_per_class.csv",
        "majority_summary": f"{prefix}_task2_majority_summary.csv",
        "majority_per_class": f"{prefix}_task2_majority_per_class.csv",
        "training_log": f"{prefix}_training_log.csv",
        "experiment_config": f"{prefix}_experiment_config.json",
    }


def locate_unique(root: Path, filename: str) -> tuple[Path | None, str]:
    if not root.exists():
        return None, "model directory missing"
    matches = sorted(path for path in root.rglob(filename) if path.is_file())
    if not matches:
        return None, "missing"
    nonempty = [path for path in matches if path.stat().st_size > 0]
    if not nonempty:
        return matches[0], "empty"
    if len(nonempty) > 1:
        return nonempty[0], f"duplicate ({len(nonempty)} files; using first)"
    return nonempty[0], "present"


def audit_inputs(inputs_root: Path, test_file: Path, benchmark_file: Path | None):
    rows: list[dict[str, Any]] = []
    resolved: dict[str, dict[str, Path]] = {}
    for key, spec in MODEL_SPECS.items():
        model_root = inputs_root / key
        resolved[key] = {}
        for artifact, filename in expected_filenames(spec["prefix"]).items():
            path, status = locate_unique(model_root, filename)
            rows.append(
                {
                    "model_key": key,
                    "artifact": artifact,
                    "expected_filename": filename,
                    "status": status,
                    "resolved_path": str(path) if path else "",
                    "size_bytes": path.stat().st_size if path and path.exists() else 0,
                }
            )
            if path and path.exists() and path.stat().st_size > 0:
                resolved[key][artifact] = path
    rows.append(
        {
            "model_key": "shared",
            "artifact": "test_file",
            "expected_filename": test_file.name,
            "status": "present" if test_file.exists() and test_file.stat().st_size else "missing",
            "resolved_path": str(test_file) if test_file.exists() else "",
            "size_bytes": test_file.stat().st_size if test_file.exists() else 0,
        }
    )
    if benchmark_file is not None:
        rows.append(
            {
                "model_key": "shared",
                "artifact": "inference_benchmark",
                "expected_filename": benchmark_file.name,
                "status": "present" if benchmark_file.exists() and benchmark_file.stat().st_size else "missing",
                "resolved_path": str(benchmark_file) if benchmark_file.exists() else "",
                "size_bytes": benchmark_file.stat().st_size if benchmark_file.exists() else 0,
            }
        )
    return pd.DataFrame(rows), resolved


def require_columns(frame: pd.DataFrame, required: list[str], source: Path) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def validate_ids(frame: pd.DataFrame, expected_rows: int, source: Path) -> None:
    require_columns(frame, ["sample_id"], source)
    if len(frame) != expected_rows:
        raise ValueError(f"{source} has {len(frame)} rows; expected {expected_rows}")
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError(f"{source} contains blank or duplicate sample_id values")


def validate_probability_table(frame: pd.DataFrame, expected_rows: int, source: Path) -> None:
    validate_ids(frame, expected_rows, source)
    expected = ["sample_id", *PROBABILITY_COLUMNS]
    if list(frame.columns) != expected:
        raise ValueError(
            f"{source} probability columns/order differ. Expected: {expected}"
        )
    numeric = frame[PROBABILITY_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError(f"{source} contains non-numeric/blank probabilities")
    if ((numeric < -1e-7) | (numeric > 1 + 1e-7)).any().any():
        raise ValueError(f"{source} contains probabilities outside [0, 1]")
    if not np.allclose(numeric.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError(f"{source} probability rows do not sum to 1")


def validate_logit_table(frame: pd.DataFrame, source: Path) -> None:
    validate_ids(frame, EXPECTED_ROWS["validation"], source)
    expected = ["sample_id", *LOGIT_COLUMNS]
    if list(frame.columns) != expected:
        raise ValueError(f"{source} logit columns/order differ. Expected: {expected}")
    numeric = frame[LOGIT_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"{source} contains non-finite logits")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def summary_row(path: Path, reference: str) -> dict[str, Any]:
    frame = pd.read_csv(path)
    required = ["reference", "num_samples", "accuracy", "macro_f1"]
    require_columns(frame, required, path)
    selected = frame.loc[frame["reference"].astype(str) == reference]
    if len(selected) != 1:
        raise ValueError(f"{path} must contain exactly one reference={reference!r} row")
    return selected.iloc[0].to_dict()


def find_column(frame: pd.DataFrame, candidates: list[str], path: Path) -> str:
    lowered = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in lowered:
            return str(lowered[candidate])
    raise ValueError(f"None of columns {candidates} found in {path}; got {list(frame.columns)}")


def read_per_class(path: Path, majority: bool) -> pd.DataFrame:
    frame = pd.read_csv(path)
    label_col = find_column(
        frame, ["label", "class_label", "evasion_label", "class", "category"], path
    )
    f1_col = find_column(frame, ["f1", "f1_score", "class_f1"], path)
    frame = frame.rename(columns={label_col: "label", f1_col: "f1"})
    frame["label"] = frame["label"].astype(str)
    frame["f1"] = pd.to_numeric(frame["f1"], errors="coerce")
    if frame["f1"].isna().any():
        raise ValueError(f"{path} contains invalid per-class F1 values")
    if "reference" in frame.columns:
        desired = "supplementary_majority_vote" if majority else "annotator_mean"
        exact = frame.loc[frame["reference"].astype(str) == desired]
        if not exact.empty:
            frame = exact
        elif not majority:
            refs = {"annotator1", "annotator2", "annotator3"}
            available = frame["reference"].astype(str)
            if refs.issubset(set(available)):
                frame = frame.loc[available.isin(refs)].groupby("label", as_index=False)["f1"].mean()
    result = frame.groupby("label", as_index=False)["f1"].mean()
    unknown = sorted(set(result["label"]) - set(LABELS))
    if unknown:
        raise ValueError(f"{path} has unknown labels: {unknown}")
    missing = sorted(set(LABELS) - set(result["label"]))
    if missing:
        raise ValueError(f"{path} is missing labels: {missing}")
    return result


def validate_model_artifacts(paths: dict[str, Path]) -> dict[str, Any]:
    test_predictions = pd.read_csv(paths["test_predictions"])
    test_probabilities = pd.read_csv(paths["test_probabilities"])
    validation_predictions = pd.read_csv(paths["validation_predictions"])
    validation_probabilities = pd.read_csv(paths["validation_probabilities"])
    validation_logits = pd.read_csv(paths["validation_logits"])
    validate_ids(test_predictions, EXPECTED_ROWS["test"], paths["test_predictions"])
    validate_ids(
        validation_predictions,
        EXPECTED_ROWS["validation"],
        paths["validation_predictions"],
    )
    require_columns(test_predictions, ["predicted_label"], paths["test_predictions"])
    require_columns(
        validation_predictions,
        ["predicted_label"],
        paths["validation_predictions"],
    )
    if not set(test_predictions["predicted_label"].dropna()).issubset(LABELS):
        raise ValueError(f"Unknown predicted label in {paths['test_predictions']}")
    validate_probability_table(test_probabilities, EXPECTED_ROWS["test"], paths["test_probabilities"])
    validate_probability_table(
        validation_probabilities,
        EXPECTED_ROWS["validation"],
        paths["validation_probabilities"],
    )
    validate_logit_table(validation_logits, paths["validation_logits"])
    expected_test_ids = set(test_predictions["sample_id"].astype(str))
    expected_validation_ids = set(validation_predictions["sample_id"].astype(str))
    if set(test_probabilities["sample_id"].astype(str)) != expected_test_ids:
        raise ValueError("Test predictions and probabilities have different sample_id sets")
    for table in (validation_probabilities, validation_logits):
        if set(table["sample_id"].astype(str)) != expected_validation_ids:
            raise ValueError("Validation prediction/probability/logit sample_id sets differ")
    config = read_json(paths["experiment_config"])
    if config.get("labels") != LABELS:
        raise ValueError(f"Label order mismatch in {paths['experiment_config']}")
    required_config = [
        "model_id",
        "train_samples",
        "validation_samples",
        "test_samples",
        "random_seed",
        "max_length",
        "num_train_epochs",
        "learning_rate",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "best_validation_macro_f1",
        "formal_training_seconds",
        "formal_training_minutes",
        "data_source",
    ]
    missing_config = [key for key in required_config if key not in config]
    if missing_config:
        raise ValueError(
            f"{paths['experiment_config']} is missing required keys: {missing_config}"
        )
    expected_sizes = {"train_samples": 2700, "validation_samples": 748, "test_samples": 308}
    wrong_sizes = {
        key: config.get(key) for key, expected in expected_sizes.items() if config.get(key) != expected
    }
    if wrong_sizes:
        raise ValueError(f"Unexpected split sizes in config: {wrong_sizes}")
    if not nested(config, "data_source", "commit", default=""):
        raise ValueError(f"Data commit is missing in {paths['experiment_config']}")
    annotator = summary_row(paths["task2_summary"], "annotator_mean")
    majority = summary_row(paths["majority_summary"], "supplementary_majority_vote")
    if int(annotator["num_samples"]) != EXPECTED_ROWS["test"]:
        raise ValueError(f"Annotator summary must use 308 samples: {paths['task2_summary']}")
    if int(majority["num_samples"]) != 275:
        raise ValueError(f"Majority summary must use 275 samples: {paths['majority_summary']}")
    return {
        "test_predictions": test_predictions,
        "annotator": annotator,
        "majority": majority,
        "per_class": read_per_class(paths["task2_per_class"], majority=False),
        "majority_per_class": read_per_class(paths["majority_per_class"], majority=True),
        "config": config,
    }


def nested(config: dict[str, Any], *keys: str, default: Any = np.nan) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def build_results(validated: dict[str, dict[str, Any]], benchmark: pd.DataFrame | None) -> pd.DataFrame:
    benchmark_by_key = {}
    if benchmark is not None and not benchmark.empty:
        require_columns(benchmark, ["model_key"], Path("inference_benchmark.csv"))
        benchmark_by_key = benchmark.set_index("model_key").to_dict(orient="index")
    rows = []
    for key, data in validated.items():
        config = data["config"]
        annotator = data["annotator"]
        majority = data["majority"]
        timing = benchmark_by_key.get(key, {})
        train_batch_size = config.get("per_device_train_batch_size", np.nan)
        accumulation_steps = config.get("gradient_accumulation_steps", np.nan)
        visible_gpu_count = nested(config, "hardware", "visible_gpu_count", default=1)
        effective_train_batch_size = (
            float(train_batch_size) * float(accumulation_steps) * float(visible_gpu_count)
            if all(
                pd.notna(value)
                for value in [train_batch_size, accumulation_steps, visible_gpu_count]
            )
            else np.nan
        )
        rows.append(
            {
                "model_key": key,
                "model": MODEL_SPECS[key]["display"],
                "model_id": config.get("model_id", ""),
                "validation_macro_f1": config.get("best_validation_macro_f1", np.nan),
                "test_annotator_mean_num_samples": annotator.get("num_samples", np.nan),
                "test_annotator_mean_accuracy": annotator.get("accuracy", np.nan),
                "test_annotator_mean_macro_f1": annotator.get("macro_f1", np.nan),
                "test_annotator_mean_ece": annotator.get("ece", np.nan),
                "test_annotator_mean_brier_score": annotator.get("brier_score", np.nan),
                "test_majority_num_samples": majority.get("num_samples", np.nan),
                "test_majority_accuracy": majority.get("accuracy", np.nan),
                "test_majority_macro_f1": majority.get("macro_f1", np.nan),
                "test_majority_ece": majority.get("ece", np.nan),
                "test_majority_brier_score": majority.get("brier_score", np.nan),
                "training_seconds": config.get("formal_training_seconds", np.nan),
                "training_minutes": config.get("formal_training_minutes", np.nan),
                "inference_seconds": timing.get("mean_inference_seconds", np.nan),
                "inference_samples_per_second": timing.get("samples_per_second", np.nan),
                "inference_ms_per_sample": timing.get("milliseconds_per_sample", np.nan),
                "peak_gpu_memory_gib": timing.get("peak_gpu_memory_gib", np.nan),
                "gpu": nested(config, "hardware", "gpu", default=timing.get("gpu", "")),
                "vram_gib": nested(config, "hardware", "vram_gib"),
                "train_batch_size": train_batch_size,
                "eval_batch_size": config.get("per_device_eval_batch_size", np.nan),
                "gradient_accumulation_steps": accumulation_steps,
                "visible_gpu_count": visible_gpu_count,
                "effective_train_batch_size": effective_train_batch_size,
                "epochs": config.get("num_train_epochs", np.nan),
                "learning_rate": config.get("learning_rate", np.nan),
                "max_length": config.get("max_length", np.nan),
                "random_seed": config.get("random_seed", np.nan),
                "data_commit": nested(config, "data_source", "commit", default=""),
            }
        )
    return pd.DataFrame(rows)


def validate_cross_model_settings(results: pd.DataFrame) -> None:
    """Reject comparisons built from materially different experiment settings."""
    if len(results) < 2:
        return
    required_equal = [
        "data_commit",
        "effective_train_batch_size",
        "epochs",
        "learning_rate",
        "max_length",
        "random_seed",
    ]
    mismatches = {}
    for column in required_equal:
        values = results[column].dropna().astype(str).unique().tolist()
        if column == "data_commit" and "" in values:
            values.remove("")
        if len(values) > 1:
            mismatches[column] = values
    if mismatches:
        formatted = "; ".join(f"{key}={value}" for key, value in mismatches.items())
        raise ValueError(f"Cross-model settings are not comparable: {formatted}")


def create_figure(results: pd.DataFrame, per_class: pd.DataFrame, output: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    palette = ["#4C78A8", "#F58518"]

    metric_data = results.set_index("model")[[
        "test_annotator_mean_accuracy", "test_annotator_mean_macro_f1"
    ]].rename(columns={
        "test_annotator_mean_accuracy": "Accuracy",
        "test_annotator_mean_macro_f1": "Macro-F1",
    })
    metric_data.plot(kind="bar", ax=axes[0, 0], color=palette, width=0.75)
    axes[0, 0].set_title("Test performance: annotator mean (n=308)")
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylim(0, max(0.5, float(metric_data.max().max()) * 1.2))
    axes[0, 0].tick_params(axis="x", rotation=15)
    axes[0, 0].legend(loc="upper center", ncol=2)

    majority_data = results.set_index("model")[[
        "test_majority_accuracy", "test_majority_macro_f1"
    ]].rename(columns={
        "test_majority_accuracy": "Accuracy",
        "test_majority_macro_f1": "Macro-F1",
    })
    majority_data.plot(kind="bar", ax=axes[0, 1], color=palette, width=0.75)
    axes[0, 1].set_title("Supplementary majority-vote performance (n=275)")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylim(0, max(0.5, float(majority_data.max().max()) * 1.2))
    axes[0, 1].tick_params(axis="x", rotation=15)
    axes[0, 1].legend(loc="upper center", ncol=2)

    if not per_class.empty:
        heatmap = per_class.pivot(index="model", columns="label", values="f1").reindex(columns=LABELS)
        values = heatmap.to_numpy(dtype=float)
        image = axes[1, 0].imshow(
            values,
            cmap="Blues",
            vmin=0,
            vmax=max(0.5, float(np.nanmax(values))),
            aspect="auto",
        )
        colorbar = fig.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04)
        colorbar.set_label("F1")
        axes[1, 0].set_xticks(range(len(heatmap.columns)), heatmap.columns)
        axes[1, 0].set_yticks(range(len(heatmap.index)), heatmap.index)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                if np.isfinite(value):
                    axes[1, 0].text(
                        column,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color="white" if value > image.norm.vmax * 0.55 else "#1F2937",
                        fontsize=8,
                    )
        axes[1, 0].set_title("Per-class F1: annotator mean")
        axes[1, 0].set_xlabel("")
        axes[1, 0].set_ylabel("")
        axes[1, 0].tick_params(axis="x", rotation=45)
        axes[1, 0].tick_params(axis="y", rotation=0)
    else:
        axes[1, 0].text(0.5, 0.5, "Per-class results unavailable", ha="center", va="center")
        axes[1, 0].set_axis_off()

    efficiency = results.set_index("model")[["training_minutes", "inference_samples_per_second"]]
    efficiency["training_minutes"].plot(kind="bar", ax=axes[1, 1], color="#54A24B", width=0.55)
    axes[1, 1].set_title("Training time and inference throughput")
    axes[1, 1].set_ylabel("Training time (minutes)", color="#2E6E2E")
    axes[1, 1].set_xlabel("")
    axes[1, 1].tick_params(axis="x", rotation=15)
    throughput = efficiency["inference_samples_per_second"]
    if throughput.notna().any():
        right = axes[1, 1].twinx()
        right.plot(range(len(throughput)), throughput, color="#E45756", marker="o", linewidth=2)
        right.set_ylabel("Inference samples/second", color="#A62728")
    else:
        axes[1, 1].text(
            0.5,
            0.92,
            "Inference benchmark not supplied",
            transform=axes[1, 1].transAxes,
            ha="center",
            color="#A62728",
        )
    fig.suptitle("Task 2 nine-class baseline comparison", fontsize=16, fontweight="bold")
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_error_analysis(
    validated: dict[str, dict[str, Any]], test_file: Path
) -> tuple[pd.DataFrame, dict[str, list[tuple[str, int]]]]:
    test = pd.read_csv(test_file)
    require_columns(test, ["sample_id"], test_file)
    if len(test) != EXPECTED_ROWS["test"] or test["sample_id"].duplicated().any():
        raise ValueError("Test source must contain 308 unique sample_id rows")
    gold_candidates = ["evasion_label_majority", "gold_label", "evasion_label"]
    gold_col = next((column for column in gold_candidates if column in test.columns), None)
    if gold_col is None:
        return pd.DataFrame(), {}
    keep = ["sample_id", gold_col]
    for optional in ["question", "interview_answer", "majority_vote_count"]:
        if optional in test.columns:
            keep.append(optional)
    merged = test[keep].rename(columns={gold_col: "gold_label"}).copy()
    merged["sample_id"] = merged["sample_id"].astype(str)
    model_columns = []
    for key, data in validated.items():
        pred = data["test_predictions"][["sample_id", "predicted_label"]].copy()
        pred["sample_id"] = pred["sample_id"].astype(str)
        column = f"{key}_prediction"
        pred = pred.rename(columns={"predicted_label": column})
        merged = merged.merge(pred, on="sample_id", how="left", validate="one_to_one")
        model_columns.append(column)
    valid_majority = merged["gold_label"].isin(LABELS)
    if "majority_vote_count" in merged.columns:
        valid_majority &= pd.to_numeric(
            merged["majority_vote_count"], errors="coerce"
        ).ge(2)
    majority = merged.loc[valid_majority].copy()
    if majority.empty:
        return pd.DataFrame(), {}
    if len(majority) != 275:
        raise ValueError(
            f"Expected 275 valid majority-vote samples, received {len(majority)}"
        )
    majority["wrong_model_count"] = sum(
        majority[column].ne(majority["gold_label"]).astype(int) for column in model_columns
    )
    majority["prediction_disagreement_count"] = majority[model_columns].nunique(axis=1)
    errors = majority.loc[majority["wrong_model_count"] >= min(2, len(model_columns))].copy()
    errors = errors.sort_values(
        ["wrong_model_count", "prediction_disagreement_count"], ascending=False
    ).head(30)
    confusions: dict[str, list[tuple[str, int]]] = {}
    for key, column in zip(validated, model_columns):
        wrong = majority.loc[majority[column] != majority["gold_label"]]
        pairs = Counter(zip(wrong["gold_label"].astype(str), wrong[column].astype(str)))
        confusions[key] = [(f"{gold} → {pred}", count) for (gold, pred), count in pairs.most_common(5)]
    return errors, confusions


def best_model_sentence(results: pd.DataFrame, column: str, label: str, higher: bool = True) -> str:
    valid = results.loc[pd.to_numeric(results[column], errors="coerce").notna()]
    if valid.empty:
        return f"- {label}：没有可用数据。"
    row = valid.loc[valid[column].idxmax() if higher else valid[column].idxmin()]
    return f"- {label}最佳：**{row['model']}**（{float(row[column]):.4f}）。"


def write_analysis(
    output: Path,
    results: pd.DataFrame,
    per_class: pd.DataFrame,
    confusions: dict[str, list[tuple[str, int]]],
    benchmark_supplied: bool,
    complete: bool,
) -> None:
    lines = [
        "# Task 2 九分类 Baseline 比较分析",
        "",
        f"状态：{'正式完整比较' if complete else '不完整草稿（不得作为最终结论）'}。",
        "",
        "## 评价口径",
        "",
        "- 主要结果使用 A 组统一评价脚本的 `annotator_mean`：全部 308 个测试样本分别对三名标注者计算后取平均。",
        "- `supplementary_majority_vote` 仅覆盖具有多数标签的 275 个样本，作为补充结果，不能与 308 样本主结果混为一谈。",
        "- 模型选择只使用 Validation Set 的 Macro-F1；Test Set 未参与调参或 Temperature Scaling。",
        "",
        "## 总体结果",
        "",
        best_model_sentence(results, "test_annotator_mean_accuracy", "测试集 Accuracy"),
        best_model_sentence(results, "test_annotator_mean_macro_f1", "测试集 Macro-F1"),
        best_model_sentence(results, "test_majority_macro_f1", "多数标签 Macro-F1"),
        best_model_sentence(results, "training_minutes", "训练耗时最短", higher=False),
    ]
    if benchmark_supplied:
        lines.append(best_model_sentence(results, "inference_samples_per_second", "推理吞吐最高"))
    else:
        lines.append("- 推理时间和峰值显存尚未统一复测；相关列保留为空，不能用训练日志代替。")

    lines.extend(["", "## 各模型优缺点", ""])
    for key in results["model_key"]:
        spec = MODEL_SPECS[key]
        lines.extend(
            [
                f"### {spec['display']}",
                "",
                f"- 优点：{spec['advantages']}",
                f"- 局限：{spec['limitations']}",
            ]
        )
        model_pc = per_class.loc[per_class["model_key"] == key] if not per_class.empty else pd.DataFrame()
        if not model_pc.empty:
            worst = model_pc.nsmallest(3, "f1")
            description = "、".join(f"{row.label} ({row.f1:.3f})" for row in worst.itertuples())
            lines.append(f"- 最弱的三个类别：{description}。")
        pairs = confusions.get(key, [])
        if pairs:
            lines.append("- 高频混淆：" + "、".join(f"{pair}（{count}）" for pair, count in pairs) + "。")

    commits = sorted(set(results["data_commit"].dropna().astype(str)) - {""})
    lines.extend(
        [
            "",
            "## 公平性与限制",
            "",
            f"- 数据提交版本：{', '.join(commits) if commits else '配置文件未记录'}。",
            "- 三个模型必须使用相同的九分类顺序、数据划分、随机种子和评价脚本；本程序已检查可机读字段。",
            "- 典型错误表优先使用多数标签；没有多数标签的测试样本不参与该部分混淆统计。",
            "- Validation logits/probabilities 只交给 C 在验证集拟合温度参数，严禁在 Test Set 上拟合。",
            "",
            "## 生成文件",
            "",
            "- `baseline_results.csv`：一行一个模型的汇总指标、训练配置和资源记录。",
            "- `baseline_comparison.png`：主结果、补充多数标签结果、逐类 F1 和效率比较。",
            "- `baseline_typical_errors.csv`：至少两个模型预测错误的代表样本（最多 30 条）。",
            "- `baseline_input_audit.csv`：全部输入文件的存在性与大小检查。",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit, resolved = audit_inputs(args.inputs_root, args.test_file, args.benchmark_file)
    audit_path = args.output_dir / "baseline_input_audit.csv"
    audit.to_csv(audit_path, index=False)

    invalid_statuses = audit.loc[~audit["status"].astype(str).str.startswith(("present", "duplicate"))]
    model_complete = {
        key: len(paths) == len(expected_filenames(MODEL_SPECS[key]["prefix"]))
        for key, paths in resolved.items()
    }
    if (not args.test_file.exists() or not all(model_complete.values())) and not args.allow_incomplete:
        missing = audit.loc[~audit["status"].astype(str).str.startswith(("present", "duplicate")), [
            "model_key", "expected_filename", "status"
        ]]
        print(missing.to_string(index=False))
        raise SystemExit(
            f"Input audit failed. See {audit_path}. Collect all artifacts, then rerun."
        )

    validated: dict[str, dict[str, Any]] = {}
    for key in MODEL_SPECS:
        if not model_complete[key]:
            continue
        validated[key] = validate_model_artifacts(resolved[key])
    if not validated:
        raise SystemExit(f"No complete model artifact set found. See {audit_path}")

    # Enforce identical sample order as an additional handover guarantee.
    prediction_orders = [
        data["test_predictions"]["sample_id"].astype(str).tolist()
        for data in validated.values()
    ]
    if len(prediction_orders) > 1 and any(order != prediction_orders[0] for order in prediction_orders[1:]):
        raise ValueError("Test sample_id order differs between models; realign before comparison")

    benchmark = None
    if args.benchmark_file and args.benchmark_file.exists():
        benchmark = pd.read_csv(args.benchmark_file)
    results = build_results(validated, benchmark)
    validate_cross_model_settings(results)
    results_path = args.output_dir / "baseline_results.csv"
    results.to_csv(results_path, index=False, float_format="%.10g")

    per_class_rows = []
    for key, data in validated.items():
        table = data["per_class"].copy()
        table.insert(0, "model", MODEL_SPECS[key]["display"])
        table.insert(0, "model_key", key)
        per_class_rows.append(table)
    per_class = pd.concat(per_class_rows, ignore_index=True) if per_class_rows else pd.DataFrame()
    if not per_class.empty:
        per_class.to_csv(args.output_dir / "baseline_per_class_results.csv", index=False)

    errors, confusions = build_error_analysis(validated, args.test_file)
    errors.to_csv(args.output_dir / "baseline_typical_errors.csv", index=False)
    create_figure(results, per_class, args.output_dir / "baseline_comparison.png")
    complete = all(model_complete.values()) and len(validated) == len(MODEL_SPECS)
    write_analysis(
        args.output_dir / "baseline_analysis.md",
        results,
        per_class,
        confusions,
        benchmark_supplied=benchmark is not None,
        complete=complete,
    )

    print("Baseline comparison created:")
    for path in sorted(args.output_dir.iterdir()):
        print(f"- {path.name} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
