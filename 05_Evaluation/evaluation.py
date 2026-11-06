
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score
)


# ============================================================
# 1. Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "01_Data"
SPLIT_DIR = DATA_DIR / "data_splits"

OUTPUT_DIR = ROOT / "05_Evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TASK1_LABELS = [
    "Clear Reply",
    "Ambivalent",
    "Clear Non-Reply"
]

TASK2_LABELS = [
    "Explicit",
    "Implicit",
    "Dodging",
    "General",
    "Deflection",
    "Declining to answer",
    "Claims ignorance",
    "Clarification",
    "Partial/half-answer"
]

ANNOTATOR_COLUMNS = [
    "annotator1",
    "annotator2",
    "annotator3"
]

ECE_BINS = 15

TASK1_PROBABILITY_COLUMNS = [
    "p_Clear_Reply",
    "p_Ambivalent",
    "p_Clear_Non_Reply"
]

TASK2_PROBABILITY_COLUMNS = [
    "p_Explicit",
    "p_Implicit",
    "p_Dodging",
    "p_General",
    "p_Deflection",
    "p_Declining_to_answer",
    "p_Claims_ignorance",
    "p_Clarification",
    "p_Partial_half_answer"
]


# ============================================================
# 2. Data alignment and validation
# ============================================================

def load_test_data():
    """Load the shared test set."""

    test_path = SPLIT_DIR / "test.csv"

    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    test_df = pd.read_csv(test_path)

    if "sample_id" not in test_df.columns:
        raise ValueError("Test set is missing sample_id.")

    if test_df["sample_id"].isna().any():
        raise ValueError("Test set contains missing sample_id.")

    if not test_df["sample_id"].is_unique:
        raise ValueError("Test set contains duplicate sample_id.")

    return test_df


def align_probability_file(test_df, probability_file, probability_columns):
    """
    Align probability rows to test_df using sample_id.

    The input probability CSV must contain:
    sample_id + the required probability columns.

    Different row orders are allowed, but IDs must match exactly.
    """

    pred_df = pd.read_csv(probability_file)

    required_columns = ["sample_id"] + probability_columns

    missing = [
        col for col in required_columns
        if col not in pred_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in probability file: {missing}"
        )

    if pred_df["sample_id"].isna().any():
        raise ValueError("Probability file contains missing sample_id.")

    if not pred_df["sample_id"].is_unique:
        raise ValueError("Probability file contains duplicate sample_id.")

    test_ids = set(test_df["sample_id"])
    prediction_ids = set(pred_df["sample_id"])

    if test_ids != prediction_ids:
        missing_ids = test_ids - prediction_ids
        extra_ids = prediction_ids - test_ids

        raise ValueError(
            "sample_id mismatch between test set and predictions. "
            f"Missing IDs: {len(missing_ids)}; "
            f"Extra IDs: {len(extra_ids)}."
        )

    # Reorder predictions to match the test set exactly.
    pred_df = (
        pred_df.set_index("sample_id")
        .loc[test_df["sample_id"]]
        .reset_index()
    )

    probabilities = pred_df[probability_columns].to_numpy(dtype=float)

    return pred_df, probabilities


def validate_prediction_file(
    test_df,
    probability_df,
    probabilities,
    prediction_file,
    labels
):
    """
    Optionally validate a separate predicted-label CSV.

    Required columns:
    sample_id, predicted_label

    Its rows are aligned by sample_id.
    Predicted labels must agree with probability argmax.
    """

    if prediction_file is None:
        return

    label_df = pd.read_csv(prediction_file)

    required = ["sample_id", "predicted_label"]

    missing = [col for col in required if col not in label_df.columns]

    if missing:
        raise ValueError(
            f"Missing columns in prediction file: {missing}"
        )

    if label_df["sample_id"].isna().any():
        raise ValueError("Prediction file contains missing sample_id.")

    if not label_df["sample_id"].is_unique:
        raise ValueError("Prediction file contains duplicate sample_id.")

    test_ids = set(test_df["sample_id"])

    if set(label_df["sample_id"]) != test_ids:
        raise ValueError(
            "sample_id mismatch between test set and prediction file."
        )

    label_df = (
        label_df.set_index("sample_id")
        .loc[test_df["sample_id"]]
        .reset_index()
    )

    unknown = set(label_df["predicted_label"].dropna()) - set(labels)

    if unknown:
        raise ValueError(f"Unknown predicted labels: {unknown}")

    expected_predictions = np.asarray(labels)[
        np.argmax(probabilities, axis=1)
    ]

    actual_predictions = label_df["predicted_label"].to_numpy()

    if not np.array_equal(actual_predictions, expected_predictions):
        raise ValueError(
            "predicted_label does not match probability argmax."
        )


# ============================================================
# 3. Metric functions
# ============================================================

def calculate_ece(y_true, probabilities, labels, n_bins=15):
    """Calculate top-label Expected Calibration Error."""

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    confidence = np.max(probabilities, axis=1)
    predictions = np.asarray(labels)[
        np.argmax(probabilities, axis=1)
    ]

    correct = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (
                (confidence >= bin_edges[i]) &
                (confidence <= bin_edges[i + 1])
            )
        else:
            mask = (
                (confidence >= bin_edges[i]) &
                (confidence < bin_edges[i + 1])
            )

        if np.any(mask):
            accuracy_in_bin = np.mean(correct[mask])
            confidence_in_bin = np.mean(confidence[mask])

            ece += (
                np.sum(mask) / len(y_true)
            ) * abs(accuracy_in_bin - confidence_in_bin)

    return float(ece)


def calculate_brier_score(y_true, probabilities, labels):
    """
    Multiclass Brier score:
    mean of the sum of squared errors across classes per sample.
    """

    label_to_index = {
        label: i for i, label in enumerate(labels)
    }

    y_indices = np.array([
        label_to_index[label] for label in y_true
    ])

    one_hot = np.zeros(
        (len(y_true), len(labels)),
        dtype=float
    )

    one_hot[np.arange(len(y_true)), y_indices] = 1.0

    return float(
        np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
    )


def evaluate_predictions(
    y_true,
    probabilities,
    labels,
    task_name,
    reference_name,
    model_name
):
    """Evaluate one prediction system against one reference."""

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    if probabilities.ndim != 2:
        raise ValueError("Probabilities must be a 2D array.")

    if probabilities.shape != (len(y_true), len(labels)):
        raise ValueError(
            f"Expected probability shape "
            f"({len(y_true)}, {len(labels)}), "
            f"got {probabilities.shape}."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities contain NaN or infinity.")

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Probabilities must be between 0 and 1.")

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-5
    ):
        raise ValueError("Each probability row must sum to 1.")

    unknown = set(y_true) - set(labels)

    if unknown:
        raise ValueError(f"Unknown true labels: {unknown}")

    predictions = np.asarray(labels)[
        np.argmax(probabilities, axis=1)
    ]

    accuracy = accuracy_score(y_true, predictions)

    macro_f1 = f1_score(
        y_true,
        predictions,
        labels=labels,
        average="macro",
        zero_division=0
    )

    precision, recall, per_class_f1, support = (
        precision_recall_fscore_support(
            y_true,
            predictions,
            labels=labels,
            zero_division=0
        )
    )

    ece = calculate_ece(
        y_true,
        probabilities,
        labels,
        ECE_BINS
    )

    brier = calculate_brier_score(
        y_true,
        probabilities,
        labels
    )

    summary = {
        "model": model_name,
        "task": task_name,
        "reference": reference_name,
        "num_samples": len(y_true),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "ece": ece,
        "brier_score": brier
    }

    per_class = pd.DataFrame({
        "model": model_name,
        "task": task_name,
        "reference": reference_name,
        "class": labels,
        "precision": precision,
        "recall": recall,
        "f1": per_class_f1,
        "support": support
    })

    return summary, per_class


# ============================================================
# 4. Task 1: Three-class clarity evaluation
# ============================================================

def evaluate_task1(
    test_df,
    probability_file,
    model_name="unspecified",
    prediction_file=None
):
    """
    Evaluate Task 1 against clarity_label.

    Probability CSV:
    sample_id + three probability columns.

    Optional prediction CSV:
    sample_id + predicted_label.
    """

    probability_df, probabilities = align_probability_file(
        test_df,
        probability_file,
        TASK1_PROBABILITY_COLUMNS
    )

    validate_prediction_file(
        test_df,
        probability_df,
        probabilities,
        prediction_file,
        TASK1_LABELS
    )

    if "clarity_label" not in test_df.columns:
        raise ValueError("Test set is missing clarity_label.")

    y_true = test_df["clarity_label"]

    if y_true.isna().any():
        raise ValueError("Missing clarity labels in test set.")

    return evaluate_predictions(
        y_true.to_numpy(),
        probabilities,
        TASK1_LABELS,
        "Task 1",
        "original_clarity_label",
        model_name
    )


# ============================================================
# 5. Task 2: Evaluation against each annotator
# ============================================================

def evaluate_task2_annotators(
    test_df,
    probability_file,
    model_name="unspecified",
    prediction_file=None
):
    """
    Evaluate Task 2 independently against annotator1/2/3.

    Probability CSV:
    sample_id + nine probability columns.

    Optional prediction CSV:
    sample_id + predicted_label.
    """

    probability_df, probabilities = align_probability_file(
        test_df,
        probability_file,
        TASK2_PROBABILITY_COLUMNS
    )

    validate_prediction_file(
        test_df,
        probability_df,
        probabilities,
        prediction_file,
        TASK2_LABELS
    )

    summaries = []
    class_tables = []

    for annotator in ANNOTATOR_COLUMNS:

        if annotator not in test_df.columns:
            raise ValueError(f"Missing annotation column: {annotator}")

        y_true = test_df[annotator]

        if y_true.isna().any():
            raise ValueError(f"Missing labels in {annotator}.")

        summary, per_class = evaluate_predictions(
            y_true.to_numpy(),
            probabilities,
            TASK2_LABELS,
            "Task 2",
            annotator,
            model_name
        )

        summaries.append(summary)
        class_tables.append(per_class)

    # Descriptive mean and standard deviation across annotators.
    metric_columns = [
        "accuracy",
        "macro_f1",
        "ece",
        "brier_score"
    ]

    summary_df = pd.DataFrame(summaries)

    mean_row = {
        "model": model_name,
        "task": "Task 2",
        "reference": "annotator_mean",
        "num_samples": int(summary_df["num_samples"].mean())
    }

    std_row = {
        "model": model_name,
        "task": "Task 2",
        "reference": "annotator_std",
        "num_samples": int(summary_df["num_samples"].std(ddof=0))
    }

    for metric in metric_columns:
        mean_row[metric] = float(summary_df[metric].mean())
        std_row[metric] = float(summary_df[metric].std(ddof=0))

    summaries.extend([mean_row, std_row])

    return summaries, class_tables


# ============================================================
# 6. Task 2: Supplementary majority-vote evaluation
# ============================================================

def evaluate_task2_majority(
    test_df,
    probability_file,
    model_name="unspecified",
    prediction_file=None
):
    """
    Supplementary evaluation against majority-vote labels.

    Samples labelled No_Majority are excluded from this
    supplementary analysis only.

    They remain included in the three-annotator evaluation.
    """

    if "evasion_label_majority" not in test_df.columns:
        raise ValueError(
            "Missing evasion_label_majority in test set."
        )

    probability_df, probabilities = align_probability_file(
        test_df,
        probability_file,
        TASK2_PROBABILITY_COLUMNS
    )

    validate_prediction_file(
        test_df,
        probability_df,
        probabilities,
        prediction_file,
        TASK2_LABELS
    )

    majority = test_df["evasion_label_majority"]
    mask = majority.isin(TASK2_LABELS).to_numpy()

    if mask.sum() == 0:
        raise ValueError("No valid majority-vote samples found.")

    return evaluate_predictions(
        majority[mask].to_numpy(),
        probabilities[mask],
        TASK2_LABELS,
        "Task 2",
        "supplementary_majority_vote",
        model_name
    )


# ============================================================
# 7. Save evaluation results
# ============================================================

def save_results(summaries, class_tables, prefix):
    """
    Save model-level summary metrics and per-class metrics.

    This produces individual evaluation outputs.
    D remains responsible for the group's consolidated
    all_results.csv.
    """

    summary_df = pd.DataFrame(summaries)

    summary_path = OUTPUT_DIR / f"{prefix}_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    class_path = None

    if class_tables:
        per_class_df = pd.concat(
            class_tables,
            ignore_index=True
        )

        class_path = OUTPUT_DIR / f"{prefix}_per_class.csv"
        per_class_df.to_csv(class_path, index=False)

    print("\nEvaluation summary:")
    print(summary_df.to_string(index=False))

    print(f"\nSaved summary: {summary_path}")

    if class_path is not None:
        print(f"Saved per-class results: {class_path}")


# ============================================================
# 8. Main entry
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("UNIFIED EVALUATION MODULE")
    print("=" * 60)

    print("\nThis module provides reusable evaluation functions.")

    print("\nShared test set:")
    print(SPLIT_DIR / "test.csv")

    print("\nRequired probability file formats:")

    print("\nTask 1:")
    print("sample_id, p_Clear_Reply, p_Ambivalent, p_Clear_Non_Reply")

    print("\nTask 2:")
    print("sample_id, " + ", ".join(TASK2_PROBABILITY_COLUMNS))

    print("\nOptional separate prediction file:")
    print("sample_id, predicted_label")

    print("\nTask 2 references:")
    print("annotator1, annotator2, annotator3")
    print("supplementary_majority_vote")

    print("\nNo model predictions have been evaluated yet.")
