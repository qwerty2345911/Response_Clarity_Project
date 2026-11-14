
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


# 1. Project paths and fixed class order

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HANDOVER_ROOT = (
    PROJECT_ROOT
    / "02_Baselines"
    / "member_b_task2_handover"
    / "model_artifacts"
)

OUTPUT_DIR = Path(__file__).resolve().parent

MODELS = ["roberta", "deberta_v3", "xlnet"]

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

PROB_COLS = [
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

LOGIT_COLS = [
    col.replace("p_", "logit_")
    for col in PROB_COLS
]


# 2. Data loading

def model_file(model, filename):
    return (
        HANDOVER_ROOT
        / model
        / "predictions"
        / filename
    )


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    return pd.read_csv(path)


def validate_ids(frames, context):
    reference = frames[0]["sample_id"].astype(str)

    if reference.duplicated().any():
        raise ValueError(
            f"Duplicate sample IDs in {context}."
        )

    reference_set = set(reference)

    for frame in frames[1:]:
        ids = frame["sample_id"].astype(str)

        if ids.duplicated().any():
            raise ValueError(
                f"Duplicate sample IDs in {context}."
            )

        if set(ids) != reference_set:
            raise ValueError(
                f"Sample ID mismatch in {context}."
            )


def reorder_by_sample_id(frame, sample_ids):
    frame = frame.copy()

    frame["sample_id"] = (
        frame["sample_id"].astype(str)
    )

    return (
        frame.set_index("sample_id")
        .loc[sample_ids]
        .reset_index()
    )


def load_test_predictions_and_probabilities():

    predictions = []
    probabilities = []

    for model in MODELS:

        pred = load_csv(
            model_file(
                model,
                f"{model}_predictions.csv"
            )
        )

        prob = load_csv(
            model_file(
                model,
                f"{model}_probabilities.csv"
            )
        )

        predictions.append(pred)
        probabilities.append(prob)

    validate_ids(
        predictions,
        "Test predictions"
    )

    validate_ids(
        probabilities,
        "Test probabilities"
    )

    validate_ids(
        predictions + probabilities,
        "Test prediction/probability files"
    )

    sample_ids = (
        predictions[0]["sample_id"]
        .astype(str)
        .tolist()
    )

    predictions = [
        reorder_by_sample_id(frame, sample_ids)
        for frame in predictions
    ]

    probabilities = [
        reorder_by_sample_id(frame, sample_ids)
        for frame in probabilities
    ]

    return predictions, probabilities


def load_validation_data(model):

    pred = load_csv(
        model_file(
            model,
            f"{model}_validation_predictions.csv"
        )
    )

    logits = load_csv(
        model_file(
            model,
            f"{model}_validation_logits.csv"
        )
    )

    probs = load_csv(
        model_file(
            model,
            f"{model}_validation_probabilities.csv"
        )
    )

    validate_ids(
        [pred, logits, probs],
        f"{model} validation files"
    )

    sample_ids = (
        pred["sample_id"]
        .astype(str)
        .tolist()
    )

    logits = reorder_by_sample_id(
        logits, sample_ids
    )

    probs = reorder_by_sample_id(
        probs, sample_ids
    )

    return pred, logits, probs



# 3. Validation utilities

def validate_labels(values, context):

    invalid = set(values) - set(LABELS)

    if invalid:
        raise ValueError(
            f"Invalid labels in {context}: {invalid}"
        )


def probability_matrix(frame, context):

    probs = frame[PROB_COLS].to_numpy(
        dtype=float
    )

    if not np.isfinite(probs).all():
        raise ValueError(
            f"Non-finite probabilities in {context}"
        )

    if (probs < 0).any() or (probs > 1).any():
        raise ValueError(
            f"Probabilities outside [0, 1] in {context}"
        )

    if not np.allclose(
        probs.sum(axis=1),
        1.0,
        atol=1e-5
    ):
        raise ValueError(
            f"Probability sums are invalid in {context}"
        )

    return probs


def softmax(logits):

    logits = np.asarray(
        logits,
        dtype=float
    )

    logits = (
        logits
        - logits.max(axis=1, keepdims=True)
    )

    exp_logits = np.exp(logits)

    return (
        exp_logits
        / exp_logits.sum(axis=1, keepdims=True)
    )


def temperature_scale(probs, temperature):

    if temperature <= 0:
        raise ValueError(
            "Temperature must be positive."
        )

    probs = np.clip(
        probs,
        1e-15,
        1.0
    )

    scaled = probs ** (1.0 / temperature)

    return (
        scaled
        / scaled.sum(axis=1, keepdims=True)
    )


# 4. Calibration metrics

def expected_calibration_error(
    y_true,
    probs,
    n_bins=15
):

    y_true = np.asarray(y_true)

    predictions = probs.argmax(axis=1)

    confidence = probs.max(axis=1)

    correct = np.array([
        LABELS.index(label)
        for label in y_true
    ]) == predictions

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1
    )

    ece = 0.0

    for i in range(n_bins):

        if i == n_bins - 1:
            mask = (
                (confidence >= edges[i])
                & (confidence <= edges[i + 1])
            )
        else:
            mask = (
                (confidence >= edges[i])
                & (confidence < edges[i + 1])
            )

        if mask.any():

            bin_accuracy = correct[mask].mean()

            bin_confidence = confidence[mask].mean()

            ece += (
                mask.mean()
                * abs(
                    bin_accuracy - bin_confidence
                )
            )

    return float(ece)


def multiclass_brier_score(y_true, probs):

    indices = np.array([
        LABELS.index(label)
        for label in y_true
    ])

    one_hot = np.eye(
        len(LABELS)
    )[indices]

    return float(
        np.mean(
            np.sum(
                (probs - one_hot) ** 2,
                axis=1
            )
        )
    )


def calculate_metrics(y_true, probs):

    predicted = np.asarray(LABELS)[
        probs.argmax(axis=1)
    ]

    return {
        "Accuracy": accuracy_score(
            y_true,
            predicted
        ),

        "Macro-F1": f1_score(
            y_true,
            predicted,
            labels=LABELS,
            average="macro",
            zero_division=0
        ),

        "ECE": expected_calibration_error(
            y_true,
            probs
        ),

        "Brier Score": multiclass_brier_score(
            y_true,
            probs
        ),
    }


# 5. Test ground-truth labels

def load_test_gold():

    path = (
        PROJECT_ROOT
        / "01_Data"
        / "data_splits"
        / "test.csv"
    )

    data = load_csv(path)

    gold = data[
        ["sample_id", "evasion_label_majority"]
    ].copy()

    gold.columns = [
        "sample_id",
        "gold_label"
    ]

    gold["sample_id"] = (
        gold["sample_id"].astype(str)
    )

    # Exclude annotation-status labels.
    valid_mask = gold["gold_label"].isin(
        LABELS
    )

    excluded = gold.loc[
        ~valid_mask
    ].copy()

    valid_gold = gold.loc[
        valid_mask
    ].copy()

    return valid_gold, excluded
