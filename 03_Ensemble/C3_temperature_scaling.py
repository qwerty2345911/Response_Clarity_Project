
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

INPUT_ROOT = (
    PROJECT_ROOT
    / "02_Baselines"
    / "member_b_task2_handover"
    / "model_artifacts"
)

OUTPUT_DIR = BASE_DIR / "results" / "C3_temperature_scaling"
ARTIFACT_DIR = BASE_DIR / "artifact"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

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

N_BINS = 15


def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def expected_calibration_error(probs, y_true, n_bins=N_BINS):
    predictions = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    accuracies = (predictions == y_true).astype(float)

    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == 0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences > lower) & (confidences <= upper)

        if np.any(mask):
            bin_accuracy = accuracies[mask].mean()
            bin_confidence = confidences[mask].mean()

            ece += (
                mask.sum() / len(y_true)
            ) * abs(bin_accuracy - bin_confidence)

    return float(ece)


def multiclass_brier_score(probs, y_true):
    one_hot = np.eye(len(LABELS))[y_true]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def negative_log_likelihood(probs, y_true):
    selected_probs = probs[np.arange(len(y_true)), y_true]
    selected_probs = np.clip(selected_probs, 1e-12, 1.0)

    return float(-np.mean(np.log(selected_probs)))


def calculate_metrics(probs, y_true):
    predictions = np.argmax(probs, axis=1)

    return {
        "accuracy": float(np.mean(predictions == y_true)),
        "ece": expected_calibration_error(probs, y_true),
        "brier_score": multiclass_brier_score(probs, y_true),
        "nll": negative_log_likelihood(probs, y_true),
    }


def load_validation_data(model):
    folder = INPUT_ROOT / model / "predictions"

    logits_path = folder / f"{model}_validation_logits.csv"
    predictions_path = folder / f"{model}_validation_predictions.csv"

    if not logits_path.exists():
        raise FileNotFoundError(logits_path)

    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)

    logits_df = pd.read_csv(logits_path)
    predictions_df = pd.read_csv(predictions_path)

    required_logits = {"sample_id", *LOGIT_COLS}
    required_predictions = {
        "sample_id",
        "gold_label",
        "predicted_label"
    }

    if not required_logits.issubset(logits_df.columns):
        raise ValueError(f"Missing logit columns for {model}")

    if not required_predictions.issubset(predictions_df.columns):
        raise ValueError(f"Missing validation prediction columns for {model}")

    logits_df["sample_id"] = logits_df["sample_id"].astype(str)
    predictions_df["sample_id"] = predictions_df["sample_id"].astype(str)

    if logits_df["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate validation IDs in {model} logits")

    if predictions_df["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate validation IDs in {model} predictions")

    if set(logits_df["sample_id"]) != set(predictions_df["sample_id"]):
        raise ValueError(f"Validation sample ID mismatch for {model}")

    predictions_df = (
        predictions_df
        .set_index("sample_id")
        .loc[logits_df["sample_id"]]
        .reset_index()
    )

    if not predictions_df["gold_label"].isin(LABELS).all():
        raise ValueError(f"Unexpected validation labels in {model}")

    logits = logits_df[LOGIT_COLS].to_numpy(dtype=float)

    if not np.isfinite(logits).all():
        raise ValueError(f"Non-finite logits in {model}")

    y_true = predictions_df["gold_label"].map(
        {label: i for i, label in enumerate(LABELS)}
    ).to_numpy(dtype=int)

    return logits, y_true


def load_test_probabilities(model):
    path = (
        INPUT_ROOT
        / model
        / "predictions"
        / f"{model}_probabilities.csv"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    data = pd.read_csv(path)
    data["sample_id"] = data["sample_id"].astype(str)

    if data["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate test IDs in {model}")

    if not {"sample_id", *PROB_COLS}.issubset(data.columns):
        raise ValueError(f"Missing test probability columns in {model}")

    probs = data[PROB_COLS].to_numpy(dtype=float)

    if not np.isfinite(probs).all():
        raise ValueError(f"Non-finite test probabilities in {model}")

    if (probs < 0).any() or (probs > 1).any():
        raise ValueError(f"Invalid test probabilities in {model}")

    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError(f"Test probability sums are invalid in {model}")

    return data, probs


def fit_temperature(logits, y_true):
    def objective(temperature):
        scaled_probs = softmax(logits / temperature)
        return negative_log_likelihood(scaled_probs, y_true)

    result = minimize_scalar(
        objective,
        bounds=(0.05, 10.0),
        method="bounded"
    )

    if not result.success:
        raise RuntimeError("Temperature optimization failed")

    if result.x < 0.051 or result.x > 9.999:
        print(
            f"Warning: Temperature {result.x:.4f} "
            "is close to the optimization boundary."
        )

    return float(result.x)


def plot_calibration_comparison(results_df):
    metrics = [
        ("ece", "Expected Calibration Error (ECE)"),
        ("brier_score", "Brier Score"),
        ("nll", "Negative Log-Likelihood (NLL)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x = np.arange(len(MODELS))
    width = 0.35

    for ax, (metric, title) in zip(axes, metrics):
        original = results_df[f"original_{metric}"]
        calibrated = results_df[f"calibrated_{metric}"]

        ax.bar(
            x - width / 2,
            original,
            width,
            label="Before Calibration"
        )

        ax.bar(
            x + width / 2,
            calibrated,
            width,
            label="After Calibration"
        )

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=15)
        ax.set_ylabel("Score")
        ax.legend()

    fig.suptitle("Temperature Scaling: Calibration Comparison")
    fig.tight_layout()

    output_path = OUTPUT_DIR / "calibration_comparison.png"

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Calibration comparison plot: {output_path}")


def main():
    calibration_results = []
    temperature_parameters = {}
    calibrated_test_probabilities = {}

    test_sample_ids = None

    for model in MODELS:
        print(f"Processing {model}...")

        logits, y_true = load_validation_data(model)

        original_val_probs = softmax(logits)

        temperature = fit_temperature(logits, y_true)

        calibrated_val_probs = softmax(logits / temperature)

        original_metrics = calculate_metrics(
            original_val_probs,
            y_true
        )

        calibrated_metrics = calculate_metrics(
            calibrated_val_probs,
            y_true
        )

        calibration_results.append({
            "model": model,
            "temperature": temperature,
            "original_accuracy": original_metrics["accuracy"],
            "calibrated_accuracy": calibrated_metrics["accuracy"],
            "original_ece": original_metrics["ece"],
            "calibrated_ece": calibrated_metrics["ece"],
            "original_brier_score": original_metrics["brier_score"],
            "calibrated_brier_score": calibrated_metrics["brier_score"],
            "original_nll": original_metrics["nll"],
            "calibrated_nll": calibrated_metrics["nll"],
        })

        temperature_parameters[model] = temperature

        test_df, test_probs = load_test_probabilities(model)

        current_ids = test_df["sample_id"].to_numpy()

        if test_sample_ids is None:
            test_sample_ids = current_ids
        elif not np.array_equal(test_sample_ids, current_ids):
            raise ValueError(f"Test sample ID mismatch for {model}")

        calibrated_test_probs = np.power(
            np.clip(test_probs, 1e-12, 1.0),
            1.0 / temperature
        )

        calibrated_test_probs /= calibrated_test_probs.sum(
            axis=1,
            keepdims=True
        )

        calibrated_test_probabilities[model] = calibrated_test_probs

        print(f"Temperature: {temperature:.4f}")

    results_df = pd.DataFrame(calibration_results)

    results_path = OUTPUT_DIR / "calibration_results.csv"
    results_df.to_csv(results_path, index=False)

    plot_calibration_comparison(results_df)

    parameter_path = ARTIFACT_DIR / "temperature_parameters.json"

    with open(parameter_path, "w", encoding="utf-8") as file:
        json.dump(
            temperature_parameters,
            file,
            indent=4
        )

    for model in MODELS:
        output = pd.DataFrame(
            calibrated_test_probabilities[model],
            columns=PROB_COLS
        )

        output.insert(0, "sample_id", test_sample_ids)

        output_path = (
            OUTPUT_DIR
            / f"{model}_calibrated_test_probabilities.csv"
        )

        output.to_csv(output_path, index=False)

    print("Temperature Scaling completed.")
    print(f"Calibration results: {results_path}")
    print(f"Temperature parameters: {parameter_path}")
    print(f"Calibrated test probabilities saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
