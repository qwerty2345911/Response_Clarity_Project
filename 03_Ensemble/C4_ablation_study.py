
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

INPUT_ROOT = (
    PROJECT_ROOT
    / "02_Baselines"
    / "member_b_task2_handover"
    / "model_artifacts"
)

DATA_ROOT = PROJECT_ROOT / "01_Data" / "data_splits"

RESULTS_DIR = BASE_DIR / "results" 
OUTPUT_DIR = RESULTS_DIR / "C4_ablation_study"
OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
DOCS_DIR = BASE_DIR / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

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

LABEL_TO_ID = {
    label: i for i, label in enumerate(LABELS)
}

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

N_BINS = 15


def validate_probabilities(probs, name):
    if probs.ndim != 2 or probs.shape[1] != len(LABELS):
        raise ValueError(f"Invalid probability shape: {name}")

    if not np.isfinite(probs).all():
        raise ValueError(f"Non-finite probabilities: {name}")

    if (probs < 0).any() or (probs > 1).any():
        raise ValueError(f"Probabilities outside [0, 1]: {name}")

    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError(f"Probability rows do not sum to 1: {name}")


def load_csv_with_ids(path, required_columns):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if not set(required_columns).issubset(df.columns):
        raise ValueError(
            f"Missing columns in {path.name}: "
            f"{set(required_columns) - set(df.columns)}"
        )

    df["sample_id"] = df["sample_id"].astype(str)

    if df["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample IDs in {path}")

    return df


def align_dataframe(df, sample_ids, name):
    indexed = df.set_index("sample_id")

    missing_ids = set(sample_ids) - set(indexed.index)

    if missing_ids:
        raise ValueError(
            f"{name} is missing {len(missing_ids)} test sample IDs."
        )

    return indexed.loc[sample_ids].reset_index()


def load_test_data():
    path = DATA_ROOT / "test.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {"sample_id", "evasion_label_majority"}

    if not required.issubset(df.columns):
        raise ValueError("test.csv is missing required columns.")

    df["sample_id"] = df["sample_id"].astype(str)

    if df["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample IDs in test.csv")

    labels = df["evasion_label_majority"]

    valid_mask = labels.isin(LABELS)

    valid_df = df.loc[valid_mask].copy()
    excluded_df = df.loc[~valid_mask].copy()

    if valid_df.empty:
        raise ValueError("No valid test labels were found.")

    y_true = (
        valid_df["evasion_label_majority"]
        .map(LABEL_TO_ID)
        .to_numpy(dtype=int)
    )

    exclusion_labels = (
        excluded_df["evasion_label_majority"]
        .fillna("Missing_Annotation")
        .astype(str)
        .value_counts()
        .rename_axis("excluded_label")
        .reset_index(name="sample_count")
    )

    exclusion_path = OUTPUT_DIR / "test_label_exclusion_summary.csv"
    exclusion_labels.to_csv(exclusion_path, index=False)

    print(f"Total test samples: {len(df)}")
    print(f"Evaluated samples: {len(valid_df)}")
    print(f"Excluded samples: {len(excluded_df)}")
    print("\nExcluded label distribution:")
    print(exclusion_labels.to_string(index=False))

    return (
        df["sample_id"].to_numpy(),
        valid_df["sample_id"].to_numpy(),
        y_true,
        len(df),
        len(valid_df),
        len(excluded_df),
        exclusion_labels,
    )


def load_model_test_data(model, all_test_ids):
    folder = INPUT_ROOT / model / "predictions"

    predictions_path = folder / f"{model}_predictions.csv"
    probabilities_path = folder / f"{model}_probabilities.csv"

    predictions_df = load_csv_with_ids(
        predictions_path,
        ["sample_id", "predicted_label"]
    )

    probabilities_df = load_csv_with_ids(
        probabilities_path,
        ["sample_id", *PROB_COLS]
    )

    predictions_df = align_dataframe(
        predictions_df,
        all_test_ids,
        f"{model} predictions"
    )

    probabilities_df = align_dataframe(
        probabilities_df,
        all_test_ids,
        f"{model} probabilities"
    )

    if not np.array_equal(
        predictions_df["sample_id"].to_numpy(),
        probabilities_df["sample_id"].to_numpy()
    ):
        raise ValueError(f"ID mismatch in {model} test files.")

    probs = probabilities_df[PROB_COLS].to_numpy(dtype=float)

    validate_probabilities(probs, f"{model} test probabilities")

    predictions = predictions_df["predicted_label"].to_numpy()

    if not pd.Series(predictions).isin(LABELS).all():
        raise ValueError(f"Invalid predicted labels in {model}.")

    return predictions, probs


def load_validation_predictions(model):
    path = (
        INPUT_ROOT
        / model
        / "predictions"
        / f"{model}_validation_predictions.csv"
    )

    df = load_csv_with_ids(
        path,
        ["sample_id", "gold_label", "predicted_label"]
    )

    if not df["gold_label"].isin(LABELS).all():
        raise ValueError(f"Invalid validation gold labels in {model}.")

    if not df["predicted_label"].isin(LABELS).all():
        raise ValueError(f"Invalid validation predictions in {model}.")

    return df


def select_best_individual_model():
    scores = {}

    for model in MODELS:
        df = load_validation_predictions(model)

        score = f1_score(
            df["gold_label"],
            df["predicted_label"],
            labels=LABELS,
            average="macro",
            zero_division=0
        )

        scores[model] = float(score)

        print(
            f"{model} validation Macro-F1: {score:.4f}"
        )

    best_model = max(scores, key=scores.get)

    print(
        f"\nSelected best individual model: {best_model}"
    )

    return best_model, scores[best_model], scores


def load_hard_voting(all_test_ids, model_predictions, model_probabilities):
    path = RESULTS_DIR / "C1_hard_voting" / "hard_voting_predictions.csv"

    hard_df = load_csv_with_ids(
        path,
        ["sample_id", "predicted_label"]
    )

    hard_df = align_dataframe(
        hard_df,
        all_test_ids,
        "Hard Voting predictions"
    )

    hard_predictions = []

    hard_probabilities = []

    for i in range(len(all_test_ids)):
        votes = [
            LABEL_TO_ID[model_predictions[model][i]]
            for model in MODELS
        ]

        counts = np.bincount(votes, minlength=len(LABELS))

        max_votes = counts.max()

        candidates = np.flatnonzero(counts == max_votes)

        if len(candidates) == 1:
            winner = int(candidates[0])

        else:
            mean_probs = np.mean(
                [
                    model_probabilities[model][i]
                    for model in MODELS
                ],
                axis=0
            )

            winner = int(
                candidates[
                    np.argmax(mean_probs[candidates])
                ]
            )

        hard_predictions.append(LABELS[winner])

        hard_probabilities.append(
            counts.astype(float) / len(MODELS)
        )

    hard_predictions = np.asarray(hard_predictions)
    hard_probabilities = np.asarray(hard_probabilities)

    saved_predictions = hard_df["predicted_label"].to_numpy()

    if not np.array_equal(hard_predictions, saved_predictions):
        raise ValueError(
            "Reproduced Hard Voting predictions do not match "
            "hard_voting_predictions.csv."
        )

    return hard_predictions, hard_probabilities


def load_soft_voting(all_test_ids):
    path = RESULTS_DIR / "C2_soft_voting" / "soft_voting_probabilities.csv"

    df = load_csv_with_ids(
        path,
        ["sample_id", *PROB_COLS]
    )

    df = align_dataframe(
        df,
        all_test_ids,
        "Soft Voting probabilities"
    )

    probs = df[PROB_COLS].to_numpy(dtype=float)

    validate_probabilities(probs, "Soft Voting probabilities")

    predictions = np.asarray(LABELS)[np.argmax(probs, axis=1)]

    return predictions, probs


def load_temperature_scaled_voting(all_test_ids):
    calibrated_probabilities = []

    for model in MODELS:
        path = (
            RESULTS_DIR / "C3_temperature_scaling"
            / f"{model}_calibrated_test_probabilities.csv"
        )

        df = load_csv_with_ids(
            path,
            ["sample_id", *PROB_COLS]
        )

        df = align_dataframe(
            df,
            all_test_ids,
            f"{model} calibrated probabilities"
        )

        probs = df[PROB_COLS].to_numpy(dtype=float)

        validate_probabilities(
            probs,
            f"{model} calibrated probabilities"
        )

        calibrated_probabilities.append(probs)

    ensemble_probs = np.mean(
        calibrated_probabilities,
        axis=0
    )

    validate_probabilities(
        ensemble_probs,
        "Temperature-scaled Soft Voting"
    )

    predictions = np.asarray(LABELS)[
        np.argmax(ensemble_probs, axis=1)
    ]

    return predictions, ensemble_probs


def expected_calibration_error(probs, y_true, prediction_ids):
    confidences = probs[
        np.arange(len(y_true)),
        prediction_ids
    ]

    accuracies = (prediction_ids == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, N_BINS + 1)

    ece = 0.0

    for i in range(N_BINS):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == 0:
            mask = (
                (confidences >= lower)
                & (confidences <= upper)
            )
        else:
            mask = (
                (confidences > lower)
                & (confidences <= upper)
            )

        if np.any(mask):
            bin_accuracy = accuracies[mask].mean()
            bin_confidence = confidences[mask].mean()

            ece += (
                mask.sum() / len(y_true)
            ) * abs(bin_accuracy - bin_confidence)

    return float(ece)

def multiclass_brier_score(probs, y_true):
    one_hot = np.eye(len(LABELS))[y_true]

    return float(
        np.mean(
            np.sum((probs - one_hot) ** 2, axis=1)
        )
    )


def evaluate_configuration(predictions, probs, y_true):
    prediction_ids = np.array(
        [LABEL_TO_ID[label] for label in predictions],
        dtype=int
    )

    return {
        "accuracy": accuracy_score(y_true, prediction_ids),
        "macro_f1": f1_score(
            y_true,
            prediction_ids,
            labels=list(range(len(LABELS))),
            average="macro",
            zero_division=0
        ),
        "ece": expected_calibration_error(
            probs,
            y_true,
            prediction_ids
        ),
        "brier_score": multiclass_brier_score(
            probs,
            y_true
        ),
    }


def plot_ensemble_comparison(results_df):
    metrics = [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("ece", "ECE"),
        ("brier_score", "Brier Score"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes = axes.flatten()

    for ax, (metric, title) in zip(axes, metrics):
        bars = ax.bar(
            results_df["configuration"],
            results_df[metric]
        )

        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=20)

        for bar in bars:
            height = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    fig.suptitle("Ablation Study: Ensemble Comparison")
    fig.tight_layout()

    output_path = OUTPUT_DIR / "ensemble_comparison.png"

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Ensemble comparison plot: {output_path}")


def generate_ablation_report(
    results_df,
    best_model,
    best_validation_f1,
    validation_scores,
    total_samples,
    evaluated_samples,
    excluded_samples,
    exclusion_labels
):
    best_test_f1_row = results_df.loc[
        results_df["macro_f1"].idxmax()
    ]

    best_test_accuracy_row = results_df.loc[
        results_df["accuracy"].idxmax()
    ]

    best_ece_row = results_df.loc[
        results_df["ece"].idxmin()
    ]

    best_brier_row = results_df.loc[
        results_df["brier_score"].idxmin()
    ]

    validation_table = pd.DataFrame([
        {
            "Model": model,
            "Validation Macro-F1": score
        }
        for model, score in validation_scores.items()
    ])

    exclusion_table = exclusion_labels.rename(
        columns={
            "excluded_label": "Excluded label",
            "sample_count": "Sample count"
        }
    )

    report = f"""# Ablation Study Analysis

## 1. Experimental Setup

Four configurations were evaluated:

1. Best Individual Model
2. Hard Voting
3. Soft Voting
4. Soft Voting + Temperature Scaling

The best individual model was selected using Macro-F1 on the Validation Set.
Test Set performance was not used for model selection.

Temperature parameters were fitted using Validation Set logits only.
The fitted temperatures were then applied to the Test Set probabilities.

All configurations were evaluated on the same valid Test samples.

## 2. Best Individual Model Selection

Selected model: **{best_model}**

Validation Macro-F1: **{best_validation_f1:.4f}**

{validation_table.to_markdown(index=False)}

## 3. Test Set Samples

| Description | Number |
|---|---:|
| Total Test samples | {total_samples} |
| Evaluated samples | {evaluated_samples} |
| Excluded samples | {excluded_samples} |

Samples with labels outside the nine official categories were excluded
from metric calculation. Predictions were still generated for all
Test samples.

### Excluded Label Distribution

{exclusion_table.to_markdown(index=False)}

## 4. Evaluation Results

{results_df.to_markdown(index=False, floatfmt=".4f")}

Metrics:

- Accuracy: proportion of correct predictions.
- Macro-F1: unweighted average of F1 scores across the nine classes.
- ECE: top-label Expected Calibration Error using {N_BINS} equal-width bins.
- Brier Score: mean multiclass sum of squared errors between predicted
  probabilities and one-hot targets.

For Hard Voting, class probabilities were calculated as the proportion
of votes received by each class.

For Soft Voting, probabilities were averaged across the three original
models.

For Soft Voting + Temperature Scaling, each model's probabilities were
calibrated separately before averaging.

## 5. Observations

- Highest Test Macro-F1: **{best_test_f1_row['configuration']}**
  ({best_test_f1_row['macro_f1']:.4f}).
- Highest Test Accuracy: **{best_test_accuracy_row['configuration']}**
  ({best_test_accuracy_row['accuracy']:.4f}).
- Lowest ECE: **{best_ece_row['configuration']}**
  ({best_ece_row['ece']:.4f}).
- Lowest Brier Score: **{best_brier_row['configuration']}**
  ({best_brier_row['brier_score']:.4f}).

These results show the relative performance of the four configurations
on this Test Set. Calibration and classification performance are
reported separately because a lower calibration error does not
necessarily imply a higher classification score.

## 6. Limitations

The results are based on one fixed Test Set. No additional hyperparameter
tuning was performed on the Test Set.

Temperature Scaling was fitted on the Validation Set only.
"""

    output_path = DOCS_DIR / "ablation_analysis.md"

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(report)

    print(f"Ablation analysis saved: {output_path}")


def main():
    (
        all_test_ids,
        valid_test_ids,
        y_true,
        total_samples,
        evaluated_samples,
        excluded_samples,
        exclusion_labels
    ) = load_test_data()

    print("\nSelecting the best individual model...")

    best_model, best_validation_f1, validation_scores = (
        select_best_individual_model()
    )

    model_predictions = {}
    model_probabilities = {}

    print("\nLoading individual model predictions...")

    for model in MODELS:
        predictions, probs = load_model_test_data(
            model,
            all_test_ids
        )

        model_predictions[model] = predictions
        model_probabilities[model] = probs

    print("\nLoading ensemble predictions...")

    hard_predictions, hard_probs = load_hard_voting(
        all_test_ids,
        model_predictions,
        model_probabilities
    )

    soft_predictions, soft_probs = load_soft_voting(
        all_test_ids
    )

    ts_predictions, ts_probs = load_temperature_scaled_voting(
        all_test_ids
    )

    configurations = {}

    best_predictions = model_predictions[best_model]
    best_probs = model_probabilities[best_model]

    configurations["Best Individual Model"] = (
        best_predictions,
        best_probs
    )

    configurations["Hard Voting"] = (
        hard_predictions,
        hard_probs
    )

    configurations["Soft Voting"] = (
        soft_predictions,
        soft_probs
    )

    configurations["Soft Voting + Temperature Scaling"] = (
        ts_predictions,
        ts_probs
    )

    valid_indices = pd.Index(all_test_ids).get_indexer(
        valid_test_ids
    )

    if (valid_indices < 0).any():
        raise ValueError("Some valid test IDs are missing.")

    results = []

    print("\nEvaluating configurations...")

    for name, (predictions, probs) in configurations.items():
        valid_predictions = predictions[valid_indices]
        valid_probs = probs[valid_indices]

        metrics = evaluate_configuration(
            valid_predictions,
            valid_probs,
            y_true
        )

        results.append({
            "configuration": name,
            "selected_individual_model": (
                best_model if name == "Best Individual Model" else ""
            ),
            "validation_macro_f1": (
                best_validation_f1
                if name == "Best Individual Model"
                else np.nan
            ),
            **metrics,
            "total_test_samples": total_samples,
            "evaluated_samples": evaluated_samples,
            "excluded_samples": excluded_samples,
        })

        print(f"\n{name}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    results_df = pd.DataFrame(results)

    results_path = OUTPUT_DIR / "ensemble_results.csv"

    results_df.to_csv(
        results_path,
        index=False
    )

    plot_ensemble_comparison(results_df)

    generate_ablation_report(
        results_df,
        best_model,
        best_validation_f1,
        validation_scores,
        total_samples,
        evaluated_samples,
        excluded_samples,
        exclusion_labels
    )

    print("\nAblation Study completed.")
    print(f"Results: {results_path}")
    print(f"Plots: {OUTPUT_DIR}")
    print(f"Analysis: {DOCS_DIR / 'ablation_analysis.md'}")


if __name__ == "__main__":
    main()
