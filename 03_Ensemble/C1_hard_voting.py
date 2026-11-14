
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_ROOT = (
    PROJECT_ROOT
    / "02_Baselines"
    / "member_b_task2_handover"
    / "model_artifacts"
)

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "C1_hard_voting"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "hard_voting_predictions.csv"


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


def load_model_data(model):
    folder = INPUT_ROOT / model / "predictions"

    pred_path = folder / f"{model}_predictions.csv"
    prob_path = folder / f"{model}_probabilities.csv"

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    if not prob_path.exists():
        raise FileNotFoundError(f"Probability file not found: {prob_path}")

    predictions = pd.read_csv(pred_path)
    probabilities = pd.read_csv(prob_path)

    required_pred_cols = {"sample_id", "predicted_label"}
    required_prob_cols = {"sample_id", *PROB_COLS}

    if not required_pred_cols.issubset(predictions.columns):
        raise ValueError(f"Invalid prediction columns for {model}")

    if not required_prob_cols.issubset(probabilities.columns):
        raise ValueError(f"Invalid probability columns for {model}")

    predictions["sample_id"] = predictions["sample_id"].astype(str)
    probabilities["sample_id"] = probabilities["sample_id"].astype(str)

    if predictions["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample IDs in {model} predictions")

    if probabilities["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample IDs in {model} probabilities")

    if set(predictions["sample_id"]) != set(probabilities["sample_id"]):
        raise ValueError(f"Sample ID mismatch between predictions and probabilities for {model}")

    if not predictions["predicted_label"].isin(LABELS).all():
        raise ValueError(f"Unexpected predicted labels in {model}")

    probabilities = (
        probabilities
        .set_index("sample_id")
        .loc[predictions["sample_id"], PROB_COLS]
        .reset_index()
    )

    probs = probabilities[PROB_COLS].to_numpy(dtype=float)

    if not np.isfinite(probs).all():
        raise ValueError(f"Non-finite probabilities in {model}")

    if (probs < 0).any() or (probs > 1).any():
        raise ValueError(f"Probabilities outside [0, 1] in {model}")

    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError(f"Probability sums are invalid in {model}")

    return predictions, probs


def resolve_vote(predictions, probability_matrices, sample_index):
    counts = {
        label: int(np.sum(predictions == label))
        for label in LABELS
    }

    max_votes = max(counts.values())

    tied_labels = [
        label for label, count in counts.items()
        if count == max_votes
    ]

    tie_applied = len(tied_labels) > 1

    if not tie_applied:
        winner = tied_labels[0]
    else:
        mean_probs = np.mean(
            [matrix[sample_index] for matrix in probability_matrices],
            axis=0
        )

        winner = max(
            tied_labels,
            key=lambda label: (
                mean_probs[LABELS.index(label)],
                -LABELS.index(label)
            )
        )

    return winner, max_votes, tie_applied


def main():
    prediction_frames = []
    probability_matrices = []

    for model in MODELS:
        predictions, probabilities = load_model_data(model)

        prediction_frames.append(predictions)
        probability_matrices.append(probabilities)

    reference_ids = prediction_frames[0]["sample_id"]

    for model, frame in zip(MODELS[1:], prediction_frames[1:]):
        if not reference_ids.equals(frame["sample_id"]):
            raise ValueError(f"Sample ID mismatch across models: {model}")

    prediction_matrix = np.column_stack([
        frame["predicted_label"].to_numpy()
        for frame in prediction_frames
    ])

    results = []

    for i, row in enumerate(prediction_matrix):
        winner, votes, tie = resolve_vote(
            row,
            probability_matrices,
            i
        )

        results.append({
            "sample_id": reference_ids.iloc[i],
            "predicted_label": winner,
            "winning_votes": votes,
            "tie_break_applied": tie,
        })

    output = pd.DataFrame(results)
    output.to_csv(OUTPUT_PATH, index=False)

    print("Hard Voting completed.")
    print(f"Number of samples: {len(output)}")
    print(f"Number of ties resolved: {output['tie_break_applied'].sum()}")
    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
