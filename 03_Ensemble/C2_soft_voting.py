
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

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "C2_soft_voting"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


def load_probabilities(model):
    path = (
        INPUT_ROOT
        / model
        / "predictions"
        / f"{model}_probabilities.csv"
    )

    if not path.exists():
        raise FileNotFoundError(f"Probability file not found: {path}")

    data = pd.read_csv(path)

    required_cols = {"sample_id", *PROB_COLS}

    if not required_cols.issubset(data.columns):
        raise ValueError(f"Missing probability columns in {model}")

    data["sample_id"] = data["sample_id"].astype(str)

    if data["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample IDs in {model}")

    probs = data[PROB_COLS].to_numpy(dtype=float)

    if not np.isfinite(probs).all():
        raise ValueError(f"Non-finite probabilities in {model}")

    if (probs < 0).any() or (probs > 1).any():
        raise ValueError(f"Probabilities outside [0, 1] in {model}")

    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError(f"Probability sums are invalid in {model}")

    return data, probs


def main():
    probability_frames = []
    probability_matrices = []

    for model in MODELS:
        frame, probabilities = load_probabilities(model)

        probability_frames.append(frame)
        probability_matrices.append(probabilities)

    reference_ids = probability_frames[0]["sample_id"]

    for model, frame in zip(MODELS[1:], probability_frames[1:]):
        if not reference_ids.equals(frame["sample_id"]):
            raise ValueError(f"Sample ID mismatch across models: {model}")

    mean_probs = np.mean(
        np.stack(probability_matrices, axis=0),
        axis=0
    )

    if not np.allclose(mean_probs.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Soft Voting produced invalid probability distributions")

    predicted_labels = np.asarray(LABELS)[
        np.argmax(mean_probs, axis=1)
    ]

    prediction_output = pd.DataFrame({
        "sample_id": reference_ids,
        "predicted_label": predicted_labels,
    })

    probability_output = pd.DataFrame(
        mean_probs,
        columns=PROB_COLS
    )

    probability_output.insert(
        0,
        "sample_id",
        reference_ids.to_numpy()
    )

    prediction_path = OUTPUT_DIR / "soft_voting_predictions.csv"
    probability_path = OUTPUT_DIR / "soft_voting_probabilities.csv"

    prediction_output.to_csv(prediction_path, index=False)
    probability_output.to_csv(probability_path, index=False)

    print("Soft Voting completed.")
    print(f"Number of samples: {len(prediction_output)}")
    print(f"Predictions saved to: {prediction_path}")
    print(f"Probabilities saved to: {probability_path}")


if __name__ == "__main__":
    main()
