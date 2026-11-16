
from pathlib import Path

from soft_voting_utils import (
    load_probabilities,
    soft_voting,
)


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


def main():
    probability_frames = []

    for model in MODELS:
        path = (
            INPUT_ROOT
            / model
            / "predictions"
            / f"{model}_probabilities.csv"
        )

        probability_frames.append(load_probabilities(path))

    prediction_output, probability_output = soft_voting(
        probability_frames
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
