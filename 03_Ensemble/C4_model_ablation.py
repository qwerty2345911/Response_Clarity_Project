
from pathlib import Path

import pandas as pd

from soft_voting_utils import (
    load_probabilities,
    soft_voting,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_ROOT = PROJECT_ROOT / "02_Baselines" / "artifacts"

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "results"
    / "C4_model_ablation"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


MODEL_DIRS = {
    "roberta": "roberta_task2_results",
    "deberta_v3": "deberta_v3_task2_results",
    "xlnet": "xlnet_task2_results",
}


SPLITS = {
    "validation": "_validation_probabilities.csv",
    "test": "_probabilities.csv",
}


EXPERIMENTS = {
    "C4-1_full_ensemble": [
        "roberta",
        "deberta_v3",
        "xlnet",
    ],
    "C4-2_without_roberta": [
        "deberta_v3",
        "xlnet",
    ],
    "C4-3_without_deberta": [
        "roberta",
        "xlnet",
    ],
    "C4-4_without_xlnet": [
        "roberta",
        "deberta_v3",
    ],
}


def get_probability_path(model, split):
    model_dir = MODEL_DIRS[model]
    filename = f"{model}{SPLITS[split]}"

    return (
        INPUT_ROOT
        / model_dir
        / "predictions"
        / filename
    )


def run_experiment(experiment_name, models, split):
    print(f"\nRunning {experiment_name} ({split})")

    probability_frames = []

    for model in models:
        path = get_probability_path(model, split)

        print(f"Loading: {model}")

        frame = load_probabilities(path)
        probability_frames.append(frame)

    prediction_output, probability_output = soft_voting(
        probability_frames
    )

    prediction_path = (
        OUTPUT_DIR
        / f"{experiment_name}_{split}_predictions.csv"
    )

    probability_path = (
        OUTPUT_DIR
        / f"{experiment_name}_{split}_probabilities.csv"
    )

    prediction_output.to_csv(prediction_path, index=False)
    probability_output.to_csv(probability_path, index=False)

    print(f"Models: {', '.join(models)}")
    print(f"Number of samples: {len(prediction_output)}")
    print(f"Predictions saved: {prediction_path}")
    print(f"Probabilities saved: {probability_path}")

    return {
        "experiment": experiment_name,
        "split": split,
        "models": "+".join(models),
        "n_samples": len(prediction_output),
        "prediction_file": prediction_path.name,
        "probability_file": probability_path.name,
    }


def main():
    summary = []

    for split in SPLITS:
        print(f"\n{'=' * 50}")
        print(f"Starting {split.upper()} experiments")
        print(f"{'=' * 50}")

        for experiment_name, models in EXPERIMENTS.items():
            result = run_experiment(
                experiment_name=experiment_name,
                models=models,
                split=split,
            )

            summary.append(result)

    summary_df = pd.DataFrame(summary)

    summary_path = OUTPUT_DIR / "C4_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nAll C4 experiments completed.")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
