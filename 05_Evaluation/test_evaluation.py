
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from evaluation import (
    load_test_data,
    TASK1_LABELS,
    TASK2_LABELS,
    TASK1_PROBABILITY_COLUMNS,
    TASK2_PROBABILITY_COLUMNS,
    evaluate_task1,
    evaluate_task2_annotators,
    evaluate_task2_majority
)


# ============================================================
# 1. Setup
# ============================================================

print("=" * 65)
print("EVALUATION MODULE - SIMULATED TEST")
print("=" * 65)

rng = np.random.default_rng(42)

test_df = load_test_data()

sample_ids = test_df["sample_id"].tolist()
n = len(test_df)

print(f"\nTest samples: {n}")
print(f"Unique sample IDs: {len(set(sample_ids))}")


# ============================================================
# 2. Generate simulated probabilities
# ============================================================

task1_probabilities = rng.dirichlet(
    np.ones(len(TASK1_LABELS)),
    size=n
)

task2_probabilities = rng.dirichlet(
    np.ones(len(TASK2_LABELS)),
    size=n
)


# ============================================================
# 3. Create temporary prediction files
# ============================================================

with tempfile.TemporaryDirectory() as temp_dir:

    temp_dir = Path(temp_dir)

    # Task 1 probability file
    task1_pred_labels = np.asarray(TASK1_LABELS)[
        np.argmax(task1_probabilities, axis=1)
    ]

    task1_df = pd.DataFrame(
        task1_probabilities,
        columns=TASK1_PROBABILITY_COLUMNS
    )

    task1_df.insert(0, "sample_id", sample_ids)

    task1_label_df = pd.DataFrame({
        "sample_id": sample_ids,
        "predicted_label": task1_pred_labels
    })

    # Deliberately shuffle rows to test sample_id alignment.
    task1_df = task1_df.sample(
        frac=1,
        random_state=10
    ).reset_index(drop=True)

    task1_label_df = task1_label_df.sample(
        frac=1,
        random_state=20
    ).reset_index(drop=True)

    task1_probability_file = temp_dir / "task1_probabilities.csv"
    task1_prediction_file = temp_dir / "task1_predictions.csv"

    task1_df.to_csv(task1_probability_file, index=False)
    task1_label_df.to_csv(task1_prediction_file, index=False)

    # Task 2 probability file
    task2_pred_labels = np.asarray(TASK2_LABELS)[
        np.argmax(task2_probabilities, axis=1)
    ]

    task2_df = pd.DataFrame(
        task2_probabilities,
        columns=TASK2_PROBABILITY_COLUMNS
    )

    task2_df.insert(0, "sample_id", sample_ids)

    task2_label_df = pd.DataFrame({
        "sample_id": sample_ids,
        "predicted_label": task2_pred_labels
    })

    task2_df = task2_df.sample(
        frac=1,
        random_state=30
    ).reset_index(drop=True)

    task2_label_df = task2_label_df.sample(
        frac=1,
        random_state=40
    ).reset_index(drop=True)

    task2_probability_file = temp_dir / "task2_probabilities.csv"
    task2_prediction_file = temp_dir / "task2_predictions.csv"

    task2_df.to_csv(task2_probability_file, index=False)
    task2_label_df.to_csv(task2_prediction_file, index=False)


    # ========================================================
    # 4. Test Task 1
    # ========================================================

    print("\n[TEST 1] Task 1 - Three-class evaluation")

    task1_summary, task1_classes = evaluate_task1(
        test_df,
        task1_probability_file,
        model_name="Simulated_Model",
        prediction_file=task1_prediction_file
    )

    assert task1_summary["num_samples"] == n
    assert np.isfinite(task1_summary["accuracy"])
    assert len(task1_classes) == 3

    print("PASSED")
    print("Samples:", task1_summary["num_samples"])
    print("Accuracy:", round(task1_summary["accuracy"], 4))
    print("Macro-F1:", round(task1_summary["macro_f1"], 4))


    # ========================================================
    # 5. Test Task 2 annotators
    # ========================================================

    print("\n[TEST 2] Task 2 - Three annotator evaluation")

    task2_summaries, task2_classes = evaluate_task2_annotators(
        test_df,
        task2_probability_file,
        model_name="Simulated_Model",
        prediction_file=task2_prediction_file
    )

    assert len(task2_summaries) == 5
    assert len(task2_classes) == 3

    references = [
        row["reference"] for row in task2_summaries
    ]

    assert references == [
        "annotator1",
        "annotator2",
        "annotator3",
        "annotator_mean",
        "annotator_std"
    ]

    for row in task2_summaries:
        assert np.isfinite(row["accuracy"])
        assert np.isfinite(row["macro_f1"])
        assert np.isfinite(row["ece"])
        assert np.isfinite(row["brier_score"])

    print("PASSED")
    for row in task2_summaries:
        print(
            row["reference"],
            "| Accuracy:", round(row["accuracy"], 4),
            "| Macro-F1:", round(row["macro_f1"], 4)
        )


    # ========================================================
    # 6. Test Task 2 majority vote
    # ========================================================

    print("\n[TEST 3] Task 2 - Supplementary majority vote")

    majority_summary, majority_classes = evaluate_task2_majority(
        test_df,
        task2_probability_file,
        model_name="Simulated_Model",
        prediction_file=task2_prediction_file
    )

    expected_majority_count = int(
        test_df["evasion_label_majority"].isin(TASK2_LABELS).sum()
    )

    assert majority_summary["num_samples"] == expected_majority_count
    assert len(majority_classes) == len(TASK2_LABELS)

    print("PASSED")
    print("Majority-vote samples:", majority_summary["num_samples"])
    print("Expected samples:", expected_majority_count)


    # ========================================================
    # 7. Test sample_id mismatch detection
    # ========================================================

    print("\n[TEST 4] Invalid sample_id detection")

    bad_df = pd.read_csv(task1_probability_file)
    bad_df.loc[0, "sample_id"] = "invalid_sample_id"

    bad_file = temp_dir / "bad_probabilities.csv"
    bad_df.to_csv(bad_file, index=False)

    try:
        evaluate_task1(
            test_df,
            bad_file,
            model_name="Invalid_Test"
        )
    except ValueError as error:
        print("PASSED - Invalid IDs correctly rejected.")
        print("Error:", error)
    else:
        raise AssertionError(
            "Invalid sample IDs were not detected."
        )


# ============================================================
# 8. Final report
# ============================================================

print("\n" + "=" * 65)
print("ALL SIMULATED EVALUATION TESTS PASSED!")
print("=" * 65)

print("\nVerified:")
print("1. Task 1 three-class evaluation")
print("2. Task 2 evaluation against all three annotators")
print("3. Annotator mean and standard deviation")
print("4. Supplementary majority-vote evaluation")
print("5. sample_id alignment despite shuffled row order")
print("6. Invalid sample_id detection")

print("\nNo real model predictions were used.")
print("No original dataset files were modified.")
