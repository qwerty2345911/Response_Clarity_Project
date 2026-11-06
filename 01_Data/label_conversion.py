
"""
Label Conversion Utilities
Project: Confidence-Aware Model Ensemble and LLM Reasoning
         for Response Clarity Classification

This module provides:
1. Label name -> numerical ID
2. Numerical ID -> label name
3. Task 2 (9 classes) -> Task 1 (3 classes)

All mappings are loaded from label_mapping.json.
Original dataset files are never modified.
"""

import json
from pathlib import Path


# ============================================================
# 1. Load label mapping configuration
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
MAPPING_FILE = CURRENT_DIR / "label_mapping.json"


if not MAPPING_FILE.exists():
    raise FileNotFoundError(
        f"Label mapping file not found: {MAPPING_FILE}"
    )

with open(MAPPING_FILE, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)


# Task 1 mappings
TASK1_CONFIG = CONFIG["task_1"]
TASK1_LABEL_TO_ID = TASK1_CONFIG["label_to_id"]
TASK1_ID_TO_LABEL = {
    value: key for key, value in TASK1_LABEL_TO_ID.items()
}

# Task 2 mappings
TASK2_CONFIG = CONFIG["task_2"]
TASK2_LABEL_TO_ID = TASK2_CONFIG["label_to_id"]
TASK2_ID_TO_LABEL = {
    value: key for key, value in TASK2_LABEL_TO_ID.items()
}

# Original hierarchical taxonomy mapping
TASK2_TO_TASK1 = CONFIG["task_2_to_task_1_mapping"]


# ============================================================
# 2. Select task mapping
# ============================================================

def _get_mappings(task):
    """
    Return label-to-ID and ID-to-label mappings.

    Args:
        task: 'task_1' or 'task_2'
    """

    if task == "task_1":
        return TASK1_LABEL_TO_ID, TASK1_ID_TO_LABEL

    if task == "task_2":
        return TASK2_LABEL_TO_ID, TASK2_ID_TO_LABEL

    raise ValueError(
        "Invalid task. Use 'task_1' or 'task_2'."
    )


# ============================================================
# 3. Convert label name to numerical ID
# ============================================================

def encode_label(label, task):
    """
    Convert a label name into its numerical ID.

    Example:
        encode_label("Clear Reply", "task_1") -> 0
        encode_label("Explicit", "task_2") -> 0
    """

    label_to_id, _ = _get_mappings(task)

    if label not in label_to_id:
        raise ValueError(
            f"Unknown label '{label}' for {task}. "
            f"Available labels: {list(label_to_id.keys())}"
        )

    return label_to_id[label]


# ============================================================
# 4. Convert numerical ID to label name
# ============================================================

def decode_label(label_id, task):
    """
    Convert a numerical ID into its label name.

    Example:
        decode_label(0, "task_1") -> "Clear Reply"
        decode_label(0, "task_2") -> "Explicit"
    """

    _, id_to_label = _get_mappings(task)

    if label_id not in id_to_label:
        raise ValueError(
            f"Unknown label ID '{label_id}' for {task}. "
            f"Available IDs: {list(id_to_label.keys())}"
        )

    return id_to_label[label_id]


# ============================================================
# 5. Convert a list of labels into numerical IDs
# ============================================================

def encode_labels(labels, task):
    """
    Convert a list of label names into numerical IDs.

    Example:
        encode_labels(
            ["Clear Reply", "Ambivalent"],
            "task_1"
        ) -> [0, 1]
    """

    return [
        encode_label(label, task)
        for label in labels
    ]


# ============================================================
# 6. Convert a list of numerical IDs into label names
# ============================================================

def decode_labels(label_ids, task):
    """
    Convert a list of numerical IDs into label names.
    """

    return [
        decode_label(label_id, task)
        for label_id in label_ids
    ]


# ============================================================
# 7. Convert Task 2 labels into Task 1 labels
# ============================================================

def convert_task2_to_task1(label):
    """
    Convert a fine-grained Task 2 label into its
    corresponding high-level Task 1 clarity label.

    This follows the original QEvasion hierarchical taxonomy.

    Example:
        Explicit -> Clear Reply
        Dodging -> Ambivalent
        Claims ignorance -> Clear Non-Reply
    """

    if label not in TASK2_TO_TASK1:
        raise ValueError(
            f"Unknown Task 2 label: '{label}'. "
            f"Available labels: {list(TASK2_TO_TASK1.keys())}"
        )

    return TASK2_TO_TASK1[label]


def convert_task2_labels_to_task1(labels):
    """
    Convert a list of Task 2 labels into Task 1 labels.
    """

    return [
        convert_task2_to_task1(label)
        for label in labels
    ]


# ============================================================
# 8. Display available mappings
# ============================================================

def print_mappings():
    """Print all label mappings."""

    print("\nTASK 1: CLARITY-LEVEL CLASSIFICATION")
    print("-" * 50)

    for label, label_id in TASK1_LABEL_TO_ID.items():
        print(f"{label_id}: {label}")

    print("\nTASK 2: EVASION-LEVEL CLASSIFICATION")
    print("-" * 50)

    for label, label_id in TASK2_LABEL_TO_ID.items():
        print(f"{label_id}: {label}")

    print("\nORIGINAL TAXONOMY: TASK 2 -> TASK 1")
    print("-" * 50)

    for label, clarity_label in TASK2_TO_TASK1.items():
        print(f"{label} -> {clarity_label}")


# ============================================================
# 9. Run basic tests
# ============================================================

def run_tests():
    """Run basic conversion tests."""

    # Task 1 encoding and decoding
    assert encode_label("Clear Reply", "task_1") == 0
    assert decode_label(0, "task_1") == "Clear Reply"

    # Task 2 encoding and decoding
    assert encode_label("Explicit", "task_2") == 0
    assert decode_label(0, "task_2") == "Explicit"

    # List conversion
    assert encode_labels(
        ["Clear Reply", "Ambivalent"],
        "task_1"
    ) == [0, 1]

    assert decode_labels(
        [0, 1],
        "task_1"
    ) == ["Clear Reply", "Ambivalent"]

    # Hierarchical taxonomy conversion
    assert convert_task2_to_task1(
        "Explicit"
    ) == "Clear Reply"

    assert convert_task2_to_task1(
        "Dodging"
    ) == "Ambivalent"

    assert convert_task2_to_task1(
        "Claims ignorance"
    ) == "Clear Non-Reply"

    # All nine Task 2 labels should have a Task 1 mapping
    assert set(TASK2_LABEL_TO_ID.keys()) == set(
        TASK2_TO_TASK1.keys()
    )

    print("\n" + "=" * 55)
    print("LABEL CONVERSION TESTS")
    print("=" * 55)
    print("Task 1 encoding/decoding: PASSED")
    print("Task 2 encoding/decoding: PASSED")
    print("List conversion: PASSED")
    print("Task 2 -> Task 1 conversion: PASSED")
    print("All nine hierarchical mappings: PASSED")
    print("=" * 55)
    print("ALL TESTS PASSED!")


# ============================================================
# 10. Main
# ============================================================

if __name__ == "__main__":
    print("Label Conversion Utility")
    print(f"Mapping file: {MAPPING_FILE}")

    print_mappings()
    run_tests()
