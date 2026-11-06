
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# 1. Project paths and configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "01_Data"
PROCESSED_DIR = DATA_DIR / "processed"
SPLIT_DIR = DATA_DIR / "data_splits"

TRAIN_INPUT = PROCESSED_DIR / "train_cleaned.csv"
TEST_INPUT = PROCESSED_DIR / "test_cleaned.csv"

RANDOM_SEED = 42

TRAIN_SIZE = 2700
VAL_SIZE = 748
TEST_SIZE = 308

LABEL_COLUMN = "evasion_label"


# ============================================================
# 2. Check input files
# ============================================================

if not TRAIN_INPUT.exists():
    raise FileNotFoundError(f"Training file not found: {TRAIN_INPUT}")

if not TEST_INPUT.exists():
    raise FileNotFoundError(f"Test file not found: {TEST_INPUT}")

train_all = pd.read_csv(TRAIN_INPUT)
test_original = pd.read_csv(TEST_INPUT)

print("=" * 65)
print("DATA SPLITTING")
print("=" * 65)

print(f"Original processed training set: {train_all.shape}")
print(f"Original processed test set: {test_original.shape}")


# ============================================================
# 3. Validate required columns and sample indices
# ============================================================

if LABEL_COLUMN not in train_all.columns:
    raise ValueError(f"Training data missing column: {LABEL_COLUMN}")

if train_all[LABEL_COLUMN].isna().any():
    raise ValueError("Missing evasion labels found in training data.")

required_majority_columns = [
    "evasion_label_majority",
    "majority_vote_count"
]

for column in required_majority_columns:
    if column not in test_original.columns:
        raise ValueError(
            f"Test data missing majority-vote column: {column}"
        )

for name, df in [
    ("train", train_all),
    ("test", test_original)
]:
    if "index" not in df.columns:
        raise ValueError(f"{name} data missing index column.")

    if df["index"].isna().any():
        raise ValueError(f"{name} data has missing index values.")

    if not df["index"].is_unique:
        raise ValueError(f"{name} data has duplicate index values.")

    if df["index"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{name} data has empty index values.")


# ============================================================
# 4. Generate sample IDs
# ============================================================

# Train and validation originate from the original training set.
train_all["sample_id"] = (
    "train_" + train_all["index"].astype(str)
)

# Test retains all original official test samples.
# Project-derived majority-vote columns are also retained.
test_original["sample_id"] = (
    "test_" + test_original["index"].astype(str)
)


# ============================================================
# 5. Check expected original dataset sizes
# ============================================================

if len(train_all) != TRAIN_SIZE + VAL_SIZE:
    raise ValueError(
        f"Expected {TRAIN_SIZE + VAL_SIZE} training samples, "
        f"but found {len(train_all)}."
    )

if len(test_original) != TEST_SIZE:
    raise ValueError(
        f"Expected {TEST_SIZE} test samples, "
        f"but found {len(test_original)}."
    )


# ============================================================
# 6. Stratified train-validation split
# ============================================================

print("\nCreating stratified train-validation split...")

train_df, val_df = train_test_split(
    train_all,
    train_size=TRAIN_SIZE,
    test_size=VAL_SIZE,
    stratify=train_all[LABEL_COLUMN],
    random_state=RANDOM_SEED
)

# Restore original row order within each split.
train_df = train_df.sort_index().reset_index(drop=True)
val_df = val_df.sort_index().reset_index(drop=True)
test_df = test_original.reset_index(drop=True)


# ============================================================
# 7. Validate split sizes
# ============================================================

assert len(train_df) == TRAIN_SIZE
assert len(val_df) == VAL_SIZE
assert len(test_df) == TEST_SIZE

assert len(train_df) + len(val_df) == len(train_all)

print("\nSplit sizes:")
print(f"Train:      {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test:       {len(test_df)}")
print(f"Total:      {len(train_df) + len(val_df) + len(test_df)}")


# ============================================================
# 8. Validate no overlap between train and validation
# ============================================================

train_indices = set(train_df["index"])
val_indices = set(val_df["index"])
original_train_indices = set(train_all["index"])

assert train_indices.isdisjoint(val_indices), (
    "Index overlap found between train and validation."
)

assert train_indices.union(val_indices) == original_train_indices, (
    "Some original training samples are missing from the split."
)

# Confirm that all original test samples are preserved.
assert set(test_df["index"]) == set(test_original["index"])

print("\nIndex validation passed.")


# ============================================================
# 9. Validate sample IDs
# ============================================================

all_sample_ids = pd.concat(
    [
        train_df["sample_id"],
        val_df["sample_id"],
        test_df["sample_id"]
    ],
    ignore_index=True
)

assert all_sample_ids.notna().all(), (
    "Missing sample_id found."
)

assert all_sample_ids.is_unique, (
    "Duplicate sample_id found across splits."
)

assert all_sample_ids.nunique() == 3756

assert train_df["sample_id"].is_unique
assert val_df["sample_id"].is_unique
assert test_df["sample_id"].is_unique

print("Sample ID validation passed.")
print(f"Unique sample IDs: {all_sample_ids.nunique()}")


# ============================================================
# 10. Validate label classes
# ============================================================

expected_labels = set(train_all[LABEL_COLUMN].unique())

assert set(train_df[LABEL_COLUMN].unique()) == expected_labels
assert set(val_df[LABEL_COLUMN].unique()) == expected_labels

print("\nEvasion label validation passed.")
print(f"Number of classes: {len(expected_labels)}")


# ============================================================
# 11. Create output directory and save split files
# ============================================================

SPLIT_DIR.mkdir(parents=True, exist_ok=True)

train_path = SPLIT_DIR / "train.csv"
val_path = SPLIT_DIR / "validation.csv"
test_path = SPLIT_DIR / "test.csv"

train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

print("\nSplit files saved:")
print(train_path)
print(val_path)
print(test_path)


# ============================================================
# 12. Generate label distribution statistics
# ============================================================

train_counts = train_df[LABEL_COLUMN].value_counts()
val_counts = val_df[LABEL_COLUMN].value_counts()

statistics = pd.DataFrame({
    "Train_Count": train_counts,
    "Validation_Count": val_counts
}).fillna(0).astype(int)

statistics["Train_Percentage"] = (
    statistics["Train_Count"] / len(train_df) * 100
).round(2)

statistics["Validation_Percentage"] = (
    statistics["Validation_Count"] / len(val_df) * 100
).round(2)

statistics["Total"] = (
    statistics["Train_Count"] +
    statistics["Validation_Count"]
)

statistics = statistics.sort_index()

statistics_path = SPLIT_DIR / "split_statistics.csv"

statistics.to_csv(
    statistics_path,
    encoding="utf-8-sig"
)


# ============================================================
# 13. Generate split summary JSON
# ============================================================

summary = {
    "project": "Response Clarity Classification",
    "random_seed": RANDOM_SEED,
    "splitting_strategy": "Stratified sampling",
    "label_column": LABEL_COLUMN,
    "dataset_sizes": {
        "original_train": len(train_all),
        "train": len(train_df),
        "validation": len(val_df),
        "test": len(test_df),
        "total": len(train_df) + len(val_df) + len(test_df)
    },
    "number_of_evasion_classes": len(expected_labels),
    "train_evasion_distribution": {
        str(k): int(v)
        for k, v in train_df[LABEL_COLUMN].value_counts().items()
    },
    "validation_evasion_distribution": {
        str(k): int(v)
        for k, v in val_df[LABEL_COLUMN].value_counts().items()
    },
    "test_clarity_distribution": {
        str(k): int(v)
        for k, v in test_df["clarity_label"].value_counts().items()
    },
    "sample_id": {
        "column": "sample_id",
        "train_validation_prefix": "train_",
        "test_prefix": "test_",
        "total_unique_ids": int(all_sample_ids.nunique())
    }
}

summary_path = SPLIT_DIR / "split_summary.json"

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)


# ============================================================
# 14. Generate data splitting protocol
# ============================================================

protocol = f"""# Data Splitting Protocol

## Dataset

Source: QEvasion Hugging Face dataset.

The processed original training set contains {len(train_all)} samples.
The processed original test set contains {len(test_df)} samples.

## Splitting strategy

The original training split ({len(train_all)} samples) was divided into:

- Training set: {len(train_df)} samples
- Validation set: {len(val_df)} samples

Stratified sampling was performed based on `{LABEL_COLUMN}`.

Random seed: {RANDOM_SEED}

All {len(test_df)} samples from the original official test split were preserved.

Project-derived majority-vote columns and `sample_id` were added to the
processed test data. The original raw test file remains unchanged.

## Final dataset sizes

| Split | Samples |
|---|---:|
| Training | {len(train_df)} |
| Validation | {len(val_df)} |
| Test | {len(test_df)} |
| **Total** | **{len(train_df) + len(val_df) + len(test_df)}** |

## Sample ID

A project-generated `sample_id` column is included in all final split files.

- Training and validation IDs use the `train_` prefix.
- Test IDs use the `test_` prefix.

The IDs support sample identification, prediction alignment,
model-output merging, and error analysis.

All {all_sample_ids.nunique()} sample IDs are unique across the three splits.

## Data integrity

- Training and validation samples do not overlap.
- All original training samples are assigned to either training or validation.
- All original test samples are retained.
- The official test split is not used for training or validation.
- The original raw dataset files remain unchanged.
"""

protocol_path = SPLIT_DIR / "split_protocol.md"

with open(protocol_path, "w", encoding="utf-8") as f:
    f.write(protocol)


# ============================================================
# 15. Final report
# ============================================================

print("\n" + "=" * 65)
print("DATA SPLITTING COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nFinal dataset files:")

for path in [
    train_path,
    val_path,
    test_path
]:
    print(f"{path.name}: {pd.read_csv(path).shape}")

print("\nAdditional reports:")
print(statistics_path)
print(summary_path)
print(protocol_path)

print("\nFinal sample counts:")
print(f"Train:      {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test:       {len(test_df)}")
print(f"Total:      {len(all_sample_ids)}")

print("\nAll validation checks passed.")
