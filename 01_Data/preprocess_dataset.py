
import pandas as pd
from pathlib import Path

# ==========================================
# 1. Project paths
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "01_Data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "train": RAW_DIR / "train.csv",
    "test": RAW_DIR / "test.csv",
}

TEXT_COLUMNS = [
    "question",
    "interview_answer",
]

LABEL_COLUMNS = [
    "clarity_label",
    "evasion_label",
    "annotator1",
    "annotator2",
    "annotator3",
]

# ==========================================
# 2. Load and validate datasets
# ==========================================

datasets = {}

for name, path in FILES.items():

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    required_columns = TEXT_COLUMNS + LABEL_COLUMNS

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{name} is missing columns: {missing_columns}"
        )

    if "index" not in df.columns:
        raise ValueError(f"{name} missing index column")

    if df["index"].isna().any():
        raise ValueError(f"{name} has missing index values")

    if not df["index"].is_unique:
        raise ValueError(f"{name} has duplicate index values")
    
    datasets[name] = df
    print(f"{name}: loaded {len(df)} rows")

# ==========================================
# 3. Clean text and label fields
# ==========================================

for name, df in datasets.items():

    for col in TEXT_COLUMNS:

        df[col] = (
            df[col]
            .fillna("")
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    for col in LABEL_COLUMNS:

        df[col] = df[col].apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )

# ==========================================
# 4. Check required text fields
# ==========================================

for name, df in datasets.items():

    empty_questions = (df["question"] == "").sum()
    empty_answers = (df["interview_answer"] == "").sum()

    print(f"\n{name} data quality:")
    print(f"Empty questions: {empty_questions}")
    print(f"Empty answers: {empty_answers}")

# Keep all original rows.
# Do not remove samples based on missing task labels.
# Task 2 test labels are stored in three annotator columns.

# ==========================================
# 5. Save unified processed datasets
# ==========================================

for name, df in datasets.items():

    output_path = PROCESSED_DIR / f"{name}_cleaned.csv"

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nSaved: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

# ==========================================
# 6. Completion summary
# ==========================================

print("\n" + "=" * 55)
print("DATA PREPROCESSING COMPLETED")
print("=" * 55)

print(f"Train: {len(datasets['train'])} rows")
print(f"Test:  {len(datasets['test'])} rows")

print("\nTask 1 label column: clarity_label")
print("Task 2 training label column: evasion_label")
print("Task 2 test reference labels: annotator1, annotator2, annotator3")

print("\nOriginal raw datasets were not modified.")
