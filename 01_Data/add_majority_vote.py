
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "01_Data" / "processed"
TEST_FILE = DATA_DIR / "test_cleaned.csv"

df = pd.read_csv(TEST_FILE)

annotator_cols = ["annotator1", "annotator2", "annotator3"]

for col in annotator_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")


def get_vote_result(row):
    labels = [
        str(row[col]).strip()
        for col in annotator_cols
        if pd.notna(row[col]) and str(row[col]).strip() != ""
    ]

    if len(labels) != 3:
        return pd.Series(["Missing_Annotation", 0])

    counts = pd.Series(labels).value_counts()

    max_votes = counts.iloc[0]

    if max_votes >= 2:
        return pd.Series([counts.index[0], int(max_votes)])

    return pd.Series(["No_Majority", 0])


# Generate majority label and vote count
df[["evasion_label_majority", "majority_vote_count"]] = df.apply(
    get_vote_result, axis=1
)

# Save updated dataset
df.to_csv(TEST_FILE, index=False, encoding="utf-8-sig")

print("=" * 55)
print("MAJORITY VOTE STATISTICS")
print("=" * 55)

print("\nMajority label distribution:")
print(df["evasion_label_majority"].value_counts())

print("\nVote count distribution:")
print(
    df["majority_vote_count"]
    .value_counts()
    .sort_index()
    .rename(index={0: "No majority", 2: "2 votes", 3: "3 votes"})
)

print("\nDetailed label and vote count:")
print(
    df.groupby(
        ["evasion_label_majority", "majority_vote_count"]
    ).size().to_string()
)

print("\nTotal samples:", len(df))
print("No majority:", (df["evasion_label_majority"] == "No_Majority").sum())
print("Missing annotation:", (df["evasion_label_majority"] == "Missing_Annotation").sum())
print("2 votes:", (df["majority_vote_count"] == 2).sum())
print("3 votes:", (df["majority_vote_count"] == 3).sum())

print("\nUpdated file:", TEST_FILE)
print("\nCompleted successfully.")
