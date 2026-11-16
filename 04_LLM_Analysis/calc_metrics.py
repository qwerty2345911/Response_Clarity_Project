import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(PROJECT_ROOT / "04_LLM_Analysis" / "merged" / "merged_test.csv")

# 只保留有有效多数票的 275 条
sub = df[df["majority_vote_count"] > 0]

for m in ["llm_only", "llm_assisted"]:
    y_true = sub["evasion_label_majority"]
    y_pred = sub[f"{m}__pred_label"]
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"{m} -> Accuracy: {acc:.4f}, Macro-F1: {f1:.4f}")