from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "01_Data"
SPLIT_DIR = DATA_DIR / "data_splits"
EVAL_DIR = PROJECT_ROOT / "05_Evaluation"

REQUIRED_FILES = [
    SPLIT_DIR / "train.csv",
    SPLIT_DIR / "validation.csv",
    SPLIT_DIR / "test.csv",
    DATA_DIR / "label_mapping.json",
    EVAL_DIR / "evaluation.py",
    EVAL_DIR / "evaluation_protocol.md",
]

TASK2_LABELS = [
    "Explicit", "Implicit", "Dodging", "General", "Deflection",
    "Declining to answer", "Claims ignorance", "Clarification",
    "Partial/half-answer",
]

def main():
    print("=== 1. 文件存在性检查 ===")
    for f in REQUIRED_FILES:
        print(("[OK] " if f.exists() else "[MISSING] ") + str(f.relative_to(PROJECT_ROOT)))

    print("\n=== 2. test.csv 检查 ===")
    test = pd.read_csv(SPLIT_DIR / "test.csv")
    print("rows:", len(test))
    print("columns:", list(test.columns))
    assert len(test) == 308, f"test 行数应为 308，实际 {len(test)}"

    for col in ["sample_id", "clarity_label", "annotator1", "annotator2", "annotator3",
                "evasion_label_majority", "majority_vote_count"]:
        assert col in test.columns, f"缺少列 {col}"
        print(f"[OK] 存在列 {col}")

    assert test["sample_id"].is_unique, "sample_id 有重复"
    print("[OK] sample_id 唯一")

    no_majority = test[test["majority_vote_count"] == 0]
    print("no_majority 样本数:", len(no_majority))
    assert len(no_majority) == 33, f"no_majority 应为 33，实际 {len(no_majority)}"

    print("\n=== 3. 标签值检查 ===")
    for a in ["annotator1", "annotator2", "annotator3"]:
        bad = set(test[a].dropna().unique()) - set(TASK2_LABELS)
        assert not bad, f"{a} 出现非九类标签: {bad}"
        print(f"[OK] {a} 标签均在九类内")

    majority_valid = test[test["majority_vote_count"] > 0]["evasion_label_majority"]
    bad = set(majority_valid.unique()) - set(TASK2_LABELS)
    assert not bad, f"evasion_label_majority 出现非九类标签: {bad}"
    print("[OK] evasion_label_majority 标签均在九类内")

    print("\nALL CHECKS PASSED")

if __name__ == "__main__":
    main()