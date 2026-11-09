from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT = PROJECT_ROOT / "01_Data" / "data_splits" / "test.csv"
IN_DIR = PROJECT_ROOT / "04_LLM_Analysis" / "inputs"   # 放 B/C 文件
OUT_DIR = PROJECT_ROOT / "04_LLM_Analysis" / "merged"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASK2_PROB_COLS = [
    "p_Explicit", "p_Implicit", "p_Dodging", "p_General", "p_Deflection",
    "p_Declining_to_answer", "p_Claims_ignorance", "p_Clarification",
    "p_Partial_half_answer",
]
TASK2_LABELS = [
    "Explicit", "Implicit", "Dodging", "General", "Deflection",
    "Declining to answer", "Claims ignorance", "Clarification",
    "Partial/half-answer",
]

def load_prob(path: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert "sample_id" in df.columns, f"{path} 缺少 sample_id"
    missing = [c for c in TASK2_PROB_COLS if c not in df.columns]
    assert not missing, f"{path} 缺少概率列: {missing}"
    assert df["sample_id"].is_unique, f"{path} sample_id 有重复"
    # 概率和约为 1
    s = df[TASK2_PROB_COLS].sum(axis=1)
    assert ((s - 1).abs() < 1e-3).all(), f"{path} 存在概率和不为 1 的行"
    # 预测标签
    df["pred_label"] = df[TASK2_PROB_COLS].values.argmax(axis=1)
    df["pred_label"] = df["pred_label"].map(dict(enumerate(TASK2_LABELS)))
    df["confidence"] = df[TASK2_PROB_COLS].max(axis=1)
    return df[["sample_id", "pred_label", "confidence"] + TASK2_PROB_COLS].assign(model=model_name)

def main():
    test = pd.read_csv(SPLIT)
    print("test rows:", len(test))
    print("test columns:", list(test.columns))

    prob_files = {
        "roberta": IN_DIR / "roberta_task2_probabilities.csv",
        "deberta": IN_DIR / "deberta_task2_probabilities.csv",
        "soft_voting": IN_DIR / "soft_voting_probabilities.csv",
    }

    frames = []
    for name, path in prob_files.items():
        if not path.exists():
            print(f"[skip] {name}: {path} 不存在")
            continue
        df = load_prob(path, name)
        # 检查覆盖 308
        missing_ids = set(test["sample_id"]) - set(df["sample_id"])
        extra_ids = set(df["sample_id"]) - set(test["sample_id"])
        assert not missing_ids, f"{name} 缺少 {len(missing_ids)} 个 sample_id"
        assert not extra_ids, f"{name} 多出 {len(extra_ids)} 个 sample_id"
        frames.append(df)
        print(f"[ok] {name}: {len(df)} rows")

    if not frames:
        print("没有可用输入，先放 dummy 到 inputs/ 测试")
        return

    long = pd.concat(frames, ignore_index=True)

    # 宽表：一行一个样本，每个模型一组 pred/conf/prob
    base = test[["sample_id", "clarity_label", "annotator1", "annotator2",
                 "annotator3", "evasion_label_majority", "majority_vote_count"]].copy()
    base["is_no_majority"] = (base["majority_vote_count"] == 0).astype(int)

    for name in long["model"].unique():
        sub = long[long["model"] == name].drop(columns=["model"])
        sub = sub.rename(columns={c: f"{name}__{c}" for c in sub.columns if c != "sample_id"})
        base = base.merge(sub, on="sample_id", how="left")

    out = OUT_DIR / "merged_test.csv"
    base.to_csv(out, index=False)
    print("written:", out)
    print("shape:", base.shape)

if __name__ == "__main__":
    main()