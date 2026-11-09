from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MERGED = PROJECT_ROOT / "04_LLM_Analysis" / "merged" / "merged_test.csv"
OUT_DIR = PROJECT_ROOT / "04_LLM_Analysis"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TASK2_LABELS = [
    "Explicit", "Implicit", "Dodging", "General", "Deflection",
    "Declining to answer", "Claims ignorance", "Clarification",
    "Partial/half-answer",
]

def main():
    df = pd.read_csv(MERGED)
    print("rows:", len(df))

    models = [c.replace("__pred_label", "") for c in df.columns if c.endswith("__pred_label")]
    print("models:", models)

    rows = []
    for m in models:
        gold_col = "evasion_label_majority"
        mask = df["majority_vote_count"] > 0
        sub = df[mask].copy()
        pred = sub[f"{m}__pred_label"]
        gold = sub[gold_col]
        wrong = pred != gold
        for _, r in sub[wrong].iterrows():
            rows.append({
                "sample_id": r["sample_id"],
                "model": m,
                "gold_majority": r[gold_col],
                "annotator1": r["annotator1"],
                "annotator2": r["annotator2"],
                "annotator3": r["annotator3"],
                "pred_label": r[f"{m}__pred_label"],
                "confidence": r[f"{m}__confidence"],
                "is_no_majority": r["is_no_majority"],
            })

    # no_majority 全量也留一份
    nm = df[df["is_no_majority"] == 1][["sample_id", "annotator1", "annotator2", "annotator3"]]
    nm.to_csv(OUT_DIR / "no_majority_samples.csv", index=False)
    print("no_majority saved:", len(nm))

    err = pd.DataFrame(rows)
    err.to_csv(OUT_DIR / "error_analysis.csv", index=False)
    print("errors:", len(err))

    # 混淆矩阵：每个模型在 majority 275 上
    for m in models:
        sub = df[df["majority_vote_count"] > 0]
        cm = confusion_matrix(sub["evasion_label_majority"], sub[f"{m}__pred_label"], labels=TASK2_LABELS)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(cm, display_labels=TASK2_LABELS)
        disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
        ax.set_title(f"{m} confusion (majority 275)")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"cm_{m}.png", dpi=150)
        plt.close(fig)

    # top 混淆类别对
    pairs = []
    for m in models:
        sub = df[df["majority_vote_count"] > 0]
        for g in TASK2_LABELS:
            for p in TASK2_LABELS:
                if g == p:
                    continue
                cnt = ((sub["evasion_label_majority"] == g) & (sub[f"{m}__pred_label"] == p)).sum()
                if cnt > 0:
                    pairs.append({"model": m, "gold": g, "pred": p, "count": int(cnt)})
    pd.DataFrame(pairs).sort_values(["model", "count"], ascending=[True, False]).to_csv(
        OUT_DIR / "top_confusions.csv", index=False
    )

    # 按 gold 的 per-class 错误率
    per_class = []
    for m in models:
        sub = df[df["majority_vote_count"] > 0]
        for g in TASK2_LABELS:
            s = sub[sub["evasion_label_majority"] == g]
            if len(s) == 0:
                continue
            acc = (s[f"{m}__pred_label"] == g).mean()
            per_class.append({"model": m, "label": g, "n": len(s), "acc": round(float(acc), 4)})
    pd.DataFrame(per_class).to_csv(OUT_DIR / "per_class_acc.csv", index=False)

    print("figures:", FIG_DIR)

if __name__ == "__main__":
    main()