import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = PROJECT_ROOT / "03_Ensemble" / "results" / "C4_model_ablation"
TEST_CSV = PROJECT_ROOT / "01_Data" / "data_splits" / "test.csv"

# 计算多分类 Brier Score
def calculate_brier_score(y_true, y_prob, classes):
    # 将真实标签转为 one-hot
    y_true_one_hot = np.zeros_like(y_prob)
    for i, label in enumerate(y_true):
        if label in classes:
            y_true_one_hot[i, classes.index(label)] = 1
    # 计算平方误差的均值
    return np.mean(np.sum((y_prob - y_true_one_hot) ** 2, axis=1))

# 计算 ECE (Top-label)
def calculate_ece(y_true, y_prob, n_bins=15):
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    
    # 真实标签的索引
    # 这里假设 y_true 已经转成了对应的数字索引
    accuracies = (predictions == y_true)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def main():
    test_df = pd.read_csv(TEST_CSV)
    test_df = test_df[test_df["majority_vote_count"] > 0]
    gold_labels = test_df["evasion_label_majority"].tolist()
    test_ids = test_df["sample_id"]

    files = {
        "Full Ensemble": "C4-1_full_ensemble_test_probabilities.csv",
        "Without RoBERTa": "C4-2_without_roberta_test_probabilities.csv",
        "Without DeBERTa": "C4-3_without_deberta_test_probabilities.csv",
        "Without XLNet": "C4-4_without_xlnet_test_probabilities.csv"
    }

    print("=" * 80)
    print(f"{'Configuration':<20} | {'Accuracy':<10} | {'Macro-F1':<10} | {'ECE':<10} | {'Brier':<10}")
    print("=" * 80)

    for name, filename in files.items():
        filepath = TEST_DIR / filename
        if not filepath.exists():
            print(f"[错误] 找不到文件: {filepath}")
            continue
            
        df = pd.read_csv(filepath)
        df = df[df["sample_id"].isin(test_ids)].set_index("sample_id").reindex(test_ids)
        
        prob_cols = [c for c in df.columns if c.startswith("p_")]
        # 还原真实标签列名
        labels_map = [c.replace("p_", "").replace("_", " ") for c in prob_cols]
        labels_map = [l.replace("Partial half answer", "Partial/half-answer") for l in labels_map]
        
        y_prob = df[prob_cols].values
        pred_idx = y_prob.argmax(axis=1)
        pred_labels = [labels_map[i] for i in pred_idx]
        
        # 将 gold_labels 转为索引，方便算 ECE
        gold_idx = [labels_map.index(l) for l in gold_labels]
        
        acc = accuracy_score(gold_labels, pred_labels)
        macro_f1 = f1_score(gold_labels, pred_labels, average="macro", zero_division=0)
        ece = calculate_ece(np.array(gold_idx), y_prob)
        brier = calculate_brier_score(gold_labels, y_prob, labels_map)
        
        print(f"{name:<20} | {acc:.4f}     | {macro_f1:.4f}     | {ece:.4f}     | {brier:.4f}")

if __name__ == "__main__":
    main()