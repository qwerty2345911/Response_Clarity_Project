import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

# 1. 路径配置
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = PROJECT_ROOT / "03_Ensemble" / "results" / "C4_model_ablation"
TEST_CSV = PROJECT_ROOT / "01_Data" / "data_splits" / "test.csv"

def main():
    # 2. 读取真实测试集，筛选出 275 条有明确多数票的样本
    test_df = pd.read_csv(TEST_CSV)
    test_df = test_df[test_df["majority_vote_count"] > 0]
    gold_labels = test_df["evasion_label_majority"]
    test_ids = test_df["sample_id"]

    # 3. 定义 C 同学生成的四个消融实验文件
    files = {
        "C4-1 Full Ensemble (ALL)": "C4-1_full_ensemble_test_probabilities.csv",
        "C4-2 Without RoBERTa": "C4-2_without_roberta_test_probabilities.csv",
        "C4-3 Without DeBERTa": "C4-3_without_deberta_test_probabilities.csv",
        "C4-4 Without XLNet": "C4-4_without_xlnet_test_probabilities.csv"
    }

    print("=" * 50)
    print(" C4 Model Ablation 真实指标计算 (275 Majority Samples)")
    print("=" * 50)

    # 4. 循环计算每个组合的指标
    for name, filename in files.items():
        filepath = TEST_DIR / filename
        if not filepath.exists():
            print(f"[错误] 找不到文件: {filepath}")
            continue
            
        df = pd.read_csv(filepath)
        
        # 对齐 sample_id，保证顺序与 gold_labels 绝对一致
        df = df[df["sample_id"].isin(test_ids)].set_index("sample_id").reindex(test_ids)
        
        # 提取概率列
        prob_cols = [c for c in df.columns if c.startswith("p_")]
        if not prob_cols:
            print(f"[错误] {name} 找不到以 'p_' 开头的概率列")
            continue
            
        # 将列名转回原标签 (例如 p_Partial_half_answer -> Partial/half-answer)
        labels_map = [c.replace("p_", "").replace("_", " ") for c in prob_cols]
        labels_map = [l.replace("Partial half answer", "Partial/half-answer") for l in labels_map]
        
        # 获取概率最大的预测标签
        pred_idx = df[prob_cols].values.argmax(axis=1)
        pred_labels = [labels_map[i] for i in pred_idx]
        
        # 计算指标
        acc = accuracy_score(gold_labels, pred_labels)
        macro_f1 = f1_score(gold_labels, pred_labels, average="macro", zero_division=0)
        
        print(f"{name:<25} | Accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f}")

if __name__ == "__main__":
    main()