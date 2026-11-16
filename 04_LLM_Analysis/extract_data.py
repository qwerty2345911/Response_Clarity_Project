import pandas as pd
from pathlib import Path

# 1. 读取数据
PROJECT_ROOT = Path(__file__).resolve().parents[1]
test_df = pd.read_csv(PROJECT_ROOT / "01_Data" / "data_splits" / "test.csv")

# 2. 你需要的 5 个 sample_id
target_ids = ["test_300", "test_239", "test_12", "test_49", "test_281"]

# 3. 过滤并提取需要的列
cases = test_df[test_df["sample_id"].isin(target_ids)]

# 4. 打印结果
for _, row in cases.iterrows():
    print("=" * 60)
    print(f"Sample ID: {row['sample_id']}")
    print(f"Question : {row['question']}")
    print(f"Answer   : {row['interview_answer']}\n")
    print(f"Annotator1: {row['annotator1']}")
    print(f"Annotator2: {row['annotator2']}")
    print(f"Annotator3: {row['annotator3']}")
    print(f"Majority  : {row['evasion_label_majority']}")
    print("=" * 60 + "\n")