import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "04_LLM_Analysis" / "llm_outputs"

for setting in ["llm_only", "llm_assisted"]:
    file_path = OUT_DIR / f"llm_results_{setting}.csv"
    if not file_path.exists():
        print(f"找不到 {file_path}")
        continue
    
    df = pd.read_csv(file_path)
    total_time = df["latency_sec"].sum()
    mean_time = df["latency_sec"].mean()
    
    print(f"=== {setting} 真实耗时 ===")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均单条延迟: {mean_time:.2f} 秒")
    print(f"每秒处理量: {len(df) / total_time:.2f} 条/秒")
    print("-" * 30)