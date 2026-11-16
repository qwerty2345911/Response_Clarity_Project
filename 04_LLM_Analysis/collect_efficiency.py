from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "04_LLM_Analysis" / "efficiency_results.csv"

TEMPLATE_COLS = [
    "model_name", "params", "train_time", "inference_time", "hardware",
    "batch_size", "epochs", "num_models", "llm_calls", "llm_cost", "notes",
]

def main():
    if not OUT.exists():
        pd.DataFrame(columns=TEMPLATE_COLS).to_csv(OUT, index=False)
        print("created template:", OUT)
        return
    df = pd.read_csv(OUT)
    print(df.to_string())

if __name__ == "__main__":
    main()