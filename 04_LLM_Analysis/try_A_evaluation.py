from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "05_Evaluation"))

from evaluation import (  # noqa: E402
    load_test_data,
    evaluate_task1,
    evaluate_task2_annotators,
    evaluate_task2_majority,
    save_results,
)

DUMMY = PROJECT_ROOT / "04_LLM_Analysis" / "dummy"
OUT = PROJECT_ROOT / "04_LLM_Analysis" / "dummy_eval"
OUT.mkdir(parents=True, exist_ok=True)

def main():
    test_df = load_test_data()

    # Task 1
    s, c = evaluate_task1(
        test_df,
        str(DUMMY / "dummy_task1_probabilities.csv"),
        model_name="DummyTask1",
    )
    save_results([s], [c], prefix=str(OUT / "dummy_task1"))

    # Task 2 annotators
    ss, cs = evaluate_task2_annotators(
        test_df,
        str(DUMMY / "dummy_task2_probabilities.csv"),
        model_name="DummyTask2",
    )
    save_results(ss, cs, prefix=str(OUT / "dummy_task2_ann"))

    # Task 2 majority
    sm, cm = evaluate_task2_majority(
        test_df,
        str(DUMMY / "dummy_task2_probabilities.csv"),
        model_name="DummyTask2",
    )
    save_results([sm], [cm], prefix=str(OUT / "dummy_task2_majority"))

    print("done:", OUT)

if __name__ == "__main__":
    main()