# RoBERTa-base Task 2 Experiment Handover

## 1. Purpose and status

This handover covers member B's RoBERTa-base baseline for Task 2, the nine-class response-evasion classification task. The implementation was extracted from `roberta-task2-qevasion-recovered.ipynb` and reorganized as the command-line program `train_roberta.py`.

The original Kaggle run completed formal training and reported the following validation results:

| Metric | Value |
|---|---:|
| Validation loss | 1.619310 |
| Validation accuracy | 0.403743 |
| Validation macro-F1 | 0.314743 |
| Training loss | 1.720903 |

These figures were reported from the completed Kaggle run. The downloaded notebook contains code but no saved cell outputs, so the authoritative machine-readable record remains `logs/roberta_experiment_config.json` from the completed run.

## 2. Source data and required repository state

The experiment consumes A's processed data and evaluation implementation:

- Repository: `qwerty2345911/Response_Clarity_Project`
- Branch: `feature/A-data`
- Data handover: `A_data_handover.md`
- Split directory: `01_Data/data_splits/`
- Evaluation module: `05_Evaluation/evaluation.py`

Expected split sizes:

| Split | Rows |
|---|---:|
| Train | 2700 |
| Validation | 748 |
| Test | 308 |

The training and validation files must contain `sample_id`, `question`, `interview_answer`, and `evasion_label`. The test file must contain `sample_id`, `question`, `interview_answer`, `annotator1`, `annotator2`, `annotator3`, `evasion_label_majority`, and `majority_vote_count`.

`train_roberta.py` validates these files, their columns, their row counts, globally unique sample IDs, and the complete nine-class label set before loading the model.

Record the exact A-data commit used for the final submission with:

```bash
git -C /kaggle/working/Response_Clarity_Project rev-parse HEAD
```

The full training run also writes this commit automatically into `roberta_experiment_config.json`.

## 3. Label order

The probability columns and model configuration use this fixed order:

| ID | Label | Probability column |
|---:|---|---|
| 0 | Explicit | `p_Explicit` |
| 1 | Implicit | `p_Implicit` |
| 2 | Dodging | `p_Dodging` |
| 3 | General | `p_General` |
| 4 | Deflection | `p_Deflection` |
| 5 | Declining to answer | `p_Declining_to_answer` |
| 6 | Claims ignorance | `p_Claims_ignorance` |
| 7 | Clarification | `p_Clarification` |
| 8 | Partial/half-answer | `p_Partial_half_answer` |

Do not alphabetize or otherwise reorder these labels.

## 4. Model inputs and training configuration

- Model: `FacebookAI/roberta-base`
- Architecture: `AutoModelForSequenceClassification`
- Number of labels: 9
- Inputs: `question` and `interview_answer` passed to the tokenizer as a sequence pair
- Maximum sequence length: 256
- Epochs: 3
- Learning rate: `2e-5`
- Training batch size per device: 8
- Evaluation batch size per device: 16
- Gradient accumulation steps: 1
- Weight decay: 0.01
- Warmup: 10% of estimated training steps
- Scheduler: linear
- Maximum gradient norm: 1.0
- Optimizer: `adamw_torch`
- Random and data seed: 42
- Mixed precision: fp16 on CUDA
- Evaluation and checkpoint frequency: once per epoch
- Model-selection metric: validation macro-F1
- Checkpoints retained: 2

The script requires an explicit `--mode` argument. This prevents an accidental full training run when someone only intends to inspect the command.

## 5. Environment preparation

In a Kaggle Notebook, enable a GPU accelerator and Internet access before cloning the repository or downloading the Hugging Face model.

Clone A's branch if the repository is not already present:

```bash
git clone \
  --depth 1 \
  --branch feature/A-data \
  --single-branch \
  https://github.com/qwerty2345911/Response_Clarity_Project.git \
  /kaggle/working/Response_Clarity_Project
```

Place `train_roberta.py` in `02_Baselines/`:

```text
Response_Clarity_Project/
├── 01_Data/
├── 02_Baselines/
│   ├── train_roberta.py
│   └── roberta_experiment_handover.md
├── 05_Evaluation/
└── A_data_handover.md
```

Check the CLI without starting training:

```bash
python /kaggle/working/Response_Clarity_Project/02_Baselines/train_roberta.py --help
```

## 6. Smoke test

Run the smoke test before a new full experiment:

```bash
python /kaggle/working/Response_Clarity_Project/02_Baselines/train_roberta.py \
  --mode smoke \
  --project-dir /kaggle/working/Response_Clarity_Project \
  --output-dir /kaggle/working/roberta_task2_outputs
```

The smoke test uses 64 stratified training rows, 32 stratified validation rows, one epoch, maximum length 128, training batch size 4, and evaluation batch size 8. A successful run ends with:

```text
SMOKE TEST PASSED
```

It also checks that the logits have shape `(32, 9)`, are finite, and produce probability rows that sum to one. Smoke-test metrics are diagnostic only and must not be reported as the formal result.

## 7. Full training and evaluation

The command below reproduces the Notebook's formal configuration:

```bash
python /kaggle/working/Response_Clarity_Project/02_Baselines/train_roberta.py \
  --mode train \
  --project-dir /kaggle/working/Response_Clarity_Project \
  --output-dir /kaggle/working/roberta_task2_outputs \
  --model-id FacebookAI/roberta-base \
  --seed 42 \
  --max-length 256 \
  --epochs 3 \
  --learning-rate 2e-5 \
  --train-batch-size 8 \
  --eval-batch-size 16
```

Use `--mode all` instead of `--mode train` to run the smoke test immediately before formal training.

Do not rerun formal training merely to create the two handover files. The existing successful weights and result files remain valid. Rerun only when reproducing the experiment, correcting the implementation, or generating a deliberately new run.

## 8. Output layout

With the default Kaggle paths, the script creates:

```text
/kaggle/working/roberta_task2_outputs/
├── smoke_test/
├── formal_training/
├── best_model/
├── predictions/
│   ├── roberta_validation_predictions.csv
│   ├── roberta_validation_probabilities.csv
│   ├── roberta_validation_logits.csv
│   ├── roberta_predictions.csv
│   └── roberta_probabilities.csv
├── results/
│   ├── roberta_task2_summary.csv
│   ├── roberta_task2_per_class.csv
│   ├── roberta_task2_majority_summary.csv
│   └── roberta_task2_majority_per_class.csv
└── logs/
    ├── roberta_training_log.csv
    └── roberta_experiment_config.json
```

The script also creates:

```text
/kaggle/working/roberta_task2_results.zip
```

The lightweight ZIP contains predictions, evaluation tables, the training log, and the experiment configuration. It deliberately excludes the large model weights.

## 9. Prediction contract

`roberta_predictions.csv` contains one row per test sample:

```text
sample_id,predicted_label
```

`roberta_probabilities.csv` contains `sample_id` followed by the nine probability columns in the fixed label order from Section 3. The script checks that the test logits have shape `(308, 9)`, that probability rows sum to one, and that test `sample_id` values remain unique.

The corresponding validation files contain 748 rows. The validation prediction file additionally includes `gold_label`; the validation logits file retains the raw nine-class logits.

## 10. Evaluation protocol

The script imports A's `05_Evaluation/evaluation.py` instead of duplicating evaluation logic. It runs:

1. `load_test_data()`;
2. `evaluate_task2_annotators()`;
3. `save_results()` with prefix `roberta_task2`;
4. `evaluate_task2_majority()`;
5. `save_results()` with prefix `roberta_task2_majority`.

The primary evaluation compares predictions separately with annotators 1, 2, and 3 and reports aggregate statistics. Majority-vote results are supplementary because samples without a valid majority are excluded.

The four evaluation CSV files are first written by A's module under `05_Evaluation/` and then copied into the experiment's `results/` directory.

## 11. Model and result backup

The best model is stored under `best_model/`. A complete backup should preserve the model configuration, weights, tokenizer configuration, vocabulary, merges, and special-token files.

Recommended handover artifacts:

- Current Kaggle Notebook: <https://www.kaggle.com/code/wuhuilan0819/roberta-task2-qevasion-recovered/edit>
- Fixed Kaggle Notebook version: replace the edit URL with the successful saved Version URL before final submission.
- Lightweight results: `roberta_task2_results.zip`.
- Model archive: `roberta_best_model.zip` from the completed run.
- Private Kaggle Dataset or approved shared storage: add the model-backup URL here before final submission.

Do not commit the approximately 479 MB model archive to GitHub. Commit the script, this handover document, and the small result tables instead.

## 12. Known limitations and interpretation

- Validation macro-F1 is substantially lower than validation accuracy, indicating uneven performance across the nine classes. Per-class tables should be used when interpreting the model.
- The downloaded Notebook did not retain its executed outputs. The metrics in Section 1 came from the completed run report, while exact environment, GPU, run time, checkpoint path, and A-data commit should be read from the saved `roberta_experiment_config.json`.
- The script retains the Notebook's strict expected split sizes. A new data revision will intentionally fail validation until the split contract is reviewed and updated.
- `processing_class=tokenizer` and `eval_strategy` follow the Transformers API that successfully ran in the Kaggle Notebook. Use the pinned project environment when reproducing the run.
- CPU execution is available only through the explicit `--allow-cpu` option. It disables fp16 and is not recommended for formal training.

## 13. Final handover checklist

- [ ] `train_roberta.py` passes `python -m py_compile`.
- [ ] `python train_roberta.py --help` displays without starting training.
- [ ] A-data branch and exact commit are recorded.
- [ ] Smoke test prints `SMOKE TEST PASSED`.
- [ ] Best model and tokenizer files are preserved.
- [ ] Validation and test prediction files have 748 and 308 rows respectively.
- [ ] Probability columns follow the fixed nine-class order.
- [ ] Annotator and majority-vote evaluation CSV files are present.
- [ ] `roberta_experiment_config.json` and `roberta_training_log.csv` are present.
- [ ] A fixed Kaggle Version link is inserted into this document.
- [ ] The large model archive is backed up outside GitHub.
- [ ] Script, handover document, and small result files are committed to member B's branch.
