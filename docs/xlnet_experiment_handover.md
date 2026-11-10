# XLNet-base-cased Task 2 Experiment Handover

## 1. Scope and completion status

This handover covers member B's XLNet-base-cased baseline for Task 2, the nine-class response-evasion classification task. The training, validation prediction, test prediction, A-group evaluation, model backup, and final output audit completed successfully in Kaggle.

The reproducible command-line implementation is `train_xlnet.py`. It was reconstructed from the completed `xlnet-base-task2-qevasion.ipynb` and the exported experiment configuration and evaluation summaries.

## 2. Authoritative experiment record

| Item | Recorded value |
|---|---|
| Model | `xlnet/xlnet-base-cased` |
| Task | Task 2, nine-class classification |
| Input fields | `question`, `interview_answer` |
| Train / validation / test rows | 2700 / 748 / 308 |
| Data branch | `feature/A-data` |
| Data commit | `95069879832d95dfc0262acb1b9559a706a496ec` |
| Random seed | 42 |
| Best checkpoint | `checkpoint-1014` |
| Best validation macro-F1 | 0.272062 |
| Training time | 241.403 seconds / 4.023 minutes |
| GPU | Tesla T4, 14.56 GiB, one visible GPU |

The complete checkpoint path in the Kaggle run was:

```text
/kaggle/working/xlnet_task2_outputs/formal_training/checkpoint-1014
```

## 3. Software environment

The exported experiment configuration recorded:

| Package | Version |
|---|---|
| Python platform | Kaggle Python environment |
| PyTorch | `2.11.0+cu130` |
| Transformers | `5.20.2` |
| pandas | `2.3.3` |
| NumPy | `2.1.3` |

The XLNet tokenizer uses a SentencePiece model. Ensure `sentencepiece` is installed before loading the tokenizer.

Transformers instantiates `XLNetTokenizer` or `XLNetTokenizerFast` and `XLNetForSequenceClassification` for this checkpoint. Both tokenizer variants are acceptable.

## 4. Data contract

The experiment consumes A's processed data and evaluation implementation from:

```text
Response_Clarity_Project/
├── 01_Data/
│   ├── data_splits/
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   └── test.csv
│   └── label_mapping.json
├── 05_Evaluation/
│   └── evaluation.py
└── A_data_handover.md
```

Training and validation require:

```text
sample_id, question, interview_answer, evasion_label
```

Test data require:

```text
sample_id, question, interview_answer,
annotator1, annotator2, annotator3,
evasion_label_majority, majority_vote_count
```

`train_xlnet.py` verifies the files, columns, fixed split sizes, globally unique non-null sample IDs, and label set before downloading the model.

## 5. Fixed label and probability order

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

Do not alphabetize or reorder the labels or probability columns.

## 6. Training configuration

| Parameter | Value |
|---|---:|
| Maximum sequence length | 256 |
| Epochs | 3 |
| Learning rate | `2e-5` |
| Per-device training batch size | 4 |
| Gradient accumulation steps | 2 |
| Effective training batch size | 8 |
| Per-device evaluation batch size | 8 |
| Weight decay | 0.01 |
| Warmup steps | 101 |
| Scheduler | Linear |
| Maximum gradient norm | 1.0 |
| Optimizer | `adamw_torch` |
| Mixed precision | FP16 |
| Tokenizer padding side | Left |
| Recurrent memory during training | Disabled |
| Recurrent memory during evaluation | Disabled |
| Evaluation strategy | Once per epoch |
| Checkpoint strategy | Once per epoch |
| Checkpoint retention | 2 |
| Model-selection metric | Validation macro-F1, higher is better |

The script requires an explicit `--mode` value so that it cannot start a full GPU training run accidentally.

## 7. XLNet-specific reproducibility settings

### 7.1 FP32 master weights

The script explicitly converts every newly loaded model to FP32 before constructing the Trainer:

```python
model = model.float()
```

Trainer still uses `fp16=True`; FP32 master weights are retained while autocast performs mixed-precision computation. This also prevents the FP16 gradient-unscale failure encountered while preparing the preceding DeBERTa baseline.

### 7.2 Left padding

XLNet places its classification token at the end of the sequence. The tokenizer is therefore fixed to left padding:

```python
tokenizer.padding_side = "left"
```

### 7.3 Recurrent memory disabled

Task 2 rows are independent classification examples. Transformer-XL recurrent memory is not used across rows:

```python
model.config.use_mems_train = False
model.config.use_mems_eval = False
```

This prevents unnecessary memory states from being retained or returned during evaluation. Do not change these settings without rerunning the smoke test and output-shape checks.

## 8. Repository preparation

Clone the exact A-data revision used by the completed run:

```bash
git clone \
  --branch feature/A-data \
  --single-branch \
  https://github.com/qwerty2345911/Response_Clarity_Project.git \
  /kaggle/working/Response_Clarity_Project

git -C /kaggle/working/Response_Clarity_Project checkout \
  95069879832d95dfc0262acb1b9559a706a496ec
```

Place the two B-member files under:

```text
Response_Clarity_Project/
└── 02_Baselines/
    ├── train_xlnet.py
    └── xlnet_experiment_handover.md
```

Check the script interface without starting training:

```bash
python 02_Baselines/train_xlnet.py --help
```

## 9. Smoke test command

```bash
python 02_Baselines/train_xlnet.py \
  --mode smoke \
  --project-dir /kaggle/working/Response_Clarity_Project \
  --output-dir /kaggle/working/xlnet_task2_outputs
```

The smoke test uses 64 stratified training rows, 32 stratified validation rows, one epoch, maximum length 128, training batch size 4, and evaluation batch size 8. It checks for finite logits with shape `(32, 9)` and probability rows that sum to one.

Expected completion marker:

```text
SMOKE TEST PASSED
```

Smoke-test metrics are diagnostic and must not be used as the formal result.

## 10. Full reproduction command

```bash
python 02_Baselines/train_xlnet.py \
  --mode train \
  --project-dir /kaggle/working/Response_Clarity_Project \
  --output-dir /kaggle/working/xlnet_task2_outputs \
  --model-id xlnet/xlnet-base-cased \
  --seed 42 \
  --max-length 256 \
  --epochs 3 \
  --learning-rate 2e-5 \
  --train-batch-size 4 \
  --eval-batch-size 8 \
  --gradient-accumulation-steps 2
```

Use `--mode all` instead of `--mode train` to run the smoke test and formal experiment in one invocation.

Do not rerun formal training merely to produce handover files. The supplied result files and backed-up model remain the authoritative completed run.

## 11. Formal evaluation results

### 11.1 Three annotators

| Reference | Samples | Accuracy | Macro-F1 | ECE | Brier score |
|---|---:|---:|---:|---:|---:|
| Annotator 1 | 308 | 0.357143 | 0.276447 | 0.047308 | 0.764362 |
| Annotator 2 | 308 | 0.217532 | 0.207158 | 0.154649 | 0.836331 |
| Annotator 3 | 308 | 0.295455 | 0.267816 | 0.076727 | 0.801087 |
| Annotator mean | 308 | 0.290043 | 0.250474 | 0.092895 | 0.800593 |
| Annotator standard deviation | — | 0.057124 | 0.030831 | 0.045289 | 0.029383 |

### 11.2 Supplementary majority vote

| Reference | Samples | Accuracy | Macro-F1 | ECE | Brier score |
|---|---:|---:|---:|---:|---:|
| Supplementary majority vote | 275 | 0.316364 | 0.269064 | 0.063091 | 0.783448 |

The annotator-based evaluation is primary. Majority vote is supplementary because 33 of the 308 test samples do not have a valid majority and are excluded from that calculation.

## 12. Evaluation implementation

The script imports A's `05_Evaluation/evaluation.py` and calls:

1. `load_test_data()`;
2. `evaluate_task2_annotators()`;
3. `save_results()` using prefix `xlnet_task2`;
4. `evaluate_task2_majority()`;
5. `save_results()` using prefix `xlnet_task2_majority`.

This preserves the shared evaluation protocol rather than duplicating metric logic inside the baseline script.

## 13. Output contract

```text
/kaggle/working/xlnet_task2_outputs/
├── smoke_test/
├── formal_training/
├── best_model/
├── predictions/
│   ├── xlnet_validation_predictions.csv
│   ├── xlnet_validation_probabilities.csv
│   ├── xlnet_validation_logits.csv
│   ├── xlnet_predictions.csv
│   └── xlnet_probabilities.csv
├── results/
│   ├── xlnet_task2_summary.csv
│   ├── xlnet_task2_per_class.csv
│   ├── xlnet_task2_majority_summary.csv
│   └── xlnet_task2_majority_per_class.csv
└── logs/
    ├── xlnet_training_log.csv
    └── xlnet_experiment_config.json
```

Additional archives and backup directory:

```text
/kaggle/working/xlnet_task2_results.zip
/kaggle/working/xlnet_best_model.zip
/kaggle/working/xlnet_task2_private_backup/
├── xlnet_task2_results.zip
├── xlnet_best_model.zip
└── backup_manifest.json
```

The lightweight result ZIP contains predictions, evaluation tables, logs, and configuration. It excludes the large model weights.

## 14. Final artifact validation

The script verifies:

- all required artifacts exist and are non-empty;
- validation files contain 748 rows;
- test files contain 308 rows;
- test sample IDs are unique;
- probability columns follow the fixed nine-class order;
- probability rows sum to one;
- a `model.safetensors` or `pytorch_model.bin` weight file exists;
- model and result ZIP files exist;
- `backup_manifest.json` is non-empty.

Expected final marker:

```text
FINAL XLNET DELIVERY CHECK: PASS
```

## 15. Handover locations

- Completed Notebook source: `xlnet-base-task2-qevasion.ipynb`
- Fixed Kaggle Version URL: not included in the supplied files; insert before final group submission.
- Private model-backup URL: not included in the supplied files; insert before final group submission.
- Source-code branch: member B's baseline branch, recommended `feature/B-baselines`.

Do not commit model weights, checkpoint directories, or the large model ZIP to GitHub. Commit the training script, this handover document, small prediction/evaluation CSV files, configuration JSON, and training log.

## 16. Known limitations

- The best validation macro-F1 is 0.272062, and the annotator-mean test macro-F1 is 0.250474. Per-class evaluation should be reviewed before drawing conclusions about model quality.
- Performance varies across annotators: test accuracy ranges from 0.217532 to 0.357143.
- Majority-vote evaluation covers only 275 samples and must not replace the three-annotator primary evaluation.
- The script intentionally enforces the original split sizes and label contract. A new A-data revision requires explicit review rather than silent reuse.
- Reproduction should use the recorded data commit and environment versions where possible.

## 17. Final checklist

- [ ] `python -m py_compile train_xlnet.py` passes.
- [ ] `python train_xlnet.py --help` displays without starting training.
- [ ] A-data commit is `95069879832d95dfc0262acb1b9559a706a496ec`.
- [ ] Smoke test prints `SMOKE TEST PASSED`.
- [ ] Formal run selects the best checkpoint by validation macro-F1.
- [ ] Best model and tokenizer files are preserved.
- [ ] Validation and test prediction row counts are 748 and 308.
- [ ] Annotator and majority-vote result files are present.
- [ ] Experiment configuration and training log are present.
- [ ] Result ZIP, model ZIP, and non-empty manifest are backed up.
- [ ] Fixed Kaggle Version and private backup links are inserted above.
- [ ] Small deliverables are committed to member B's branch.
