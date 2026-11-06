
# Evaluation Protocol

## 1. Overview

This document defines the shared evaluation protocol for the
Response Clarity Classification project.

All group members must use the same test set, sample IDs,
label order, probability column order, and evaluation metrics.

Evaluation is performed separately for:

- Task 1: Three-class clarity classification.
- Task 2: Nine-class evasion classification.

## 2. Shared Dataset

The shared dataset files are located in:

`01_Data/data_splits/`

| Split | Number of samples |
|---|---:|
| Train | 2,700 |
| Validation | 748 |
| Test | 308 |
| Total | 3,756 |

The training and validation sets were generated from the
original training split using stratified sampling based on
`evasion_label`, with random seed 42.

The original test split was preserved.

Every sample has a unique `sample_id`.

- Training samples: `train_<original_index>`
- Validation samples: `train_<original_index>`
- Test samples: `test_<original_index>`

All prediction files must preserve these IDs.

## 3. Label Definitions and Order

### 3.1 Task 1: Clarity Classification

The three classes must follow this exact order:

1. Clear Reply
2. Ambivalent
3. Clear Non-Reply

The reference labels are stored in the `clarity_label`
column of the shared dataset.

### 3.2 Task 2: Evasion Classification

The nine classes must follow this exact order:

1. Explicit
2. Implicit
3. Dodging
4. General
5. Deflection
6. Declining to answer
7. Claims ignorance
8. Clarification
9. Partial/half-answer

The training and validation reference labels are stored in
the `evasion_label` column.

The test set contains three independent annotations:

- `annotator1`
- `annotator2`
- `annotator3`

These annotations must be evaluated separately.

## 4. Prediction File Format

### 4.1 Probability Files

Each model must provide probability estimates for every
sample in the shared test set.

Task 1 probability file:

`sample_id, p_Clear_Reply, p_Ambivalent, p_Clear_Non_Reply`

Task 2 probability file:

`sample_id, p_Explicit, p_Implicit, p_Dodging, p_General, p_Deflection, p_Declining_to_answer, p_Claims_ignorance, p_Clarification, p_Partial_half_answer`

Requirements:

1. Each row must contain one unique `sample_id`.
2. All 308 test sample IDs must be included.
3. Probability columns must follow the fixed class order.
4. Probabilities must be between 0 and 1.
5. Each row's probabilities must sum to 1.
6. Probability files must not contain missing or duplicate IDs.

The evaluation code aligns predictions with the test set
using `sample_id`. Therefore, row order does not need to be
identical, provided that all IDs match exactly.

### 4.2 Predicted-label Files

A separate predicted-label file may be provided with:

`sample_id, predicted_label`

The predicted label must correspond to the class with the
highest predicted probability.

For Task 2, the predicted label is evaluated independently
against each annotator's reference label.

The evaluator checks that predicted labels agree with the
probability argmax when a predicted-label file is supplied.

## 5. Evaluation Metrics

The following metrics are reported:

### 5.1 Accuracy

The proportion of correctly classified samples.

### 5.2 Macro-F1

The unweighted average of F1 scores across all predefined
classes.

All classes are included in the calculation, including
classes with zero support in the evaluated reference set.

### 5.3 Per-class Precision, Recall and F1

Precision, recall, F1-score, and support are reported
separately for every class.

### 5.4 Expected Calibration Error (ECE)

ECE measures the difference between prediction confidence
and empirical accuracy.

The implementation uses top-label confidence and 15
equally spaced confidence bins over [0, 1].

Lower ECE indicates better calibration.

### 5.5 Multiclass Brier Score

The multiclass Brier score is calculated as the mean,
across samples, of the sum of squared differences between
predicted class probabilities and one-hot reference labels.

Lower Brier scores indicate better probabilistic predictions.

## 6. Task 2 Evaluation Protocol

### 6.1 Individual Annotator Evaluation

The same Task 2 predictions are evaluated independently
against all three annotators.

The evaluator reports:

- annotator1
- annotator2
- annotator3

It also reports the arithmetic mean and population standard
deviation across the three annotator-level metric values.

These aggregate statistics describe variation across
annotator references. They are not a pooled evaluation over
a single combined gold-label set.

### 6.2 Supplementary Majority-vote Evaluation

A supplementary evaluation is performed using
`evasion_label_majority`.

Only samples with a valid majority-vote label are included.

The current test set contains:

- 275 samples with majority-vote labels.
- 33 samples without a majority vote.

The 33 samples without a majority vote are excluded only
from this supplementary evaluation.

They remain included in the primary evaluation against
each of the three annotators.

The majority-vote labels are project-derived supplementary
references, not assumed to be official gold labels.

## 7. Evaluation Code

The shared evaluation implementation is:

`05_Evaluation/evaluation.py`

It provides:

- `load_test_data()`
- `evaluate_task1()`
- `evaluate_task2_annotators()`
- `evaluate_task2_majority()`
- `save_results()`

Example usage:

```python
from evaluation import (
    load_test_data,
    evaluate_task1,
    evaluate_task2_annotators,
    evaluate_task2_majority,
    save_results
)

test_df = load_test_data()

# Task 1
summary, class_table = evaluate_task1(
    test_df,
    "path/to/task1_probabilities.csv",
    model_name="RoBERTa",
    prediction_file="path/to/task1_predictions.csv"
)

save_results(
    [summary],
    [class_table],
    prefix="roberta_task1"
)

# Task 2
summaries, class_tables = evaluate_task2_annotators(
    test_df,
    "path/to/task2_probabilities.csv",
    model_name="RoBERTa",
    prediction_file="path/to/task2_predictions.csv"
)

save_results(
    summaries,
    class_tables,
    prefix="roberta_task2"
)

# Supplementary majority-vote evaluation
majority_summary, majority_classes = evaluate_task2_majority(
    test_df,
    "path/to/task2_probabilities.csv",
    model_name="RoBERTa",
    prediction_file="path/to/task2_predictions.csv"
)

save_results(
    [majority_summary],
    [majority_classes],
    prefix="roberta_task2_majority"
)
```

The prediction file arguments are optional.

## 8. Output Files

The evaluation code can generate:

- `<prefix>_summary.csv`
- `<prefix>_per_class.csv`

These are individual evaluation outputs.

The consolidated `all_results.csv` and overall
`result_analysis.md` are maintained by the group member
responsible for final result integration.

## 9. Reproducibility and Restrictions

1. All group members must use the shared split files.
2. The test set must not be used for model training or
   model selection.
3. All models must use the same fixed label order.
4. All prediction files must include valid sample IDs.
5. Probability rows must sum to 1.
6. Test predictions must be aligned by sample ID.
7. Task 2 results must be reported separately for the
   three annotators.
8. Majority-vote evaluation must be identified as
   supplementary analysis.
9. Simulated predictions are for software testing only
   and must not be reported as model results.
10. This project's train-validation split is a
    reproducible project split, not a claim to reproduce
    the exact original paper split.

