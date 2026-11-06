# Dataset Description

## 1. Dataset Overview

The following statistics describe the processed source datasets.

| Dataset | Samples | Columns | Duplicate rows |
|---|---:|---:|---:|
| Train | 3448 | 20 | 0 |
| Test | 308 | 22 | 0 |

The original datasets contain 20 columns each.
The processed Test dataset contains 22 columns after adding two derived majority-vote fields.

## 2. Data Processing Pipeline

The main data preparation steps are:

1. `preprocess_dataset.py`: Clean and preprocess the original datasets.
2. `add_majority_vote.py`: Generate majority-vote labels for the test set using three annotators.
3. `create_data_splits.py`: Split the original training data into training and validation sets while preserving the official test set.

The scripts should be executed in the order listed above, from the project root directory.


## 3. Dataset Fields

### Train

title, date, president, url, question_order, interview_question, interview_answer, gpt3.5_summary, gpt3.5_prediction, question, annotator_id, annotator1, annotator2, annotator3, inaudible, multiple_questions, affirmative_questions, index, clarity_label, evasion_label

### Test

title, date, president, url, question_order, interview_question, interview_answer, gpt3.5_summary, gpt3.5_prediction, question, annotator_id, annotator1, annotator2, annotator3, inaudible, multiple_questions, affirmative_questions, index, clarity_label, evasion_label, evasion_label_majority, majority_vote_count

### Additional derived fields in processed Test

- `evasion_label_majority`: Project-generated majority-vote label based on the three annotators.
- `majority_vote_count`: Number of annotators supporting the majority label.

## 4. Missing Values

| Dataset | Field | Missing count |
|---|---|---:|
| Train | annotator1 | 3448 |
| Train | annotator2 | 3448 |
| Train | annotator3 | 3448 |
| Test | title | 308 |
| Test | date | 308 |
| Test | president | 308 |
| Test | gpt3.5_summary | 308 |
| Test | gpt3.5_prediction | 308 |
| Test | annotator_id | 308 |
| Test | evasion_label | 308 |

## 5. Label Distributions

### Train: clarity_label

| Label | Count | Percentage |
|---|---:|---:|
| Ambivalent | 2040 | 59.16% |
| Clear Reply | 1052 | 30.51% |
| Clear Non-Reply | 356 | 10.32% |

### Train: evasion_label

| Label | Count | Percentage |
|---|---:|---:|
| Explicit | 1052 | 30.51% |
| Dodging | 706 | 20.48% |
| Implicit | 488 | 14.15% |
| General | 386 | 11.19% |
| Deflection | 381 | 11.05% |
| Declining to answer | 145 | 4.21% |
| Claims ignorance | 119 | 3.45% |
| Clarification | 92 | 2.67% |
| Partial/half-answer | 79 | 2.29% |

### Test: clarity_label

| Label | Count | Percentage |
|---|---:|---:|
| Ambivalent | 206 | 66.88% |
| Clear Reply | 79 | 25.65% |
| Clear Non-Reply | 23 | 7.47% |

### Test: annotator1

| Label | Count | Percentage |
|---|---:|---:|
| Explicit | 106 | 34.42% |
| Dodging | 58 | 18.83% |
| Implicit | 54 | 17.53% |
| Deflection | 30 | 9.74% |
| General | 29 | 9.42% |
| Declining to answer | 10 | 3.25% |
| Claims ignorance | 9 | 2.92% |
| Partial/half-answer | 8 | 2.60% |
| Clarification | 4 | 1.30% |

### Test: annotator2

| Label | Count | Percentage |
|---|---:|---:|
| General | 78 | 25.32% |
| Dodging | 72 | 23.38% |
| Implicit | 54 | 17.53% |
| Explicit | 53 | 17.21% |
| Deflection | 22 | 7.14% |
| Claims ignorance | 11 | 3.57% |
| Declining to answer | 9 | 2.92% |
| Partial/half-answer | 5 | 1.62% |
| Clarification | 4 | 1.30% |

### Test: annotator3

| Label | Count | Percentage |
|---|---:|---:|
| Explicit | 80 | 25.97% |
| Implicit | 67 | 21.75% |
| General | 65 | 21.10% |
| Dodging | 43 | 13.96% |
| Deflection | 23 | 7.47% |
| Declining to answer | 14 | 4.55% |
| Claims ignorance | 7 | 2.27% |
| Partial/half-answer | 5 | 1.62% |
| Clarification | 4 | 1.30% |

## 6. Test Set Majority Voting

The three original annotator labels are preserved.
A majority-vote label is generated from annotator1, annotator2, and annotator3.

### 6.1 Vote Count Distribution

| Voting result | Samples | Percentage |
|---|---:|---:|
| 3 votes agree | 125 | 40.58% |
| 2 votes agree | 150 | 48.70% |
| No majority | 33 | 10.71% |
| Missing annotations | 0 | 0.00% |
| Total | 308 | 100.00% |

- Samples with a majority label: **275**.
- Samples without a majority: **33**.

### 6.2 Majority Vote Label Distribution

Percentages below are calculated among samples with a valid majority label.

| Label | Count | Percentage |
|---|---:|---:|
| Explicit | 77 | 28.00% |
| Implicit | 57 | 20.73% |
| Dodging | 50 | 18.18% |
| General | 48 | 17.45% |
| Deflection | 17 | 6.18% |
| Declining to answer | 11 | 4.00% |
| Claims ignorance | 8 | 2.91% |
| Clarification | 4 | 1.45% |
| Partial/half-answer | 3 | 1.09% |

### 6.3 Majority Voting Rules

- Three annotators agree: use their shared label.
- Two annotators agree: use the label shared by two annotators.
- Three annotators disagree: assign `No_Majority`.
- Incomplete annotations: assign `Missing_Annotation`.
- The original annotator labels are not modified.
- Majority-vote labels are project-generated supplementary references, not official gold labels.

The samples without a majority label are retained in the test set and are not removed from the primary evaluation.

## 7. Sample ID

A unique `sample_id` is assigned to every sample in the final train, validation, and test splits.

| Split | Samples | Unique sample IDs |
|---|---:|---:|
| Train | 2700 | 2700 |
| Validation | 748 | 748 |
| Test | 308 | 308 |
| **Total** | **3756** | **3756** |

- Train and Validation IDs use the `train_` prefix.
- Test IDs use the `test_` prefix.
- All sample IDs are globally unique across the three splits.
- Sample IDs are used to identify samples, align predictions and probabilities across models, merge model outputs, and support error analysis.

## 8. Train-Test Leakage Check

- Exact normalized QA pairs shared by Train and Test: **0**.
- Normalization: lowercase text, collapse repeated whitespace, and strip leading/trailing whitespace.
- This check does not detect paraphrases or near-duplicate answers.

## 9. Key Observations

- The training set contains clarity and evasion labels.
- The test set contains clarity labels and three original annotator columns for evasion labels.
- The test set's original evasion_label column is empty.
- An additional majority-vote label and vote-count column are available in the processed test set.
- The final data splits contain unique sample IDs.
- No exact normalized question-answer overlap was found between Train and Test.

## 10. Notes

Statistics are generated from the processed datasets.
The raw datasets were not modified during this analysis.
The original official Test split is preserved, while majority-vote fields are added as supplementary references.
The final training data was divided into training and validation subsets using stratified sampling.
Task 2 evaluation compares model predictions separately against the three annotators. Majority-vote evaluation is supplementary and uses samples with a valid majority label.
The exact Macro-F1 implementation should be verified against the official evaluator, if available.