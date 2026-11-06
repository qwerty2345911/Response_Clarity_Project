
# Label Mapping and Original Taxonomy

## 1. Project Information

**Project:** Confidence-Aware Model Ensemble and LLM Reasoning for Response Clarity Classification

**Dataset:** ailsntua/QEvasion

**Source Paper:**

Thomas, K., Filandrianos, G., Lymperaiou, M., Zerva, C., & Stamou, G. (2024).

"I Never Said That": A dataset, taxonomy and baselines on response clarity classification.

Findings of EMNLP 2024.

DOI: https://doi.org/10.18653/v1/2024.findings-emnlp.300

---

## 2. Original QEvasion Taxonomy

The original QEvasion taxonomy follows a two-level hierarchical structure.

- Level 1: Clarity Level (3 classes)
- Level 2: Evasion Level (9 classes)

The nine fine-grained evasion labels are grouped into three high-level clarity categories.

### 2.1 Original Taxonomy Mapping

| Level 1: Clarity Level | Level 2: Evasion Level |
|---|---|
| Clear Reply | Explicit |
| Ambivalent Reply | Implicit |
| Ambivalent Reply | General |
| Ambivalent Reply | Partial |
| Ambivalent Reply | Dodging |
| Ambivalent Reply | Deflection |
| Clear Non-Reply | Declining to answer |
| Clear Non-Reply | Claims ignorance |
| Clear Non-Reply | Clarification |

**Important naming note:**

The original paper uses `Ambivalent Reply` and `Partial`.

The released Hugging Face dataset uses `Ambivalent` and `Partial/half-answer`.

This project preserves the original dataset label names in all CSV files and code.

---

## 3. Task 1: Clarity-level Classification

### 3.1 Task Definition

Task 1 predicts the high-level clarity category of an interview response based on the question and answer.

Input:
- Question
- Interview answer

Output:
- One of three clarity labels.

### 3.2 Label Mapping

Original dataset column: `clarity_label`

| Label | Label ID | Description |
|---|---:|---|
| Clear Reply | 0 | The response clearly provides the information requested by the question. |
| Ambivalent | 1 | The response is unclear or allows multiple interpretations. |
| Clear Non-Reply | 2 | The response does not provide the requested information. |

### 3.3 Task 1 Rules

- Use the original `clarity_label` column.
- Preserve the original dataset label names.
- Do not independently change the class order.
- Use the same label-to-ID mapping across all models.

---

## 4. Task 2: Evasion-level Classification

### 4.1 Task Definition

Task 2 predicts the fine-grained evasion technique used in an interview response.

Input:
- Question
- Interview answer

Output:
- One of nine evasion labels.

### 4.2 Label Mapping

Training dataset column: `evasion_label`

Test dataset original annotator columns:
- `annotator1`
- `annotator2`
- `annotator3`

| Label | Label ID | Description |
|---|---:|---|
| Explicit | 0 | The requested information is explicitly stated in the expected form. |
| Implicit | 1 | The requested information is conveyed indirectly rather than explicitly. |
| Dodging | 2 | The response ignores the question altogether. |
| General | 3 | The response is too general and lacks the requested specificity. |
| Deflection | 4 | The response starts on topic but shifts the focus to a different point. |
| Declining to answer | 5 | The respondent directly or indirectly refuses to answer. |
| Claims ignorance | 6 | The respondent claims not to know the requested information. |
| Clarification | 7 | The respondent asks for clarification instead of providing the requested information. |
| Partial/half-answer | 8 | The response provides only a specific component of the requested information. |

**Naming note:** The original paper uses `Partial`; the released dataset uses `Partial/half-answer`.

---

## 5. Mapping Between Task 2 and Task 1

The original taxonomy defines the following hierarchical relationship.

| Task 2: Evasion Label | Task 1: Clarity Label |
|---|---|
| Explicit | Clear Reply |
| Implicit | Ambivalent |
| General | Ambivalent |
| Partial/half-answer | Ambivalent |
| Dodging | Ambivalent |
| Deflection | Ambivalent |
| Declining to answer | Clear Non-Reply |
| Claims ignorance | Clear Non-Reply |
| Clarification | Clear Non-Reply |

This mapping is part of the original two-level taxonomy.

It may be used to understand the relationship between the two classification tasks. It does not mean that Task 1 labels should be regenerated from Task 2 labels when the original `clarity_label` is available.

---

## 6. Test Set Annotations and Majority Voting

### 6.1 Original Annotator Labels

The test dataset contains three annotator columns:

- `annotator1`
- `annotator2`
- `annotator3`

All three original annotations must be preserved.

### 6.2 Project-derived Majority Voting

This project additionally generates:

- `evasion_label_majority`
- `majority_vote_count`

The rules are:

| Annotation Situation | Derived Label | Vote Count |
|---|---|---:|
| All three annotators agree | Shared label | 3 |
| Two annotators agree | Shared label | 2 |
| All three annotators disagree | No_Majority | 0 |
| One or more annotations are missing | Missing_Annotation | 0 |

Current test set statistics:

| Voting Result | Samples |
|---|---:|
| Three annotators agree | 125 |
| Two annotators agree | 150 |
| No majority | 33 |
| Missing annotations | 0 |
| **Total** | **308** |

### 6.3 Evaluation Warning

`No_Majority` and `Missing_Annotation` are annotation-status labels, not official classification classes.

The majority-vote column is a project-derived reference. It must not be described as the official evaluation method of the original paper without further verification.

The original three annotator labels must remain available for multi-annotator evaluation.


---

## 7. Data Column Reference

| Dataset | Column | Purpose |
|---|---|---|
| Train | `clarity_label` | Task 1 reference labels |
| Train | `evasion_label` | Task 2 reference labels |
| Test | `clarity_label` | Task 1 reference labels |
| Test | `annotator1` | First annotator's Task 2 label |
| Test | `annotator2` | Second annotator's Task 2 label |
| Test | `annotator3` | Third annotator's Task 2 label |
| Test | `evasion_label_majority` | Project-derived majority-vote label |
| Test | `majority_vote_count` | Number of agreeing annotators |

---

## 8. Rules for All Group Members

1. Use the same label-to-ID mapping in all experiments.
2. Preserve original dataset label names.
3. Do not independently reorder the classes.
4. Task 1 uses `clarity_label`.
5. Task 2 training uses `evasion_label`.
6. Preserve all three original test annotator columns.
7. Do not treat `No_Majority` as a tenth class.
8. Do not silently discard samples without a majority.
9. Clearly state the evaluation protocol used for Task 2.
10. Refer to this document and `label_mapping.json` as the shared label mapping specification.

---

## 9. Related Files

- `label_mapping.json`: Machine-readable label definitions and hierarchical mappings.
- `label_mapping.md`: Human-readable taxonomy and label documentation.
- `processed/train_cleaned.csv`: Processed training dataset.
- `processed/test_cleaned.csv`: Processed test dataset.
- `dataset_statistics.csv`: Dataset statistics.
- `data_description.md`: Dataset description and data quality report.

---

**Source:** Thomas et al. (2024), "I Never Said That": A dataset, taxonomy and baselines on response clarity classification, especially the original taxonomy and label descriptions.

**Note:** This document records the original taxonomy and the group's current data conventions. 