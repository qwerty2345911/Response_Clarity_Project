# Data Splitting Protocol

## Dataset

Source: QEvasion Hugging Face dataset.

The processed original training set contains 3448 samples.
The processed original test set contains 308 samples.

## Splitting strategy

The original training split (3448 samples) was divided into:

- Training set: 2700 samples
- Validation set: 748 samples

Stratified sampling was performed based on `evasion_label`.

Random seed: 42

All 308 samples from the original official test split were preserved.

Project-derived majority-vote columns and `sample_id` were added to the
processed test data. The original raw test file remains unchanged.

## Final dataset sizes

| Split | Samples |
|---|---:|
| Training | 2700 |
| Validation | 748 |
| Test | 308 |
| **Total** | **3756** |

## Sample ID

A project-generated `sample_id` column is included in all final split files.

- Training and validation IDs use the `train_` prefix.
- Test IDs use the `test_` prefix.

The IDs support sample identification, prediction alignment,
model-output merging, and error analysis.

All 3756 sample IDs are unique across the three splits.

## Data integrity

- Training and validation samples do not overlap.
- All original training samples are assigned to either training or validation.
- All original test samples are retained.
- The official test split is not used for training or validation.
- The original raw dataset files remain unchanged.
