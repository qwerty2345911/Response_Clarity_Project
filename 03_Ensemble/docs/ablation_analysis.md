# Ablation Study Analysis

## 1. Experimental Setup

Four configurations were evaluated:

1. Best Individual Model
2. Hard Voting
3. Soft Voting
4. Soft Voting + Temperature Scaling

The best individual model was selected using Macro-F1 on the Validation Set.
Test Set performance was not used for model selection.

Temperature parameters were fitted using Validation Set logits only.
The fitted temperatures were then applied to the Test Set probabilities.

All configurations were evaluated on the same valid Test samples.

## 2. Best Individual Model Selection

Selected model: **roberta**

Validation Macro-F1: **0.3026**

| Model      |   Validation Macro-F1 |
|:-----------|----------------------:|
| roberta    |              0.302642 |
| deberta_v3 |              0.148622 |
| xlnet      |              0.272062 |

## 3. Test Set Samples

| Description | Number |
|---|---:|
| Total Test samples | 308 |
| Evaluated samples | 275 |
| Excluded samples | 33 |

Samples with labels outside the nine official categories were excluded
from metric calculation. Predictions were still generated for all
Test samples.

### Excluded Label Distribution

| Excluded label   |   Sample count |
|:-----------------|---------------:|
| No_Majority      |             33 |

## 4. Evaluation Results

| configuration                     | selected_individual_model   |   validation_macro_f1 |   accuracy |   macro_f1 |    ece |   brier_score |   total_test_samples |   evaluated_samples |   excluded_samples |
|:----------------------------------|:----------------------------|----------------------:|-----------:|-----------:|-------:|--------------:|---------------------:|--------------------:|-------------------:|
| Best Individual Model             | roberta                     |                0.3026 |     0.3491 |     0.2986 | 0.0501 |        0.7530 |                  308 |                 275 |                 33 |
| Hard Voting                       |                             |              nan      |     0.3345 |     0.2698 | 0.4873 |        1.1216 |                  308 |                 275 |                 33 |
| Soft Voting                       |                             |              nan      |     0.3164 |     0.2798 | 0.0976 |        0.7527 |                  308 |                 275 |                 33 |
| Soft Voting + Temperature Scaling |                             |              nan      |     0.3127 |     0.2763 | 0.0894 |        0.7526 |                  308 |                 275 |                 33 |

Metrics:

- Accuracy: proportion of correct predictions.
- Macro-F1: unweighted average of F1 scores across the nine classes.
- ECE: top-label Expected Calibration Error using 15 equal-width bins.
- Brier Score: mean multiclass sum of squared errors between predicted
  probabilities and one-hot targets.

For Hard Voting, class probabilities were calculated as the proportion
of votes received by each class.

For Soft Voting, probabilities were averaged across the three original
models.

For Soft Voting + Temperature Scaling, each model's probabilities were
calibrated separately before averaging.

## 5. Observations

- Highest Test Macro-F1: **Best Individual Model**
  (0.2986).
- Highest Test Accuracy: **Best Individual Model**
  (0.3491).
- Lowest ECE: **Best Individual Model**
  (0.0501).
- Lowest Brier Score: **Soft Voting + Temperature Scaling**
  (0.7526).

These results show the relative performance of the four configurations
on this Test Set. Calibration and classification performance are
reported separately because a lower calibration error does not
necessarily imply a higher classification score.

## 6. Limitations

The results are based on one fixed Test Set. No additional hyperparameter
tuning was performed on the Test Set.

Temperature Scaling was fitted on the Validation Set only.
