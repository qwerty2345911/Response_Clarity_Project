# Member C: Ensemble & Calibration
# 成员 C：集成学习与模型校准

## 1. Overview

Member C is responsible for implementing ensemble methods and probability calibration for the three baseline models: RoBERTa, DeBERTa-v3, and XLNet. The main tasks include Hard Voting, Soft Voting, Temperature Scaling, and Ablation Study.

成员 C 负责对三个基线模型 RoBERTa、DeBERTa-v3 和 XLNet 实现集成学习与概率校准，主要工作包括 Hard Voting、Soft Voting、Temperature Scaling 和 Ablation Study。

## 2. Implementation

### 2.1 Hard Voting

Combines the predicted labels of the three models using majority voting. A probability-based tie-breaking strategy is applied when all three models predict different classes.

将三个模型的预测标签进行多数投票。当三个模型预测结果各不相同时，使用平均预测概率进行平票处理。

Output: `results/hard_voting_predictions.csv`

### 2.2 Soft Voting

Averages the predicted class probabilities of the three models and selects the class with the highest average probability.

对三个模型输出的类别概率取平均，并选择平均概率最高的类别作为最终预测。

Outputs:
- `results/soft_voting_predictions.csv`
- `results/soft_voting_probabilities.csv`

### 2.3 Temperature Scaling

Optimizes one temperature parameter for each model using validation logits and labels by minimizing Negative Log-Likelihood (NLL). The fitted parameters are then applied to test probabilities. Test labels are not used for calibration.

使用 Validation Set 的 logits 和真实标签，通过最小化 Negative Log-Likelihood (NLL) 为每个模型拟合一个温度参数，再将其应用于 Test Set 的预测概率。校准过程中不使用 Test Set 标签。

Outputs:
- `results/calibration_results.csv`
- `results/calibration_comparison.png`
- Calibrated test probability CSV files for all three models
- `artifact/temperature_parameters.json`

### 2.4 Ablation Study

Compares four configurations: Best Individual Model, Hard Voting, Soft Voting, and Soft Voting + Temperature Scaling. The best individual model is selected based on validation Macro-F1.

比较四种配置：Best Individual Model、Hard Voting、Soft Voting 和 Soft Voting + Temperature Scaling。最佳单模型根据 Validation Set 上的 Macro-F1 选出。

Evaluation metrics: Accuracy, Macro-F1, ECE, and Multiclass Brier Score.

评估指标包括 Accuracy、Macro-F1、ECE 和 Multiclass Brier Score。

Outputs:
- `results/C4_ablation_study/ensemble_results.csv`
- `results/C4_ablation_study/ensemble_comparison.png`
- `results/C4_ablation_study/test_label_exclusion_summary.csv`
- `docs/ablation_analysis.md`

## 3. Experimental Results

### 3.1 Temperature Scaling

Validation Set calibration results:

| Model | Temperature | ECE Before | ECE After | Brier Before | Brier After |
|---|---:|---:|---:|---:|---:|
| RoBERTa | 1.1249 | 0.0439 | 0.0236 | 0.7269 | 0.7250 |
| DeBERTa-v3 | 1.0299 | 0.0384 | 0.0468 | 0.7220 | 0.7219 |
| XLNet | 1.1839 | 0.0417 | 0.0392 | 0.7507 | 0.7496 |

Temperature Scaling improved most calibration metrics for RoBERTa and XLNet, while its effect on DeBERTa-v3 was mixed.

Temperature Scaling 改善了 RoBERTa 和 XLNet 的大部分校准指标，但对 DeBERTa-v3 的效果不一致。

### 3.2 Ablation Study

Test Set results (275 valid samples out of 308; 33 `No_Majority` samples excluded):

Test Set 评估结果（308 条样本中，275 条有效样本参与评估，33 条 `No_Majority` 样本被排除）：

| Configuration | Accuracy | Macro-F1 | ECE | Brier Score |
|---|---:|---:|---:|---:|
| Best Individual (RoBERTa) | **0.3491** | **0.2986** | **0.0501** | 0.7530 |
| Hard Voting | 0.3345 | 0.2698 | 0.4873 | 1.1216 |
| Soft Voting | 0.3164 | 0.2798 | 0.0976 | 0.7527 |
| Soft Voting + TS | 0.3127 | 0.2763 | 0.0894 | **0.7526** |

RoBERTa achieved the best Accuracy and Macro-F1. Soft Voting and Temperature Scaling did not improve classification performance over the best individual model, although Temperature Scaling slightly improved calibration metrics.

RoBERTa 取得了最高的 Accuracy 和 Macro-F1。Soft Voting 和 Temperature Scaling 均未能提升分类性能，但 Temperature Scaling 略微改善了校准指标。

## 4. Project Structure

```text
03_Ensemble/
├── artifact/
│   └── temperature_parameters.json
├── docs/
│   └── ablation_analysis.md
├── results/
│   ├── hard_voting_predictions.csv
│   ├── soft_voting_predictions.csv
│   ├── soft_voting_probabilities.csv
│   ├── calibration_results.csv
│   ├── calibration_comparison.png
│   └── C4_ablation_study/
│       ├── ensemble_results.csv
│       ├── ensemble_comparison.png
│       └── test_label_exclusion_summary.csv
├── common.py
├── hard_voting.py
├── soft_voting.py
├── C3_temperature_scaling.py
└── C4_ablation_study.py
```

## 5. How to Run

Run the following commands from the project root directory:

```powershell
python .\03_Ensemble\hard_voting.py
python .\03_Ensemble\soft_voting.py
python .\03_Ensemble\C3_temperature_scaling.py
python .\03_Ensemble\C4_ablation_study.py
```
