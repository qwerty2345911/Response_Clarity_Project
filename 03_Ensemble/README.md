# Member C: Ensemble & Calibration
# 成员 C：集成学习与模型校准

## 1. Overview

Member C is responsible for implementing ensemble methods and probability calibration for the three baseline models: RoBERTa, DeBERTa-v3, and XLNet. The main tasks include Hard Voting, Soft Voting, Temperature Scaling, and Ablation Studies.

成员 C 负责对三个基线模型 RoBERTa、DeBERTa-v3 和 XLNet 实现集成学习与概率校准，主要工作包括 Hard Voting、Soft Voting、Temperature Scaling 和消融实验。

## 2. Implementation

### 2.1 Hard Voting

Combines the predicted labels of the three models using majority voting. A probability-based tie-breaking strategy is applied when all three models predict different classes.

将三个模型预测的类别进行多数投票。当三个模型预测结果各不相同时，使用平均预测概率进行平票处理。

Output:
- `results/hard_voting_predictions.csv`

### 2.2 Soft Voting

Averages the predicted class probabilities of the three models and selects the class with the highest average probability.

对三个模型输出的类别概率取平均，并选择平均概率最高的类别作为最终预测。

Outputs:
- `results/soft_voting_predictions.csv`
- `results/soft_voting_probabilities.csv`

### 2.3 Shared Soft Voting Utilities

A shared utility module, `soft_voting_utils.py`, is introduced to avoid duplicating the probability-loading and voting logic across different experiments.

The module provides two reusable functions:

- `load_probabilities()`: Loads model probability files and validates the required columns, sample IDs, probability values, and probability distributions.
- `soft_voting()`: Aligns predictions by `sample_id`, averages the probability matrices, and generates predicted labels and probability outputs.

通过新增公共模块 `soft_voting_utils.py`，避免在不同实验中重复编写概率文件读取和 Soft Voting 计算逻辑。

该模块提供两个可复用函数：

- `load_probabilities()`：读取模型概率文件，并检查必要列、样本 ID、概率值及概率分布是否合法。
- `soft_voting()`：根据 `sample_id` 对齐不同模型的预测结果，对概率矩阵取平均，并生成预测标签和概率输出。

Both C2 Soft Voting and C4 Model Ablation reuse these functions. Their input paths and experiment configurations remain independently controlled by their respective scripts.

C2 Soft Voting 和 C4 Model Ablation 均调用上述公共函数。两个实验仍分别管理自己的输入路径和实验配置。

### 2.4 Temperature Scaling

Optimizes one temperature parameter for each model using validation logits and labels by minimizing Negative Log-Likelihood (NLL). The fitted parameters are then applied to test probabilities. Test labels are not used for calibration.

使用 Validation Set 的 logits 和真实标签，通过最小化 Negative Log-Likelihood (NLL) 为每个模型拟合一个温度参数，再将其应用于 Test Set 的预测概率。校准过程中不使用 Test Set 标签。

Outputs:
- `results/calibration_results.csv`
- `results/calibration_comparison.png`
- Calibrated test probability CSV files for all three models
- `artifact/temperature_parameters.json`

### 2.5 Ensemble Strategy Ablation Study

Compares four configurations:

1. Best Individual Model
2. Hard Voting
3. Soft Voting
4. Soft Voting + Temperature Scaling

The best individual model is selected based on validation Macro-F1.

比较四种配置：

1. 最佳单模型
2. Hard Voting
3. Soft Voting
4. Soft Voting + Temperature Scaling

最佳单模型根据 Validation Set 上的 Macro-F1 选出。

Evaluation metrics: Accuracy, Macro-F1, ECE, and Multiclass Brier Score.

评估指标包括 Accuracy、Macro-F1、ECE 和 Multiclass Brier Score。

Outputs:
- `results/C4_ablation_study/ensemble_results.csv`
- `results/C4_ablation_study/ensemble_comparison.png`
- `results/C4_ablation_study/test_label_exclusion_summary.csv`
- `docs/ablation_analysis.md`

### 2.6 Model Ablation Study (C4)

In addition to comparing ensemble strategies, a separate model ablation experiment is implemented to investigate the contribution of each baseline model to Soft Voting performance.

The experiment evaluates four model combinations:

| Experiment | Models |
|---|---|
| C4-1 Full Ensemble | RoBERTa + DeBERTa-v3 + XLNet |
| C4-2 Without RoBERTa | DeBERTa-v3 + XLNet |
| C4-3 Without DeBERTa-v3 | RoBERTa + XLNet |
| C4-4 Without XLNet | RoBERTa + DeBERTa-v3 |

除了比较不同集成策略外，还增加了独立的模型消融实验，用于分析各基线模型对 Soft Voting 性能的贡献。

实验比较四种模型组合：

| 实验 | 参与模型 |
|---|---|
| C4-1 完整集成 | RoBERTa + DeBERTa-v3 + XLNet |
| C4-2 去除 RoBERTa | DeBERTa-v3 + XLNet |
| C4-3 去除 DeBERTa-v3 | RoBERTa + XLNet |
| C4-4 去除 XLNet | RoBERTa + DeBERTa-v3 |

All configurations use equal-weight Soft Voting. The predicted class is selected using the argmax of the averaged probability distribution.

所有组合均采用等权 Soft Voting，即对参与模型的类别概率取算术平均，并通过 argmax 得到最终预测类别。

Both Validation and Test probability files are processed separately. The experiment uses the probability files stored under `02_Baselines/artifacts/`.

Validation 和 Test 的概率文件分别处理。该实验使用 `02_Baselines/artifacts/` 下的模型预测文件。

Outputs:
- `results/C4_model_ablation/C4-1_full_ensemble_validation_predictions.csv`
- `results/C4_model_ablation/C4-1_full_ensemble_validation_probabilities.csv`
- Corresponding Validation and Test prediction/probability files for all four configurations
- `results/C4_model_ablation/C4_summary.csv`

当前 C4 Model Ablation 脚本负责生成四组实验的 Validation 和 Test 预测文件，尚未在该脚本中计算 Accuracy、Macro-F1、ECE 或 Brier Score。最终性能比较需要接入统一 Evaluation 流程。

The experiment reuses `soft_voting_utils.py` rather than implementing probability loading and Soft Voting independently.

该实验复用 `soft_voting_utils.py`，不再单独重复实现概率读取和 Soft Voting 算法。

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

### 3.2 Ensemble Strategy Ablation Study

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

### 3.3 Model Ablation Study

The model ablation experiment generates predictions for four Soft Voting configurations using different combinations of the three baseline models.

模型消融实验针对三个基线模型的不同组合，生成四组 Soft Voting 预测结果。

The generated outputs include both Validation and Test predictions and probability distributions. Quantitative performance comparisons will be conducted using the project's shared evaluation pipeline.

实验输出包括 Validation 和 Test 的预测标签及概率分布。各组合的定量性能比较将通过项目统一 Evaluation 流程完成。

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
│   ├── C4_ablation_study/
│   │   ├── ensemble_results.csv
│   │   ├── ensemble_comparison.png
│   │   └── test_label_exclusion_summary.csv
│   ├── C4_model_ablation/
│   │   ├── C4-1_full_ensemble_...
│   │   ├── C4-2_without_roberta_...
│   │   ├── C4-3_without_deberta_...
│   │   ├── C4-4_without_xlnet_...
│   │   └── C4_summary.csv
│   └── ...
├── common.py
├── hard_voting.py
├── soft_voting.py
├── soft_voting_utils.py
├── C3_temperature_scaling.py
├── C4_ablation_study.py
└── C4_model_ablation.py
```

Note: The structure above summarizes the main files and outputs. Actual output files depend on which experiments have been executed.

注：以上目录结构展示主要代码文件和结果文件。实际生成的文件取决于已经运行的实验。

## 5. How to Run

Run the following commands from the project root directory:

在项目根目录下运行以下命令：

```powershell
python .\03_Ensemble\hard_voting.py
python .\03_Ensemble\soft_voting.py
python .\03_Ensemble\C3_temperature_scaling.py
python .\03_Ensemble\C4_ablation_study.py
python .\03_Ensemble\C4_model_ablation.py
```

The C2 Soft Voting and C4 Model Ablation scripts both import shared functions from `soft_voting_utils.py`. Make sure this file is located in the same directory as the two scripts.

C2 Soft Voting 和 C4 Model Ablation 均从 `soft_voting_utils.py` 导入公共函数。请确保该文件与两个脚本位于同一目录下。