# Task 2 Baseline Models

本目录包含成员 B 负责的 Task 2 九分类单模型 Baseline 实验，包括训练环境、三个基础模型的训练与评价代码、预测结果、验证集 logits、资源基准和最终比较材料。

三个模型为：

- RoBERTa-base
- DeBERTa-v3-base
- XLNet-base-cased

本目录依赖 A 组提供的固定数据划分、标签定义和统一评价脚本。所有模型必须使用相同的数据版本和标签顺序；不得重新随机划分数据。

## 1. 任务定义

Task 2 是九分类回答回避类型识别任务。输入由问题和访谈回答组成：

```text
question + interview_answer
```

固定标签顺序如下：

| Label ID | Label | Probability column | Logit column |
|---:|---|---|---|
| 0 | Explicit | `p_Explicit` | `logit_Explicit` |
| 1 | Implicit | `p_Implicit` | `logit_Implicit` |
| 2 | Dodging | `p_Dodging` | `logit_Dodging` |
| 3 | General | `p_General` | `logit_General` |
| 4 | Deflection | `p_Deflection` | `logit_Deflection` |
| 5 | Declining to answer | `p_Declining_to_answer` | `logit_Declining_to_answer` |
| 6 | Claims ignorance | `p_Claims_ignorance` | `logit_Claims_ignorance` |
| 7 | Clarification | `p_Clarification` | `logit_Clarification` |
| 8 | Partial/half-answer | `p_Partial_half_answer` | `logit_Partial_half_answer` |

任何训练、评价、集成和校准程序都必须保持以上顺序不变。

## 2. 数据依赖

本实验使用 A 组提供的数据和评价代码：

```text
01_Data/
├── data_splits/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
└── label_mapping.json

05_Evaluation/
└── evaluation.py

A_data_handover.md
```

固定样本数量：

| Split | Samples | Usage |
|---|---:|---|
| Train | 2,700 | Fine-tuning |
| Validation | 748 | 模型选择和 Temperature Scaling |
| Test | 308 | 最终评价 |
| Test majority-vote subset | 275 | 补充多数标签评价 |

Test Set 中 33 个样本没有有效多数标签，因此多数标签评价仅覆盖 275 个样本。多数标签结果是补充结果，不能替代覆盖全部 308 个样本的三标注者评价。

数据版本以三个模型 `*_experiment_config.json` 中的 `data_source.commit` 为准。合并前应确认三个 commit 完全相同，并将该值与 A 组最终交接版本核对。

## 3. 目录结构

```text
02_Baselines/
├── README.md
│
├── environment/
│   ├── requirements.txt
│   ├── baseline_comparison_requirements.txt
│   └── environment_setup.md
│
├── scripts/
│   ├── train_roberta.py
│   ├── train_deberta.py
│   ├── train_deberta_v3.py
│   ├── train_xlnet.py
│   ├── benchmark_baseline_inference.py
│   └── compare_baselines.py
│
├── docs/
│   ├── roberta_experiment_handover.md
│   ├── deberta_v3_experiment_handover.md
│   ├── xlnet_experiment_handover.md
│   └── baseline_comparison_setup.md
│
├── artifacts/
│   ├── roberta/
│   │   ├── logs/
│   │   ├── predictions/
│   │   └── results/
│   ├── deberta_v3/
│   │   ├── logs/
│   │   ├── predictions/
│   │   └── results/
│   └── xlnet/
│       ├── logs/
│       ├── predictions/
│       └── results/
│
└── comparison/
    ├── baseline_results.csv
    ├── baseline_per_class_results.csv
    ├── baseline_typical_errors.csv
    ├── baseline_input_audit.csv
    ├── inference_benchmark.csv
    ├── baseline_comparison.png
    └── baseline_analysis.md
```

## 4. 模型与实验配置

| Model key | Display name | Hugging Face model ID | Train batch/GPU | Gradient accumulation | Effective train batch |
|---|---|---|---:|---:|---:|
| `roberta` | RoBERTa-base | `FacebookAI/roberta-base` | 8 | 1 | 8 |
| `deberta_v3` | DeBERTa-v3-base | `microsoft/deberta-v3-base` | 4 | 2 | 8 |
| `xlnet` | XLNet-base-cased | `xlnet/xlnet-base-cased` | 4 | 2 | 8 |

统一核心配置：

| Setting | Value |
|---|---:|
| Random seed | 42 |
| Maximum sequence length | 256 |
| Epochs | 3 |
| Learning rate | `2e-5` |
| Effective train batch size | 8 |
| Evaluation batch size | 8 |
| Weight decay | 0.01 |
| Mixed precision | FP16 on CUDA |
| Model selection metric | Validation Macro-F1 |

DeBERTa-v3 和 XLNet 使用较小的显存内单步 Batch Size，并通过梯度累积保持有效训练批量为 8。这是资源适配，不构成实验设置不公平。

XLNet 还需要：

- `tokenizer.padding_side = "left"`
- `use_mems_train = False`
- `use_mems_eval = False`

每个模型的实际配置、环境版本、GPU、训练时间、最佳 checkpoint 和数据 commit 以对应的 `*_experiment_config.json` 为权威记录。

## 5. 环境安装

### 5.1 完整训练环境

在项目根目录执行：

```bash
python -m venv .venv-baselines
```

Linux/macOS：

```bash
source .venv-baselines/bin/activate
python -m pip install --upgrade pip
python -m pip install -r 02_Baselines/environment/requirements.txt
```

Windows PowerShell：

```powershell
.\.venv-baselines\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r 02_Baselines\environment\requirements.txt
```

Kaggle 已包含大部分依赖。不要无必要地升级 Kaggle 的 PyArrow、CUDA 或 RAPIDS 组件，以免破坏预装环境。完整环境检查和 GPU 设置见 [`environment_setup.md`](environment/environment_setup.md)。

### 5.2 仅生成比较结果

如果已经获得三个模型的预测和评价文件，只需要 CPU 环境：

```bash
python -m pip install -r 02_Baselines/environment/baseline_comparison_requirements.txt
```

推理速度与峰值显存必须在同一个 GPU Session 中统一复测，不能用不同 Notebook 中不一致的运行时间直接比较。

## 6. 训练脚本

脚本位于 [`scripts/`](scripts/)。运行前先确认 A 组文件存在：

```text
01_Data/data_splits/train.csv
01_Data/data_splits/validation.csv
01_Data/data_splits/test.csv
01_Data/label_mapping.json
05_Evaluation/evaluation.py
A_data_handover.md
```

先检查每个脚本的可用参数：

```bash
python 02_Baselines/scripts/train_roberta.py --help
python 02_Baselines/scripts/train_deberta.py --help
python 02_Baselines/scripts/train_xlnet.py --help
```

然后分别运行：

```bash
python 02_Baselines/scripts/train_roberta.py
python 02_Baselines/scripts/train_deberta.py
python 02_Baselines/scripts/train_xlnet.py
```

`train_deberta.py` 是满足项目交付文件名要求的入口，完整实现位于 `train_deberta_v3.py`。这两个文件必须同时保留。

每个训练脚本应依次完成：

1. 环境、CUDA 和数据文件检查；
2. 固定随机种子和标签顺序；
3. Smoke test；
4. 正式 Fine-tuning；
5. 按 Validation Macro-F1 选择最佳 checkpoint；
6. 保存最佳模型和训练日志；
7. 导出 Validation/Test 预测、概率和 Validation logits；
8. 调用 A 组统一评价脚本；
9. 执行最终文件、行数和概率校验。

## 7. 单模型交付文件

每个模型必须提交 11 个非空小型文件。以下以 `PREFIX` 表示 `roberta`、`deberta_v3` 或 `xlnet`。

### 7.1 Logs

```text
logs/PREFIX_experiment_config.json
logs/PREFIX_training_log.csv
```

### 7.2 Predictions

```text
predictions/PREFIX_predictions.csv
predictions/PREFIX_probabilities.csv
predictions/PREFIX_validation_predictions.csv
predictions/PREFIX_validation_probabilities.csv
predictions/PREFIX_validation_logits.csv
```

Test prediction schema：

```text
sample_id, predicted_label
```

Test probability schema：

```text
sample_id,
p_Explicit,
p_Implicit,
p_Dodging,
p_General,
p_Deflection,
p_Declining_to_answer,
p_Claims_ignorance,
p_Clarification,
p_Partial_half_answer
```

Validation probabilities 使用完全相同的概率列顺序。Validation logits 使用第 1 节定义的九个 logit 列。

### 7.3 Evaluation results

```text
results/PREFIX_task2_summary.csv
results/PREFIX_task2_per_class.csv
results/PREFIX_task2_majority_summary.csv
results/PREFIX_task2_majority_per_class.csv
```

`task2_summary.csv` 中的 `annotator_mean` 是主要结果，覆盖 308 个样本。`task2_majority_summary.csv` 中的 `supplementary_majority_vote` 是补充结果，覆盖 275 个有效多数标签样本。

## 8. 统一推理基准

三模型推理基准必须满足：

- 同一个 Kaggle Session；
- 同一块 GPU；
- Test Set 308 个样本；
- Batch size 8；
- Maximum length 256；
- 每个模型重复 3 次；
- 不计模型下载和加载时间；
- 计入 tokenization 和 forward inference；
- 每个模型测试后释放显存。

运行示例：

```bash
python 02_Baselines/scripts/benchmark_baseline_inference.py \
  --model-key roberta \
  --model-dir /path/to/roberta/best_model \
  --test-file 01_Data/data_splits/test.csv \
  --output-csv 02_Baselines/comparison/inference_benchmark.csv \
  --batch-size 8 \
  --max-length 256 \
  --repeats 3
```

对 `deberta_v3` 和 `xlnet` 重复运行，写入同一个 `inference_benchmark.csv`。

## 9. 生成最终比较

将三个模型的 11 个文件分别整理到：

```text
02_Baselines/artifacts/roberta/
02_Baselines/artifacts/deberta_v3/
02_Baselines/artifacts/xlnet/
```

然后在项目根目录运行：

```bash
python 02_Baselines/scripts/compare_baselines.py \
  --inputs-root 02_Baselines/artifacts \
  --test-file 01_Data/data_splits/test.csv \
  --benchmark-file 02_Baselines/comparison/inference_benchmark.csv \
  --output-dir 02_Baselines/comparison
```

程序采用严格模式，检查：

- 三个模型各 11 个文件存在且非空；
- Test 308 行、Validation 748 行；
- `sample_id` 唯一且跨模型严格对齐；
- 九分类标签和概率列顺序固定；
- 每行概率和约等于 1；
- Validation logits 有限且为九列；
- 数据 commit 一致；
- 核心超参数和有效训练批量一致；
- 主评价为 `annotator_mean`；
- 多数标签评价为 275 个有效样本。

最终生成：

```text
comparison/baseline_results.csv
comparison/baseline_per_class_results.csv
comparison/baseline_typical_errors.csv
comparison/baseline_input_audit.csv
comparison/baseline_comparison.png
comparison/baseline_analysis.md
```

`baseline_results.csv` 是机器可读的权威汇总；`baseline_analysis.md` 记录模型优缺点、弱类别、高频混淆和实验限制；`baseline_comparison.png` 用于报告和展示。

## 10. 评价口径

主要指标：

- Accuracy
- Macro-F1
- Per-class F1

同时记录：

- Expected Calibration Error（ECE）
- Brier Score
- Validation Macro-F1
- 训练时间
- 推理时间
- Samples per second
- Milliseconds per sample
- 峰值 GPU 显存
- GPU 型号和显存

主报告使用 308 个样本的 `annotator_mean`。275 个样本的 `supplementary_majority_vote` 仅作补充，不应与主结果混为同一统计口径。

为了避免 README 中的数字与重跑结果发生漂移，本目录不在此重复硬编码最终分数。请以 [`baseline_results.csv`](comparison/baseline_results.csv) 和 [`baseline_analysis.md`](comparison/baseline_analysis.md) 为准。

## 11. Kaggle 实验

| Experiment | Kaggle Notebook | Purpose |
|---|---|---|
| RoBERTa-base | [roberta-task2-qevasion-recovered](https://www.kaggle.com/code/wuhuilan0819/roberta-task2-qevasion-recovered) | 训练、预测和评价 |
| DeBERTa-v3-base | [deberta-v3-task2-qevasion](https://www.kaggle.com/code/wuhuilan0819/deberta-v3-task2-qevasion) | 训练、预测和评价 |
| XLNet-base-cased | [xlnet-base-task2-qevasion](https://www.kaggle.com/code/wuhuilan0819/xlnet-base-task2-qevasion) | 训练、预测和评价 |
| Final comparison | https://www.kaggle.com/code/wuhuilan0819/task2-baseline-comparison | 统一推理基准和 Baseline 比较 |

Kaggle Notebook Output 保存最佳模型权重和完整运行记录。GitHub 只保存小型可审阅文件，不保存大型权重和 checkpoint。

## 12. 模型权重与 Git 管理

不要提交以下内容到普通 Git 历史：

- `model.safetensors`
- `pytorch_model.bin`
- `optimizer.pt`
- `checkpoint-*`
- `*_best_model.zip`
- smoke-test 权重
- private backup
- `__pycache__/`
- 克隆产生的嵌套 `.git/`

这些文件可能达到数百 MB 或更大，应继续保存在 Kaggle Notebook Output 中。README、对应交接文档和 PR 描述应提供可访问的 Kaggle 链接。

以下小型文件应提交到 GitHub：

- 训练与比较脚本；
- 环境文件；
- 实验配置和训练日志；
- Test/Validation 预测、概率和 Validation logits；
- 统一评价结果；
- 最终比较表、图和分析文档。

## 13. 交接给成员 C

成员 C 进行集成、校准和 Temperature Scaling，需要以下材料。

每个模型的 Test 输出：

```text
PREFIX_predictions.csv
PREFIX_probabilities.csv
```

每个模型的 Validation 输出：

```text
PREFIX_validation_predictions.csv
PREFIX_validation_probabilities.csv
PREFIX_validation_logits.csv
```

同时提供：

- 三个 `*_experiment_config.json`；
- 标签与概率列顺序；
- 数据 commit；
- 模型 ID 和最佳模型版本；
- `sample_id` 对齐规则；
- 本 README 和三份模型交接文档。

### Temperature Scaling 规则

Temperature 只能使用 Validation logits 和 Validation gold labels 拟合。Test Set 只能用于应用已经拟合好的温度参数并进行一次最终评价，严禁使用 Test Set 拟合温度或选择模型。

## 14. 交接给成员 D

成员 D 用于报告、展示和总体分析的文件：

```text
comparison/baseline_results.csv
comparison/baseline_comparison.png
comparison/baseline_analysis.md
comparison/baseline_per_class_results.csv
comparison/baseline_typical_errors.csv
comparison/inference_benchmark.csv
```

报告中必须注明：

- 主评价是 308 个样本的三标注者平均结果；
- 275 样本多数标签结果只是补充；
- 三模型使用相同的有效训练批量；
- 推理性能在同一 GPU Session 下统一测试；
- DeBERTa-v3 和 XLNet 使用较小单步 Batch Size 加梯度累积是显存适配；
- 典型错误分析只使用具有有效多数标签的样本。

## 15. 复现检查

提交或合并前检查：

- [ ] A 组数据和评价文件存在；
- [ ] 三个 config 的数据 commit 相同；
- [ ] 三个 config 的标签顺序相同；
- [ ] 三个模型各有 11 个非空小型文件；
- [ ] Test 预测和概率各 308 行；
- [ ] Validation 预测、概率和 logits 各 748 行；
- [ ] Test/Validation `sample_id` 跨模型对齐；
- [ ] 所有概率行的和约等于 1；
- [ ] 三模型有效训练批量均为 8；
- [ ] `inference_benchmark.csv` 有三个模型、三行结果；
- [ ] `baseline_input_audit.csv` 有 33 条 PASS 记录；
- [ ] `baseline_results.csv` 有三个模型、三行结果；
- [ ] 比较图四个子图完整可读；
- [ ] 分析文档没有“不完整草稿”标记；
- [ ] Kaggle Notebook 链接有效；
- [ ] Git 中没有大型模型权重或 checkpoint；
- [ ] 成员 C 已获得 Validation logits/probabilities；
- [ ] 成员 D 已获得结果表、比较图和分析文档。

## 16. 已知限制

- 模型在单一数据划分上训练和比较，结果可能受该划分影响；
- 类别不平衡会使 Accuracy 高估多数类别表现，因此主要关注 Macro-F1 和逐类 F1；
- 三名标注者之间存在分歧，多数标签子集仅覆盖 275 个样本；
- 推理速度依赖 Kaggle 当时分配的具体 GPU 和系统负载；
- XLNet 对 padding 和 memory 设置更敏感；
- 模型权重不进入 Git，因此复现者需要通过 Kaggle Output 获取权重或重新训练。

