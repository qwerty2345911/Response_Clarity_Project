# A组工作交接文档：数据处理与评估

## 1. 项目概况

**项目名称：** Confidence-Aware Model Ensemble and LLM Reasoning for Response Clarity Classification

**数据集：** QEvasion

**数据来源：** Hugging Face — `ailsntua/QEvasion`

**原始论文：**

Thomas, K., Filandrianos, G., Lymperaiou, M., Zerva, C., & Stamou, G. (2024).  
"I Never Said That": A dataset, taxonomy and baselines on response clarity classification.  
Findings of EMNLP 2024.

**A组负责内容：**
- 数据集检查与预处理
- 标签映射与标签转换
- 训练集、验证集和测试集划分
- 测试集 Majority Vote 生成
- 数据统计与数据质量检查
- 统一评估程序与评估协议
- 为其他组提供统一的数据接口和预测文件规范

---

## 2. 数据集基本情况

原始 QEvasion 数据集包含：

| Dataset | Samples | Original Columns |
|---|---:|---:|
| Train | 3,448 | 20 |
| Test | 308 | 20 |
| Total | 3,756 | - |

处理后的 Test 数据集增加了两个项目生成的字段：

- `evasion_label_majority`
- `majority_vote_count`

因此，处理后的 Test 数据集共有 22 列。

### 2.1 两个分类任务

**Task 1：Clarity Classification**

三分类任务，预测回答的清晰程度：

1. Clear Reply
2. Ambivalent
3. Clear Non-Reply

参考标签来自 `clarity_label`。

**Task 2：Evasion Classification**

九分类任务，预测回答采用的具体回避方式：

1. Explicit
2. Implicit
3. Dodging
4. General
5. Deflection
6. Declining to answer
7. Claims ignorance
8. Clarification
9. Partial/half-answer

训练集和验证集的参考标签来自 `evasion_label`。

测试集使用三个独立标注者的标签：

- `annotator1`
- `annotator2`
- `annotator3`

测试集原始 `evasion_label` 列为空，因此 Task 2 测试评估不能使用该列作为参考标签。

---

## 3. 项目文件结构

主要目录结构如下：

```text
Response_Clarity_Project/
│
├── README.md
├── A_data_handover.md
│
├── 01_Data/
│   ├── raw/
│   ├── processed/
│   ├── data_splits/
│   │
│   ├── inspect_dataset.py
│   ├── save_dataset_report.py
│   ├── preprocess_dataset.py
│   ├── validate_labels.py
│   ├── label_mapping.json
│   ├── label_mapping.md
│   ├── label_conversion.py
│   ├── add_majority_vote.py
│   ├── create_data_splits.py
│   ├── preprocessing_config.json
│   │
│   ├── dataset_statistics.csv
│   └── data_description.md
│
└── 05_Evaluation/
    ├── evaluation.py
    ├── evaluation_protocol.md
    └── test_evaluation.py
```

### 3.1 主要数据文件

| 文件 | 用途 |
|---|---|
| `raw/train.csv` | 原始训练数据 |
| `raw/test.csv` | 原始测试数据 |
| `processed/train_cleaned.csv` | 清洗后的训练数据 |
| `processed/test_cleaned.csv` | 清洗并添加 Majority Vote 字段的测试数据 |
| `data_splits/train.csv` | 最终训练集 |
| `data_splits/validation.csv` | 最终验证集 |
| `data_splits/test.csv` | 最终测试集 |
| `data_splits/split_statistics.csv` | 数据划分统计 |
| `data_splits/split_summary.json` | 数据划分摘要 |
| `data_splits/split_protocol.md` | 数据划分协议 |

---

## 4. 数据处理流程

核心数据处理程序按照以下顺序运行。

所有命令均从项目根目录执行。

### Step 1：检查原始数据

运行：

```bat
python 01_Data/inspect_dataset.py
```

作用：
- 检查 Train 和 Test 的数据规模。
- 检查数据列名。
- 检查缺失值。
- 检查重复行。
- 检查标签分布。
- 检查 Train 和 Test 是否存在完全重复的标准化 QA 样本。

已完成检查，原始 Train 和 Test 均没有重复的完整数据行，标准化 QA 重合数量为 0。

### Step 2：数据预处理

运行：

```bat
python 01_Data/preprocess_dataset.py
```

作用：
- 清理文本中的多余空格。
- 统一文本格式。
- 检查必要字段。
- 保留原始数据样本。
- 生成处理后的 Train 和 Test 文件。

输出：
- `processed/train_cleaned.csv`
- `processed/test_cleaned.csv`

处理结果：
- Train：3,448 行，20 列。
- Test：308 行，20 列。

原始 raw 数据文件保持不变。

### Step 3：生成 Majority Vote

运行：

```bat
python 01_Data/add_majority_vote.py
```

作用：

根据 Test 数据中的三个标注者标签生成：

- `evasion_label_majority`
- `majority_vote_count`

目前统计结果：

| Voting Result | Samples |
|---|---:|
| Three annotators agree | 125 |
| Two annotators agree | 150 |
| No majority | 33 |
| Missing annotations | 0 |
| Total | 308 |

共有 275 个样本具有有效多数票标签，33 个样本没有多数票。

原始三个标注者的标签均予以保留。

### Step 4：生成数据划分

运行：

```bat
python 01_Data/create_data_splits.py
```

作用：
- 从原始训练集划分 Training 和 Validation。
- 按 `evasion_label` 进行分层抽样。
- 使用固定随机种子 42。
- 保留全部 308 个官方测试样本。
- 为所有数据生成唯一 `sample_id`。
- 检查数据划分和样本 ID 的完整性。

最终数据划分：

| Split | Samples |
|---|---:|
| Training | 2,700 |
| Validation | 748 |
| Test | 308 |
| **Total** | **3,756** |

注意：该划分是本项目采用的可复现划分，不代表完全复现原论文的数据划分。

### Step 5：生成数据统计报告

运行：

```bat
python 01_Data/save_dataset_report.py
```

作用：
- 生成数据集统计文件。
- 生成数据描述文档。
- 汇总标签分布。
- 汇总 Majority Vote 统计。
- 检查最终数据集的 sample ID。

输出：
- `01_Data/dataset_statistics.csv`
- `01_Data/data_description.md`

该脚本用于生成报告，不属于前三个核心数据处理步骤。

### 4.1 标签转换检查

运行：

```bat
python 01_Data/label_conversion.py
```

该程序负责：
- Task 1 标签编码和解码。
- Task 2 标签编码和解码。
- 标签列表转换。
- Task 2 到 Task 1 的层级映射。

已完成测试，输出结果为：

`ALL TESTS PASSED!`

---

## 5. 标签映射规范

所有组员必须使用统一的标签名称和类别顺序。

### 5.1 Task 1

| Label | ID |
|---|---:|
| Clear Reply | 0 |
| Ambivalent | 1 |
| Clear Non-Reply | 2 |

### 5.2 Task 2

| Label | ID |
|---|---:|
| Explicit | 0 |
| Implicit | 1 |
| Dodging | 2 |
| General | 3 |
| Deflection | 4 |
| Declining to answer | 5 |
| Claims ignorance | 6 |
| Clarification | 7 |
| Partial/half-answer | 8 |

### 5.3 Task 2 到 Task 1 的层级映射

| Task 2 | Task 1 |
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

该映射来自原始 QEvasion 分类体系。

如果原始 `clarity_label` 已经存在，应直接使用它作为 Task 1 标签，而不是重新从 Task 2 标签生成。

详细标签规范见：

- `01_Data/label_mapping.json`
- `01_Data/label_mapping.md`

---

## 6. Sample ID 规范

最终三个数据集均包含 `sample_id`。

| Split | Sample ID Format |
|---|---|
| Training | `train_<original_index>` |
| Validation | `train_<original_index>` |
| Test | `test_<original_index>` |

Training 和 Validation 均来自原始训练集，因此共享 `train_` 前缀。

Test 使用 `test_` 前缀。

全部 3,756 个 sample ID 均唯一。

### Sample ID 的用途

- 唯一样本识别。
- 对齐不同模型的预测结果。
- 对齐预测概率和预测标签。
- 合并不同模型的输出。
- 进行错误分析。
- 检查不同模型是否使用了相同测试样本。

其他组提交预测文件时，必须保留测试集原有的 `sample_id`，不得自行重新编号。

---

## 7. 测试集评估规则

### 7.1 Task 1

使用测试集的 `clarity_label` 作为真实标签。

所有模型均在相同的 308 个测试样本上进行评估。

### 7.2 Task 2：三个标注者分别评估

使用相同模型预测结果，分别与以下三个标注者的标签比较：

- `annotator1`
- `annotator2`
- `annotator3`

评估程序分别计算三个标注者的结果，并进一步计算三组指标的算术平均值和总体标准差。

这两项统计反映不同标注者参考标签下的评估结果差异，不是将三个标注者合并成一个统一的真实标签集合。

### 7.3 Majority Vote 补充评估

另外使用 `evasion_label_majority` 进行补充评估。

- 有效 Majority Vote 样本：275。
- No_Majority 样本：33。

只有具有有效多数票标签的样本参与该补充评估。

注意：
- 这 33 个样本仅在 Majority Vote 补充评估中排除。
- 它们仍然参与三个原始标注者的主要评估。
- Majority Vote 是本项目生成的补充参考标签，不应描述为原论文官方真实标签。

---

## 8. 各组预测文件提交规范

其他组完成模型后，需要向 A 组提供测试集预测概率文件。

### 8.1 Task 1 概率文件

文件必须包含：

```csv
sample_id,p_Clear_Reply,p_Ambivalent,p_Clear_Non_Reply
test_0,0.10,0.80,0.10
test_1,0.75,0.15,0.10
```

### 8.2 Task 2 概率文件

文件必须包含：

```csv
sample_id,p_Explicit,p_Implicit,p_Dodging,p_General,p_Deflection,p_Declining_to_answer,p_Claims_ignorance,p_Clarification,p_Partial_half_answer
test_0,0.10,0.10,0.20,0.10,0.10,0.10,0.10,0.10,0.10
```

以上概率仅用于展示格式，不是真实模型预测结果。

### 8.3 预测文件要求

1. 每个样本必须有唯一 `sample_id`。
2. 必须包含全部 308 个测试样本。
3. 不允许缺少或重复 sample ID。
4. 不允许出现额外的测试集之外的 ID。
5. 概率列必须按照统一类别顺序排列。
6. 所有概率必须处于 [0, 1] 范围内。
7. 每行所有类别的概率之和必须约等于 1。

### 8.4 Sample ID 自动匹配

A 组的 `evaluation.py` 已经实现自动匹配。

因此：
- 不要求其他组按照测试集原始行顺序提交。
- 程序会根据 `sample_id` 自动重新排列预测结果。
- 程序会检查缺失、重复和额外的 ID。
- 所有 308 个测试样本必须完整提交。

### 8.5 可选预测标签文件

如果其他组同时提供预测类别，可以另外提交：

```csv
sample_id,predicted_label
test_0,Ambivalent
test_1,Clear Reply
```

预测标签必须与对应概率文件中最大概率所对应的类别一致。

该文件是可选的。评估程序可以直接根据概率计算预测类别。

Task 2 的预测结果会分别与三个标注者的参考标签进行比较。

真实标签由 A 组评估程序从共享测试集读取，其他组不需要重复提供。

---

## 9. 评估指标

A 组评估程序报告以下指标：

### 9.1 Accuracy

正确分类样本占总样本的比例。

### 9.2 Macro-F1

所有预定义类别 F1 分数的算术平均值。

所有类别均参与计算，包括在当前参考标签中没有样本的类别。

### 9.3 Per-class Metrics

分别报告每个类别的：

- Precision
- Recall
- F1-score
- Support

### 9.4 Expected Calibration Error (ECE)

使用 Top-label confidence 计算 ECE。

- 使用 15 个等宽置信度区间。
- 计算预测置信度与实际准确率之间的差异。
- ECE 越低，表示预测置信度与实际准确率越接近。

### 9.5 Multiclass Brier Score

按照当前评估程序的实现：

对每个样本，计算预测概率向量与 one-hot 真实标签向量之间的平方误差之和，再对所有样本取平均。

Brier Score 越低越好。

---

## 10. 评估程序使用方法

评估程序位于：

`05_Evaluation/evaluation.py`

主要函数：

| Function | Purpose |
|---|---|
| `load_test_data()` | 加载共享测试集 |
| `evaluate_task1()` | Task 1 三分类评估 |
| `evaluate_task2_annotators()` | Task 2 三个标注者分别评估 |
| `evaluate_task2_majority()` | Task 2 Majority Vote 补充评估 |
| `save_results()` | 保存评估结果 |

### 10.1 Task 1 示例

从 `05_Evaluation/` 目录运行示例代码：

```python
from evaluation import (
    load_test_data,
    evaluate_task1,
    save_results
)

test_df = load_test_data()

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
```

### 10.2 Task 2 示例

```python
from evaluation import (
    load_test_data,
    evaluate_task2_annotators,
    evaluate_task2_majority,
    save_results
)

test_df = load_test_data()

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

`prediction_file` 是可选参数。如果没有单独的预测标签文件，可以省略。

---

## 11. 评估输出文件

评估程序可以生成：

- `<prefix>_summary.csv`
- `<prefix>_per_class.csv`

其中：

- `summary.csv` 保存整体评估指标。
- `per_class.csv` 保存各类别的 Precision、Recall、F1 和 Support。

各模型的独立评估结果由对应组员提交。

最终整合文件：
- `all_results.csv`
- `result_analysis.md`

由负责最终结果整合的组员维护。

---


## 12. 其他组员需要遵守的事项

1. 使用 A 组提供的统一数据划分。
2. 不要自行重新划分测试集。
3. 不要修改共享标签名称和类别顺序。
4. 保留所有原始测试集 sample ID。
5. Task 1 使用三类 clarity 标签。
6. Task 2 使用九类 evasion 标签。
7. Task 2 测试结果需要分别对三个标注者进行评估。
8. Majority Vote 只能作为补充评估。
9. 提交完整的 308 条测试预测概率。
10. 确保每行概率之和约等于 1。
11. 不要将模拟预测结果作为真实模型实验结果。
12. 不要使用测试集进行模型训练或模型选择。

---

## 13. 相关文档

| Document | Purpose |
|---|---|
| `01_Data/data_description.md` | 数据集描述与统计 |
| `01_Data/label_mapping.md` | 标签体系与层级映射 |
| `01_Data/label_mapping.json` | 机器可读标签配置 |
| `01_Data/preprocessing_config.json` | 数据预处理配置 |
| `01_Data/data_splits/split_protocol.md` | 数据划分协议 |
| `01_Data/data_splits/split_summary.json` | 数据划分摘要 |
| `05_Evaluation/evaluation_protocol.md` | 统一评估协议 |
| `05_Evaluation/evaluation.py` | 评估程序 |
| `05_Evaluation/test_evaluation.py` | 评估程序测试 |

---
