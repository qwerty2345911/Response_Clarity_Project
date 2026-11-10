# Task 2 九分类 Baseline 比较分析

## 评价口径

- 主要测试结果采用 A 组统一评价脚本的 `annotator_mean`，覆盖全部 308 个测试样本。
- `supplementary_majority_vote` 仅覆盖具有多数标签的 275 个样本，因此只作为补充结果。
- 模型选择使用 Validation Macro-F1；Test Set 未用于选择模型、调参或拟合 Temperature。
- 三个模型使用相同数据提交、标签顺序、随机种子、最大长度、学习率、训练轮数和有效训练批量。

## 总体结果

- 测试集 Accuracy 最佳：**RoBERTa-base**（0.3247）。
- 测试集 Macro-F1 最佳：**RoBERTa-base**（0.2903）。
- 多数标签 Macro-F1 最佳：**RoBERTa-base**（0.2986）。
- 训练时间最短：**RoBERTa-base**（2.72 分钟）。
- 推理吞吐量最高：**RoBERTa-base**（294.01 samples/s）。
- 峰值 GPU 显存最低：**RoBERTa-base**（0.52 GiB）。

## 各模型分析

### RoBERTa-base

- 主测试 Accuracy：0.3247。
- 主测试 Macro-F1：0.2903。
- 训练时间：2.72 分钟。
- 推理吞吐量：294.01 samples/s。
- 峰值 GPU 显存：0.52 GiB。
- 优点：训练和部署流程成熟，资源消耗相对可控，适合作为稳定基准。
- 局限：对细粒度回避类别和语义相近类别的区分能力可能有限。
- F1 最低的三个类别：Partial/half-answer (0.000)、General (0.010)、Deflection (0.031)。
- 高频混淆：Implicit → Explicit（38）、General → Explicit（27）、Dodging → Explicit（17）、Deflection → Explicit（10）、General → Dodging（9）。

### DeBERTa-v3-base

- 主测试 Accuracy：0.2965。
- 主测试 Macro-F1：0.1576。
- 训练时间：5.24 分钟。
- 推理吞吐量：137.91 samples/s。
- 峰值 GPU 显存：0.87 GiB。
- 优点：解耦注意力机制适合细粒度语义分类。
- 局限：参数量及显存需求更高，需要通过较小 Batch Size 和梯度累积训练。
- F1 最低的三个类别：Implicit (0.000)、General (0.000)、Deflection (0.000)。
- 高频混淆：Implicit → Explicit（43）、General → Explicit（36）、Dodging → Explicit（25）、Deflection → Explicit（13）、Implicit → Dodging（12）。

### XLNet-base-cased

- 主测试 Accuracy：0.2900。
- 主测试 Macro-F1：0.2505。
- 训练时间：4.02 分钟。
- 推理吞吐量：169.51 samples/s。
- 峰值 GPU 显存：0.55 GiB。
- 优点：排列语言建模提供了不同于掩码语言模型的归纳偏置。
- 局限：模型结构较旧，训练和推理资源效率可能低于更新的编码器模型。
- F1 最低的三个类别：Partial/half-answer (0.000)、Deflection (0.020)、General (0.054)。
- 高频混淆：Implicit → Explicit（50）、General → Explicit（36）、Dodging → Explicit（31）、Deflection → Explicit（15）、Explicit → Dodging（6）。

## 公平性和限制

- 数据 commit：`95069879832d95dfc0262acb1b9559a706a496ec`。
- 有效训练批量：8。
- 推理性能在同一个 Kaggle GPU Session 中，以 batch size 8、max length 256、3 次重复测量。
- 多数标签分析排除了没有多数标签的 33 个测试样本，不能与完整 308 样本结果直接混用。
- Validation probabilities/logits 供成员 C 在 Validation Set 上拟合 Temperature；不得在 Test Set 上拟合。

## 相关交付物

- `baseline_results.csv`：三模型总体指标、训练配置和资源记录。
- `baseline_comparison.png`：总体指标、逐类 F1 和效率比较。
- `baseline_per_class_results.csv`：三模型九分类逐类 F1。
- `baseline_typical_errors.csv`：至少两个模型同时预测错误的代表样本。
- `baseline_input_audit.csv`：33 个输入文件的完整性记录。
