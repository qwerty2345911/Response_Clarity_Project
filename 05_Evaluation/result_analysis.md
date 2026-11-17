# 综合实验结果与分析 (Result Analysis)

## 1. 实验设置与评估口径
本实验基于 QEvasion 数据集，包含 2700 条训练集与 308 条测试集。为了评估的公平性，主结果均在 **275 条具备有效多数票** 的测试样本上计算（排除了 33 条人类标注者无共识的样本）。核心评价指标包括 **Macro-F1**（应对类别不平衡的主指标）、**Accuracy** 以及衡量置信度可靠性的 **ECE** 和 **Brier Score**。

## 2. 实验结果总表
下表汇总了单模型、集成与校准以及大语言模型扩展的全量实验结果：

| 实验 ID | 模块 | 模型配置 | 评估口径 | Accuracy | Macro-F1 | ECE | Brier | 推理耗时 | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B1 | Baseline | RoBERTa-base | Single | 0.3491 | 0.2986 | 0.0501 | 0.7530 | 1.05s | 传统单模型最优 |
| B2 | Baseline | DeBERTa-v3-base | Single | 0.3200 | 0.1651 | 0.0810 | 0.7681 | 2.23s | 参数量最大，表现最差 |
| B3 | Baseline | XLNet-base-cased | Single | 0.3164 | 0.2691 | 0.0631 | 0.7834 | 1.82s | 表现居中 |
| C1 | Ensemble | Hard Voting | Ensemble | 0.3345 | 0.2698 | 0.4873 | 1.1216 | N/A | 硬投票无法输出可靠概率 |
| C2 | Ensemble | Soft Voting | Ensemble | 0.3164 | 0.2798 | 0.0976 | 0.7527 | N/A | 集成性能低于最优单模型 |
| C3 | Ensemble | Soft Voting + TS | Calibration | 0.3127 | 0.2763 | 0.0894 | 0.7526 | N/A | 校准改善 ECE，未提升 F1 |
| D1 | LLM | LLM-only | Zero-shot | 0.3745 | 0.4234 | N/A | N/A | 1470.8s | 纯 Prompt 零样本降维打击 |
| **D2** | **LLM** | **LLM-assisted** | **Reasoning** | **0.4291** | **0.4766** | **N/A** | **N/A** | **1996.3s** | **全场最优，F1 提升近 60%** |

*(注：由于 LLM 为生成式模型，其置信度基于 Token 的 Log-probs，与判别式模型的 Softmax 分布数学特性不同，故未计算传统 ECE/Brier。)*

## 3. 关键发现
### 3.1 传统 PLM 的局限与集成失效

实验表明，单模型表现中 **RoBERTa（Macro-F1: 0.2986）显著优于 DeBERTa 和 XLNet**。然而，按照预期，将这三个模型进行集成（Hard/Soft Voting）理应带来鲁棒性的提升，但 **Soft Voting 的 Macro-F1 反而降至 0.2798**。

结合成员 D 生成的混淆矩阵（见 `figures/` 目录）可以发现根因：这三个 Transformer 编码器属于**高度同质化模型**。它们均陷入了极其严重的**“多数类偏置”（Explicit Bias）**——大量真实的 `Implicit`（间接回答）和 `General`（笼统回答）被它们错误预测为 `Explicit`（明确回答）。因为三个模型犯的是完全相同的系统性错误，投票机制无法提供额外的信息增益，反而放大了这种偏差。

此外，Temperature Scaling（温度缩放）虽然成功将 ECE 从 0.0976 降至 0.0894（使概率分布更可靠），但依然无法挽救分类准确率（F1 进一步降至 0.2763），这说明**校准只能改变置信度，不能改变决策边界**。

为了探明各模型对集成的真实贡献，我们进行了模型消融测试。结果如下：

C4-1 Full Ensemble (ALL)  | Accuracy: 0.3164 | Macro-F1: 0.2758
C4-2 Without RoBERTa      | Accuracy: 0.3055 | Macro-F1: 0.2549
C4-3 Without DeBERTa      | Accuracy: 0.3164 | Macro-F1: 0.2742
C4-4 Without XLNet        | Accuracy: 0.3273 | Macro-F1: 0.2811


当移除核心模型 RoBERTa 时，Macro-F1 暴跌至 0.2549；然而，当移除 XLNet（仅保留 RoBERTa+DeBERTa）时，Macro-F1 竟然回升至 0.2811，反而成为了表现最好的集成组合。但值得注意的是，即便是最好的集成组合，依然低于单模型 RoBERTa（0.2986）。这深刻地揭示了：在同质化严重的情况下，简单的多数投票或概率平均只会引入“噪声”（如 XLNet 的负面干扰），无法带来异构模型集成的多样性红利。

### 3.2 LLM 带来的破局

在传统模型全军覆没的类别上，引入大语言模型（DeepSeek）展现出了“降维打击”的效果。纯零样本的 **LLM-only 取得了 0.4234 的 Macro-F1**，不仅远超所有单模型，甚至打破了局部类别的识别盲区（如成功识别出 `Clarification` 和 `Declining to answer` 等长尾类）。

当我们把传统模型的集成概率作为先验信息输入给 LLM（**LLM-assisted**）时，模型取得了 **0.4766 的巅峰 Macro-F1**。相比于最佳单模型 RoBERTa，**性能大幅提升了 59.6%**。这证明了：在细粒度、高度主观的文本分类任务中，同质小模型的集成往往收益有限；而引入具备强大常识推理能力的**异构大模型**，并通过上下文进行辅助引导（Reasoning Enhancement），是打破性能天花板的关键。

### 3.3 性能与效率的权衡 (Trade-off)
虽然 LLM-assisted 表现惊艳，但根据效率分析（`efficiency_analysis.md`），其代价极为高昂：
* **RoBERTa** 只需 **3.4 毫秒**就能跑完一条样本。
* **LLM-assisted** 受限于 API 网络与超长 Prompt，单条延迟高达 **6480 毫秒（6.48 秒）**，慢了近 2000 倍。

## 4. 案例与效率结论
- **错误分析**：传统模型错在“看不懂上下文”，过度依赖字面词频，导致 `Implicit` 大量误报为 `Explicit`;LLM 虽然语义理解极强，但出现了新的“过度推理”倾向——它倾向于把各种模棱两可的回答统统归类为 `Deflection`（转移话题），这说明 Zero-shot 的 Prompt 对边界定义仍需进一步约束。。
- **效率分析**：LLM 的引入带来了近 1000 倍的延迟代价，实际部署时需结合业务场景在“极致性能”与“推理成本”之间进行权衡。

## 5. 结论与展望
本项目充分验证了 Confidence-Aware Model Ensemble and LLM Reasoning 的潜力与局限。基于实验结果，我们建议在实际工业落地中采用**“级联架构（Cascade Architecture）”**：
1. **第一层初筛**：使用推理仅需 3.4ms 的 RoBERTa 处理绝大多数简单直接的样本。
2. **第二层精判**：当 RoBERTa 的置信度较低（如低于阈值），或者涉及需要复杂逻辑推理的长尾类别时，再将其路由给 LLM-assisted 进行高精度复核。

这种策略能够在保证极致分类性能的同时，将计算成本和延迟控制在可接受范围内。

未来可考虑利用 LoRA 对 LLM 进行领域微调，以进一步降低 API 成本并提高准确率。
