# LLM 实验分析 (LLM-only vs LLM-assisted)

## 1. 实验设置
- **模型**: DeepSeek-Flash (通过 API 调用)
- **Prompt**: 采用 Zero-shot 设计与 JSON 强制输出，提供九类标签定义。
- **测试集**: 包含 308 条全量样本，主评估指标基于 275 条多数票样本（与 B/C 对齐）。

## 2. 核心结果对比 (275 多数票口径)
| 模型配置 | Accuracy | Macro-F1 | 说明 |
|---|---|---|---|
| RoBERTa (Best PLM) | 0.3491 | 0.2986 | 传统微调模型的最好表现 |
| Soft Voting (Ensemble)| 0.3164 | 0.2798 | 集成不仅未提升，反而下降 |
| **LLM-only (Zero-shot)**| **0.3745** | **0.4234** | 纯大模型零样本，打破 PLM 瓶颈 |
| **LLM-assisted** | **0.4291** | **0.4766** | 加入集成概率辅助后达到全场最佳 |

## 3. 结果分析
实验结果揭示了一个极其重要的现象：传统预训练语言模型（PLM）的集成策略（Hard/Soft Voting）不仅未能提升性能，反而低于最佳单模型 RoBERTa（Macro-F1: 0.2986）。通过混淆矩阵分析可知，RoBERTa、DeBERTa 与 XLNet 在细粒度语义分类上存在高度同质化的“知识盲区”——它们均严重陷入 Explicit 多数类偏置，对 Implicit 和 General 的召回率极低。因此，同质化模型的概率平均无法带来信息增益，反而放大了系统性偏差。

相比之下，引入 DeepSeek 大语言模型打破了这一僵局。纯 Zero-shot 的 LLM-only 取得了 0.4234 的 Macro-F1；而当我们将集成模型的概率和置信度作为先验信息输入给 LLM（LLM-assisted）时，Macro-F1 实现了惊人的 0.4766（相比基线提升 59.6%）。LLM-assisted 成功利用了其强大的上下文推理能力，在长尾类别（如 Clarification 和 Declining to answer）上实现了零的突破，并大幅纠正了 Explicit 偏置。这证明了在复杂、主观的文本分类任务中，异构大模型的推理增强比单纯的模型集成更具价值。

## 4. 成本记录
- **API 延迟**: 平均单条请求耗时约 2.79 秒。
- **API 成本**: 308 条样本测试集总成本约为 0.1 元人民币（DeepSeek-Flash 极具性价比）。

## 5. 结论
LLM 在零样本和少样本的细粒度语义分类上具有传统 PLM 无法比拟的优势。将传统模型的集成概率作为先验信息输入（LLM-assisted），能有效引导 LLM 纠正其发散性，达到最佳性能。