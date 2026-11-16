# 成员 D 交接总结文档：LLM 扩展与综合分析 (LLM & Analysis)

**负责人**：成员 D
**交接模块**：LLM 实验、错误分析 (Error Analysis)、效率分析 (Efficiency Analysis)、全组结果汇总 (Result Analysis)
**状态**：✅ 全部完成，代码与结果已提交至 GitHub 分支 `feature/d-llm-analysis`。

---

## 1. 核心交付物清单 (Where is everything?)

### 📂 1.1 分析报告 (Markdown)
*   `04_LLM_Analysis/llm_analysis.md`：LLM-only 与 LLM-assisted 的实验设计与对比分析。
*   `04_LLM_Analysis/case_studies.md`：**（报告重点）** 挑选了 5 个典型错误案例，包含 Q/A 原文、模型预测、人类分歧分析。
*   `04_LLM_Analysis/efficiency_analysis.md`：训练/推理耗时、显存占用、API延迟的成本效益分析。
*   `05_Evaluation/result_analysis.md`：**（全组终极结论）** 汇总了 A、B、C、D 四人的所有实验数据，并给出了“级联架构（Cascade）”的工程建议。

### 📊 1.2 核心数据表 (CSV)
*   `05_Evaluation/all_results.csv`：**全组终极汇总表**，包含所有单模型、集成、校准、LLM 的 Accuracy, Macro-F1, ECE, Brier Score。
*   `04_LLM_Analysis/efficiency_results.csv`：所有模型的推理耗时、吞吐量、显存占用（包含真实的 API 延迟）。
*   `04_LLM_Analysis/error_analysis.csv`：记录了所有模型在测试集上犯错的样本明细。
*   `04_LLM_Analysis/no_majority_samples.csv`：33 条人类标注者无共识的特殊样本列表。

### 🖼️ 1.3 关键图表 (Figures)
位于 `04_LLM_Analysis/figures/` 目录：
*   `cm_roberta.png` / `cm_deberta.png` / `cm_xlnet.png`：传统单模型混淆矩阵。
*   `cm_soft_voting.png`：**集成模型的崩溃现场**（严重偏置向 Explicit）。
*   `cm_llm_assisted.png`：**LLM 的救场表现**（长尾类别被成功识别，但出现了 Deflection 偏置）。

---

## 2. 核心研究结论 (直接写进 Final Report 的干货)

### 💡 发现 1：同质化集成失效
传统预训练模型（RoBERTa / DeBERTa / XLNet）均陷入了严重的“多数类偏置”，大量 `Implicit` 和 `General` 被误判为 `Explicit`。**Soft Voting 的 Macro-F1 (0.2798) 甚至低于最佳的 RoBERTa 单模型 (0.2986)**。温度缩放（TS）改善了 ECE，但无法提升准确率。

### 💡 发现 2：LLM 带来降维打击
引入 DeepSeek 大模型后，打破了语义盲区：
*   **LLM-only** (Zero-shot) 达到 **Macro-F1: 0.4234**。
*   **LLM-assisted**（输入集成概率作为先验参考）达到 **Macro-F1: 0.4766**。
*   相比最佳单模型（0.2986），**性能大幅提升了 59.6%**！尤其在 `Clarification`、`Declining to answer` 等长尾类别上实现了零的突破。

### 💡 发现 3：性能与效率的权衡 (Trade-off)
*   **本地最快**：RoBERTa（单条 3.4 ms）。
*   **LLM 延迟**：LLM-assisted 单条平均延迟约 6.48 秒（慢了近 2000 倍），单次全量测试耗时约 1996 秒。
*   **成本结论**：推荐落地为**“RoBERTa 快速初筛 + LLM-assisted 疑难复核”**的级联架构。

---

## 3. 数据口径与注意事项 (⚠️ 极其重要，写报告时务必对齐)

1. **统一评估口径**：我们所有的核心对比（表 `all_results.csv`）**均严格基于 275 条有效多数票样本（Majority 275）**。因为 A 同学的数据中，有 33 条样本三位标注者意见不一（No_Majority），为了保证公平，我们在计算指标时排除了这 33 条。
2. **LLM 未计算 ECE/Brier**：在 `all_results.csv` 中，LLM 的 ECE 和 Brier 填的是 `N/A`。这是因为 LLM 属于生成式模型，其 Token 概率（Logprobs）与判别式模型的 Softmax 特性不同，直接对比会引发学术争议。全组报告需统一口径。
3. **测试集样本数**：A 的划分中，Test 集合全量为 308 条。

---

## 4. 环境与复现说明
*   **LLM 模型**：DeepSeek-Flash (通过 API 调用，JSON 模式，`logprobs=True` 提取置信度)。
*   **复现命令**：
    ```bash
    # 激活虚拟环境
    source .venv/Scripts/activate
    
    # 运行 LLM 实验 (需配置 DEEPSEEK_API_KEY 环境变量)
    python 04_LLM_Analysis/llm_experiment.py
    
    # 合并多模型结果并生成错误分析图表
    python 04_LLM_Analysis/merge_analysis_data.py
    python 04_LLM_Analysis/error_analysis.py
    
    # 复算最终指标
    python 04_LLM_Analysis/calc_metrics.py