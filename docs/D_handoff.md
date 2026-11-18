# D 模块交接与自检

## 1. A 的交接验收
- [x] data_splits/test.csv: 308 行, sample_id 唯一
- [x] 33 条 no_majority
- [x] 九类标签一致
- [x] evaluation.py 可跑通

## 2. D 的评估口径（按 A 的协议）
- Task 2 主评估: 三 annotator 分别评估, 取平均和标准差
- Task 2 补充: majority 275 条
- 33 条 no_majority: 主评估保留, majority 评估排除, 错误分析单独标记

## 3. 输入输出
- 输入: B 的 task2 概率 CSV, C 的集成概率 CSV
- 输出: error_analysis.csv, case_studies.md, efficiency_results.csv, all_results.csv

## 4. 已知问题
- [ ] 等 B 的真实概率和 logits
- [ ] 等 C 的集成结果
- [ ] LLM 待小规模试跑