# Task 2 三模型 Baseline 比较：材料审计与执行说明

## 1. 当前材料是否完整

审计日期：2026-11-10。审计范围为当前项目目录，以及此前明确提供的 Downloads/Desktop 汇总文件。

| 类别 | 当前状态 | 说明 |
|---|---|---|
| `requirements.txt` | 已完成 | 三个模型训练所需的完整环境。 |
| `environment_setup.md` | 已完成 | 包含本地与 Kaggle 环境说明。 |
| `train_roberta.py` | 已完成 | 能输出 Validation/Test 预测、概率及 Validation logits。 |
| `train_deberta.py` | 已补齐 | 作为 `train_deberta_v3.py` 的兼容入口，满足分工文档的准确文件名。 |
| `train_deberta_v3.py` | 已完成 | DeBERTa-v3 的完整实现。 |
| `train_xlnet.py` | 已完成 | XLNet 的完整实现。 |
| 三份 experiment handover | 已完成 | RoBERTa、DeBERTa-v3、XLNet 均有。 |
| 三模型 Notebook | 部分完成 | 当前目录有 DeBERTa-v3、XLNet；RoBERTa Notebook 未在当前目录中，但脚本与交接文档存在。 |
| 三模型测试预测/概率 | 当前目录缺失 | 必须从三个 Kaggle Notebook Output 下载。 |
| 三模型 Validation 概率/logits | 当前目录缺失 | 必须交给成员 C；只能在 Validation Set 拟合 Temperature。 |
| 三模型逐类 F1 | 当前目录缺失 | 比较图和“易混淆类别”分析需要。 |
| 三模型训练日志/config | 部分完成 | 本地只发现 DeBERTa-v3 与 XLNet 的 summary/config 零散文件；RoBERTa 以及完整日志仍需下载。 |
| 推理时间/峰值显存 | 尚未统一测量 | 必须在同一硬件、相同 batch size 和 max length 下复测。 |
| `baseline_results.csv` | 尚未生成 | 等输入验收通过后由 `compare_baselines.py` 生成。 |
| `baseline_comparison.png` | 尚未生成 | 同上。 |
| `baseline_analysis.md` | 尚未生成 | 同上。 |

结论：**下一阶段确实是三个 baseline 的统一比较，但正式比较前必须先把三个 Kaggle Output 的完整小型交付文件下载并归档。** 现在若直接比较，只能得到不完整结论。

## 2. 推荐目录结构

比较程序会递归寻找文件，因此 Kaggle ZIP 解压后的 `predictions/`、`results/`、`logs/` 子目录可以原样保留。

```text
baseline_inputs/
├── roberta/
│   ├── predictions/
│   ├── results/
│   └── logs/
├── deberta_v3/
│   ├── predictions/
│   ├── results/
│   └── logs/
└── xlnet/
    ├── predictions/
    ├── results/
    └── logs/
```

每个模型目录必须包含以下 11 个非空文件。`PREFIX` 分别是 `roberta`、`deberta_v3`、`xlnet`：

```text
PREFIX_predictions.csv
PREFIX_probabilities.csv
PREFIX_validation_predictions.csv
PREFIX_validation_probabilities.csv
PREFIX_validation_logits.csv
PREFIX_task2_summary.csv
PREFIX_task2_per_class.csv
PREFIX_task2_majority_summary.csv
PREFIX_task2_majority_per_class.csv
PREFIX_training_log.csv
PREFIX_experiment_config.json
```

模型权重不参与指标汇总，但统一测量推理速度时需要每个模型的 `best_model/` 目录。

## 3. 环境要求

### 3.1 只做材料验收和比较图（CPU 足够）

- Python 3.10–3.13
- 内存建议 4 GB 以上
- 不需要 GPU
- 依赖见 `baseline_comparison_requirements.txt`

Windows PowerShell：

```powershell
python -m venv .venv-baseline
.\.venv-baseline\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r baseline_comparison_requirements.txt
```

Kaggle 中通常已经包含 pandas、NumPy 和 Matplotlib，无需为了比较再次覆盖 Kaggle 的 PyArrow 或 CUDA 依赖。

### 3.2 统一推理基准（推荐 Kaggle T4 GPU）

- 三个模型必须在**同一个 Kaggle Session、同一块 GPU**上依次测试。
- Accelerator 选 GPU；运行检查必须显示 `torch.cuda.is_available() == True`。
- 三个模型统一 `batch_size=8`、`max_length=256`、`repeats=3`。
- 使用训练环境的 `requirements.txt`，或直接使用能够加载三个已保存模型的原 Kaggle 环境。
- 不要为推理基准重新训练模型；将三个 Notebook 的 `best_model` Output 作为 Kaggle Input 加入比较 Notebook。

## 4. 在 Kaggle 上统一测推理速度

新建一个只用于比较的 Kaggle Notebook，开启 GPU，将 `benchmark_baseline_inference.py` 上传为 Notebook 文件或 Dataset 文件。再把三个已保存的 `best_model` Output 加为 Input。

先寻找实际模型目录：

```python
from pathlib import Path

for config in Path("/kaggle/input").rglob("config.json"):
    parent = config.parent
    if (parent / "model.safetensors").exists() or (parent / "pytorch_model.bin").exists():
        print(parent)
```

分别运行以下命令，把占位路径替换为上一步打印出的目录。测试集必须是 A 组固定的同一个 `test.csv`：

```bash
python /kaggle/input/baseline-tools/benchmark_baseline_inference.py \
  --model-key roberta \
  --model-dir /kaggle/input/REPLACE_ROBERTA/best_model \
  --test-file /kaggle/working/Response_Clarity_Project/01_Data/data_splits/test.csv \
  --output-csv /kaggle/working/inference_benchmark.csv \
  --batch-size 8 --max-length 256 --repeats 3

python /kaggle/input/baseline-tools/benchmark_baseline_inference.py \
  --model-key deberta_v3 \
  --model-dir /kaggle/input/REPLACE_DEBERTA/best_model \
  --test-file /kaggle/working/Response_Clarity_Project/01_Data/data_splits/test.csv \
  --output-csv /kaggle/working/inference_benchmark.csv \
  --batch-size 8 --max-length 256 --repeats 3

python /kaggle/input/baseline-tools/benchmark_baseline_inference.py \
  --model-key xlnet \
  --model-dir /kaggle/input/REPLACE_XLNET/best_model \
  --test-file /kaggle/working/Response_Clarity_Project/01_Data/data_splits/test.csv \
  --output-csv /kaggle/working/inference_benchmark.csv \
  --batch-size 8 --max-length 256 --repeats 3
```

预期结果：`/kaggle/working/inference_benchmark.csv` 有 3 行，记录平均推理秒数、吞吐量、单样本延迟和峰值 GPU 显存。脚本按 `model_key` 覆盖旧行，因此单个模型失败后可安全重跑。

## 5. 生成最终比较交付物

确保已经取得 A 组固定测试集，例如：

```text
Response_Clarity_Project/01_Data/data_splits/test.csv
```

在项目根目录运行：

```powershell
python compare_baselines.py `
  --inputs-root baseline_inputs `
  --test-file Response_Clarity_Project\01_Data\data_splits\test.csv `
  --benchmark-file inference_benchmark.csv `
  --output-dir baseline_comparison_outputs
```

Linux/Kaggle：

```bash
python compare_baselines.py \
  --inputs-root baseline_inputs \
  --test-file Response_Clarity_Project/01_Data/data_splits/test.csv \
  --benchmark-file inference_benchmark.csv \
  --output-dir baseline_comparison_outputs
```

### 严格模式的预期结果

程序首先生成 `baseline_input_audit.csv`。若任何模型缺文件、文件为空、行数错误、样本不对齐、标签顺序错误、概率不合法，程序会停止，且不会伪造最终比较。

验收通过后生成：

```text
baseline_comparison_outputs/
├── baseline_input_audit.csv
├── baseline_results.csv
├── baseline_per_class_results.csv
├── baseline_comparison.png
├── baseline_typical_errors.csv
└── baseline_analysis.md
```

`--allow-incomplete` 仅用于调试或预览，生成的分析会明确标记为“不完整草稿”，不得作为小组最终结果。

## 6. 指标解释与最终检查

- 主报告：`annotator_mean`，308 个测试样本。
- 补充报告：`supplementary_majority_vote`，仅 275 个具有多数标签的样本。
- 主要模型选择指标：Validation Macro-F1。
- Test Set 只做一次最终评价，不参与选超参数，也不拟合 Temperature。
- 九分类固定顺序：Explicit、Implicit、Dodging、General、Deflection、Declining to answer、Claims ignorance、Clarification、Partial/half-answer。
- 三个 config 的 data commit 应一致；若不一致，应重新确认数据而不是直接比较。
- 训练公平性比较使用有效训练批量：`per_device_train_batch_size × gradient_accumulation_steps × GPU 数量`。显存内单步 Batch Size 可以因模型资源需求不同而调整，但原始值必须如实保留。
- 图表中 Accuracy 与 Macro-F1 从 0 开始显示，避免视觉上夸大差异。

## 7. 最终交接清单

交给成员 C：

- 三模型 Test predictions/probabilities。
- 三模型 Validation predictions/probabilities/logits。
- 固定标签与概率列顺序。
- 三模型 config 与训练日志。
- 明确说明 Temperature 只在 Validation Set 拟合。

交给成员 D：

- `baseline_results.csv`
- `baseline_comparison.png`
- `baseline_analysis.md`
- `baseline_typical_errors.csv`
- `baseline_input_audit.csv`
- 三模型训练/推理时间和资源记录。
