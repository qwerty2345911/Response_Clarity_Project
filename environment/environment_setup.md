# Baseline 训练环境配置

本文档对应成员 B 的训练环境交付物，适用于 RoBERTa-base、DeBERTa-v3-base 和 XLNet-base 的九分类微调。环境检查日期为 **2026-11-07（Asia/Shanghai）**。

## 1. 已检测到的本机环境

| 项目 | 检测结果 |
|---|---|
| 操作系统 | Windows 10.0.26200，64 位 |
| CPU | AMD Ryzen 9 5900HX with Radeon Graphics |
| CPU 线程 | 16 个逻辑处理器 |
| 内存 | 15.41 GB；检查时约 4.12 GB 可用 |
| NVIDIA GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU |
| 显存 | 4096 MiB |
| NVIDIA 驱动 | 566.07 |
| 驱动报告的 CUDA 上限 | CUDA 12.7 |
| 系统盘剩余空间 | 约 50 GB |
| 已安装 Python | 3.14.2、3.10.6 |

检查时，全局 Python 3.14 环境中的 PyTorch 为 `2.9.0+cpu`，因此：

```text
torch.cuda.is_available() = False
torch.cuda.device_count() = 0
```

这不是显卡故障，而是安装了 CPU-only PyTorch。项目已改用 Python 3.10 的独立 `.venv`，并在 `requirements.txt` 中指定 CUDA 12.6 版 PyTorch。CUDA 运行库由 PyTorch wheel 自带，无需单独安装 CUDA Toolkit；需要保留正常的 NVIDIA 驱动。

## 2. 创建和安装环境

在项目根目录的 PowerShell 中运行：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

如果 PowerShell 阻止激活脚本，可只对当前窗口临时放开：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

也可以完全不激活环境，直接使用：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

不要在本项目中直接使用系统的 `python` 或全局 `pip`，否则可能再次调用 Python 3.14 的 CPU-only PyTorch。

## 3. 安装后验证

### 3.1 验证核心依赖

```powershell
python -c "import torch, transformers, datasets, accelerate, sklearn, pandas; print('torch:', torch.__version__); print('transformers:', transformers.__version__); print('datasets:', datasets.__version__); print('accelerate:', accelerate.__version__); print('sklearn:', sklearn.__version__); print('pandas:', pandas.__version__)"
```

### 3.2 验证 CUDA

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA build:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'); print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2) if torch.cuda.is_available() else 'N/A')"
```

本项目环境已在 2026-11-07 完成实际验证，结果如下：

```text
Python: 3.10.6
PyTorch: 2.14.1+cu126
PyTorch CUDA build: 12.6
CUDA available: True
CUDA devices: 1
GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU
VRAM GiB: 4.0
Transformers: 5.20.2
Datasets: 5.1.0
Accelerate: 1.15.0
Evaluate: 0.4.6
scikit-learn: 1.7.2
pandas: 2.3.3
NumPy: 2.2.6
pip check: No broken requirements found
```

再运行一个实际 GPU 张量测试：

```powershell
python -c "import torch; x=torch.randn(1024,1024,device='cuda'); y=x@x; print(y.shape, y.device, round(torch.cuda.max_memory_allocated()/1024**2,1), 'MiB')"
```

该测试已通过，输出张量形状为 `(1024, 1024)`、设备为 `cuda:0`，峰值显存分配约 16.1 MiB。这证明 PyTorch 不仅能枚举 GPU，也能实际执行 CUDA 运算。

### 3.3 记录最终环境

依赖安装和验证成功后保存完整版本快照：

```powershell
python -m pip freeze > environment_freeze.txt
nvidia-smi > nvidia_smi.txt
```

`requirements.txt` 是项目所需的直接依赖；`environment_freeze.txt` 用于记录某次实验实际解析出的全部依赖，二者用途不同。

## 4. CPU 可行性评估

本机 CPU 和 15.41 GB 内存足以完成以下工作：

- 数据清洗、分词和 Dataset 构建；
- 小样本 smoke test；
- 模型推理及指标计算；
- 在极小数据子集上验证训练代码。

不建议仅使用 CPU 完整微调三个 base 模型。RoBERTa-base、DeBERTa-v3-base 和 XLNet-base 都约为亿级参数，CPU 全量训练预计非常慢，且会占用大量内存。CPU 应作为功能验证和 GPU 不可用时的回退方案，不应作为主要实验设备。

CPU smoke test 建议：

```text
train samples: 16-64
validation samples: 8-16
max_length: 64-128
batch size: 1-2
epochs: 1
```

## 5. 本机 RTX 3050 Ti 4GB 可行性评估

结论：**可以进行小批量微调和完整流程验证，但 4GB 显存非常紧张，不适合作为三个模型的唯一训练资源。**

建议起始配置：

```text
max_length: 256
per_device_train_batch_size: 1
per_device_eval_batch_size: 2
gradient_accumulation_steps: 16
fp16: true
gradient_checkpointing: true
eval_accumulation_steps: 8
dataloader_num_workers: 0  # Windows 下先以稳定为主
```

模型判断：

- **RoBERTa-base：**在上述节省显存配置下有较高概率可以训练；先跑小样本，再跑完整数据。
- **DeBERTa-v3-base：**可能可以运行，但比 RoBERTa 更容易显存不足；必要时把 `max_length` 降至 128，或只在免费云 GPU 上训练。
- **XLNet-base：**4GB 显存下风险最高，可能频繁 OOM 或训练过慢；优先放到免费云 GPU，并把本地实验限定为 smoke test。

发生 CUDA out of memory 时，按以下顺序调整：

1. 保持 batch size 为 1；
2. 将 `max_length` 从 256 降至 192 或 128；
3. 确认启用 FP16 和 gradient checkpointing；
4. 减小评估 batch size；
5. 重启 Python 进程释放碎片化显存；
6. 转到 Kaggle 或 Colab 免费 GPU。

不要为了让不同模型“跑起来”而悄悄改变实验条件。若某个模型必须采用更短序列或其他设置，应在结果表中记录真实配置，并在分析中说明可比性限制。

## 6. 免费 GPU 方案

### Kaggle Notebooks（优先推荐）

Kaggle 官方文档说明免费 Notebook 可使用 GPU，常见为 P100，也可能提供 T4；GPU 配额按周计算且会随资源供给变化。相较本机 4GB RTX 3050 Ti，这类 GPU 通常具有更多显存，更适合三个 base 模型的正式训练。

基本操作：

1. 新建 Kaggle Notebook；
2. 在 Settings 中将 Accelerator 设置为 GPU；
3. 上传项目代码和数据，或通过私有 Kaggle Dataset 挂载；
4. 在首个单元格安装 `requirements.txt`；
5. 运行 CUDA 检查并将输出保存到实验记录；
6. 将 checkpoint、预测 CSV 和日志写入 `/kaggle/working`，在会话结束前保存 Notebook Version 或下载结果。

### Google Colab Free

Colab 免费层可以提供 GPU，但官方明确说明资源类型、可用性和使用上限并不保证，限制会动态变化。它适合临时补充实验，但不能假设每次都能获得相同 GPU。每次运行都应记录 `nvidia-smi`、PyTorch 版本和随机种子。

### 推荐的设备分工

| 任务 | 推荐设备 |
|---|---|
| 数据处理、单元测试、指标计算 | 本机 CPU |
| 训练代码 smoke test | 本机 RTX 3050 Ti |
| RoBERTa 正式训练 | Kaggle/Colab GPU；本机可作为备选 |
| DeBERTa 正式训练 | Kaggle/Colab GPU |
| XLNet 正式训练 | 优先 Kaggle P100/T4；若失败记录资源限制 |

## 7. 实验环境记录要求

每次正式训练至少记录：

- Python、PyTorch、Transformers、Datasets 版本；
- 模型仓库名称及 revision/commit；
- GPU 型号、显存和 NVIDIA 驱动；
- 数据划分版本、标签顺序和随机种子；
- batch size、梯度累积、学习率、epoch、max length；
- 是否启用 FP16、gradient checkpointing；
- 训练时间、推理时间和最大显存占用；
- 是否发生 OOM、重启或中断。

建议固定基础随机种子为 `42`，但最终必须与成员 A 和全组确认后再作为正式实验设置。

## 8. 参考资料

- PyTorch Windows/CUDA 安装与验证：<https://pytorch.org/get-started/locally/>
- Transformers 安装说明：<https://huggingface.co/docs/transformers/installation>
- Datasets 安装说明：<https://huggingface.co/docs/datasets/installation>
- Kaggle GPU 使用说明：<https://www.kaggle.com/docs/efficient-gpu-usage>
- Google Colab 免费资源说明：<https://research.google.com/colaboratory/faq.html>
