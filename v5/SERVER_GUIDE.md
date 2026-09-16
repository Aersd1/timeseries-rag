# V5 在其他服务器和数据集上运行

## 1. 获取分支和环境

```bash
git clone --branch v5-unified-temporal-tokens --single-branch https://github.com/Aersd1/timeseries-rag.git
cd timeseries-rag
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

先按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 安装适合服务器驱动的 **torch**，不需要 torchvision/torchaudio；然后安装其余要求：

```bash
python -m pip install -r v5/requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python v4/build_native.py
python -m unittest discover -s v5/tests -v
```

需要 Python 3.11 或更新的兼容环境、GCC、NumPy/SciPy/Pandas/Matplotlib、PyTorch 2.5+。本地测试未安装 PyTorch 时会跳过 3 个神经模块测试；**服务器测试应全部通过且不跳过**。这些测试只对合成数据做前向/反向和建库评估，没有 optimizer step，不是正式训练。CUDA 训练、AMP 和训练效果只能在服务器实测。

## 2. 配置你的 CSV

```bash
cp v5/configs/server.json v5/server.local.json
```

修改 `data.sources`。普通单序列示例：

```json
{
  "glob": "/mnt/data/power/*.csv",
  "columns": ["load", "temperature"],
  "time_column": "timestamp",
  "time_kind": "datetime",
  "expected_interval": 60,
  "group": "power"
}
```

一个文件交错多个设备：

```json
{
  "glob": "/mnt/data/gpu/**/*.csv",
  "columns": ["temperature_memory", "power_draw_W"],
  "device_column": "gpu_id",
  "time_column": "timestamp",
  "time_kind": "numeric",
  "expected_interval": 1,
  "group": "gpu"
}
```

- `columns` 必须显式列出数值列；不把时间、设备 ID 当数值信号。
- `time_kind=datetime` 的 interval 单位是秒；`numeric` 与原始数值时间单位一致。时间必须按设备严格递增，不自动排序。
- 没有时间列可删除 `time_column/time_kind/expected_interval`，此时使用原文件行顺序，用户需保证顺序和采样意义正确。
- 多设备必须指定 `device_column`。逻辑序列是文件 × 列 × 设备，不会把不同文件自动接成一条。
- 缺失/非数值/Inf 分成有限值连续段，不跨段采样。如果真实时间有缺口，配置 expected_interval 后会拒绝，应在上游重新采样或分文件；仅数值连续不能证明时间连续。
- 同一数据不可通过多个重叠 glob/列配置重复导入。
- 短序列可能没有训练/验证/测试 query；记录在教师 `skipped`，不补样本、不随机打乱时间划分。

```bash
python -m v5 doctor --config v5/server.local.json
```

## 3. 导入与生成冻结教师标签（CPU）

```bash
python -m v5 ingest --config v5/server.local.json --output runs/v5/store
python -m v5 teacher --store runs/v5/store --output runs/v5/teacher
```

默认时间区间：memory `[0,25%)`、训练 query `[25%,60%)`、validation `[60%,80%)`、test `[80%,100%)`。query 上下文和 maxH future 均须完整落在自己的段里；候选 past+future 全在 memory。因此原本 Top1 不会进入任何评估。

教师逐条逻辑序列加载 V4 索引，生成普通 Top256 的精确形状距离与 H=24/96/244 的 affine future NMSE。标签以分片 NPZ 保存位置引用和标量，不复制所有候选原始窗口；另存 hard-negative 标记。只存每条 query 选出的距离，没有 N 个全库 query 距离表。

query 采样默认彼此间隔 `L+maxH`；`queries_per_series` 是上限，无法超过数据实际可用数量。默认 train=2000/validation=200/test=200 **每条序列**，不是保证会得到这么多。若希望扩大训练集，应增加独立数据/序列，不能用重叠测试片段虚增样本量。

成功后检查：

```bash
python -c "import json; m=json.load(open('runs/v5/teacher/manifest.json')); print(m['counts']); print(m['skipped'])"
```

必须有非空 train、validation、test。索引不能加载时先确认 GCC 编译生成了 Linux 的 `v4/native.so`，不要复制 Windows DLL。

## 4. 正式训练（仅在服务器执行）

```bash
python -m v5 train --store runs/v5/store --teacher runs/v5/teacher --output runs/v5/train --device cuda
```

CPU 调试可用 `--device cpu`；多卡机器选 `--device cuda:1`。没有分布式训练。默认 batch=2，64 个教师候选/查询，dim=32、hidden=64、256 codes、4 点 primitive。没有承诺具体显存占用，先小规模运行检查。

只由 validation 直接预测 NMSE 决定 `best.pt`，test 不参与。`last.pt` 包含 optimizer、GradScaler、epoch 和 Torch RNG；`history.json` 包含全部损失、验证误差和每轮耗时。保存每个完成 epoch 的状态，不提供 batch 级恢复。

中断恢复：

```bash
python -m v5 train --store runs/v5/store --teacher runs/v5/teacher --output runs/v5/train --device cuda --resume
```

恢复要求相同 dataset/config；不接受在续训时偷偷换数据、码本大小、学习率。配置中 epochs 是预设总轮数，建议提前设置足够。若要改变配置开始新实验，传新配置和新输出目录。

OOM 优先降低 `training.batch_size`、`training.candidates`，或 `model.hidden`；改变配置须开始新训练，不复用不兼容 optimizer。现有 teacher 标签可复用。

## 5. 4/8 点 primitive 与损失消融

保持同一个 store/teacher（因此相同划分、query、候选与标签），复制配置，例如 `v5/p8.local.json`：

- 仅改 `model.primitive` 为 8，比较不同 primitive。
- 仅把 `loss_weights.future` 设 0，比较是否需要未来监督。
- 把 retrieval/future 设 0，保留 rate/reconstruction/forecast/vq，作为不蒸馏的对照。
- 所有运行保留相同 seed、epochs、数据和评估预算；不是同参数量固定 Patch 模型对照。

```bash
python -m v5 train --store runs/v5/store --teacher runs/v5/teacher --config v5/p8.local.json --output runs/v5/p8_train --device cuda
```

`--config` 可改模型、损失、训练、索引和评估设置，但 length/horizons/split 不可变。后续 index/evaluate 从 checkpoint 读取该实验的配置。更改 length/horizons/split 必须重新导入并生成标签。

## 6. 建 Token 库、评估

```bash
python -m v5 index --store runs/v5/store --checkpoint runs/v5/train/best.pt --output runs/v5/index --device cuda
python -m v5 evaluate --store runs/v5/store --teacher runs/v5/teacher --checkpoint runs/v5/train/best.pt --index runs/v5/index --output runs/v5/validation --split validation --device cuda
python -m v5 analyze --results runs/v5/validation --training runs/v5/train
```

先在 validation 比较超参数和不同实验，决定最终方案，然后一次固定测试：

```bash
python -m v5 evaluate --store runs/v5/store --teacher runs/v5/teacher --checkpoint runs/v5/train/best.pt --index runs/v5/index --output runs/v5/test --split test --device cuda
python -m v5 analyze --results runs/v5/test --training runs/v5/train
```

输出目录必须是新的，避免混入之前实验。Token index 绑定 checkpoint SHA256，换模型必须重建索引。

两个检索路由：共享 Token 的 shape cosine（磁盘分块全扫描），以及 unigram/bigram BM25 倒排。粗候选比例默认 1%，再对同一候选集比较 exact shape 与 predictive head 排序，最后做 V4 式 affine analog forecast。

Recall 是 V4 精确 Top10 起点是否包含在 Token 粗候选里。默认 stride=16 会遗漏网格之外起点；`oracle_coverage` 是因此能达到的覆盖上限，不会从 recall 分母排除这些点。若实验目标是 stride-1 精确定位，预先设 `index.stride=1`，并提高 `max_windows_per_series`，承担更高存储和扫描成本。

## 7. 回传给我什么

直接发送 **`runs/v5/test/analysis_bundle.zip`**，多组消融分别发送对应 ZIP，并说明哪个是你在 validation 上选定的最终模型。

ZIP 包含：

- `REPORT.md`、`overview.png`：直接阅读的报告与图。
- `analysis.json`、`forecast_metrics.csv`、`retrieval_metrics.csv`：按组/H/方法的指标及簇 bootstrap 区间。
- `forecasts.jsonl`：每 query、H、方法的 MSE/MAE/NMSE、返回数、候选平均 future NMSE。
- `retrieval.jsonl`：Recall、库内覆盖率、候选预算、并列标记、耗时。
- `timing.json`：编码、搜索、精排、future 阶段分项时间。
- `run_public.json`：配置和环境摘要，移除服务器数据路径。
- `training_history.json` / `training_environment.json`（若传了 `--training`）。
- `bundle_manifest.json`：每个文件 SHA256，便于检查完整性。

原始 CSV、权重、完整 query/真实 future 不在回传包里。可选 `evaluation.max_examples` >0 生成服务器本地 `examples_private.json`，默认 0，始终不会自动打包。

我拿到 ZIP 后可以继续分析：码本是否塌缩、梯度/损失是否平衡、Recall 受覆盖率还是 Token 表示限制、预测头是否学到额外未来信息、RAG 相比 raw shape/random/persistence 是否有配对收益、4/8 点与消融的压缩/效果取舍。

## 8. 资源、恢复与边界

- 导入分块读取 CSV，但完整 float32 store 会占磁盘，每个已选择数值点约 4 bytes；多列分别存储。不会复制原 CSV。
- Teacher 缓存仍使用 V4 raw float64、48 维 codes 和每点均值方差，最终常驻粗略至少 `N*(8+48+16)` bytes，构建还有额外工作数组。这里 N 是单条 memory 序列的点数，不是整库；这不是峰值保证。
- 默认单序列 memory 点数上限 2,000,000；超过立即报错。请先根据机器内存和数据设置 `teacher.max_points_per_series`。代码不自动拆成不同逻辑库、也不静默减少教师搜索范围。
- 默认 token window 上限每序列 200,000；4 点 primitive 每个窗口 61 个 11-byte records，另有两个 float32 读出向量和 SQLite postings。stride=1 的重叠窗口会让索引明显大于原始数据；V5 用于可行性验证，非最终压缩存储架构。
- 教师中断：完成 `series_x/done.json` 的序列会跳过；正在建的 V4 base 若未完成，请改用新的 teacher 输出目录重建，不用不完整索引。只中断标签阶段时可重新运行同一 teacher 命令。不要同时向同一个 teacher 输出启动多个进程。
- 导入、Token 建库、评估中断需使用新输出目录；训练可按 epoch 恢复。程序不自动删除旧目录。
- store/Token 索引内部用相对路径可迁移；teacher/oracle 中旧 V4 的绝对 base 路径需重建缓存。标签 NPZ 和 manifest 本身只含相对引用。
- 目前只做同序列历史检索；没有全局对齐不同文件的时间戳，更没有百 GB / 10B / 100ms 承诺。

## 9. 文件对应

`data.py` 导入；`teacher.py` 教师生成/标签加载；`model.py` 共享 VQ 和损失；`train.py` 训练与恢复；`token_index.py` 两种 Token 索引；`evaluate.py` 对照评估；`analyze.py` 统计和打包。核心设计见 [DESIGN.md](DESIGN.md)。
