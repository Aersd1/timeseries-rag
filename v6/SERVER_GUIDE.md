# V6 服务器使用说明

已有服务器 V6 模型/结果时，优先看 [历史＋未来联合检索及运行命令](JOINT_RETRIEVAL.md)。纯 learned 加速/融合对照见 [本轮优化报告](OPTIMIZATION.md)。无需重复导入 CSV。最新默认配置增加 joint 通道；仅保留 learned 通道是在做旧目标的对照，正式双向相似检索可只保留 joint。

已有联合/PatchTST 索引时，可直接启用 [检索执行优化](SEARCH_OPTIMIZATION.md)：更新代码后，在仓库根目录运行 `python -m v6.build_native`，重启检索进程即可。无需重训或重建索引；没有编译器时自动使用 NumPy 备用实现。

## 1. 获取代码与环境

从仓库根目录执行所有命令。已有 V4/V5 分支不需要修改。

```bash
git clone --filter=blob:none --sparse --branch v6-bayesian-future-retrieval https://github.com/Aersd1/timeseries-rag.git
cd timeseries-rag
git sparse-checkout set v5 v6
conda create -n tsrag-v6 python=3.11 -y
conda activate tsrag-v6
# 先安装适合服务器驱动的 PyTorch CUDA 版本，再安装其余依赖。
pip install -r v6/requirements.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s v6/tests -v
```

已有 `tslib` 也可使用。测试仅使用临时合成数据和随机权重，执行前后向，不做优化器参数更新。CUDA 不可用时自动跳过 CUDA 项。

该分支继承了 V5 用户上传的结果；稀疏检出可避免下载 `runs/v5_nrel` 等大型旧实验。V6 只依赖 V5 的 CSV 解析和通用函数，不要求 V4 native 编译，也不读取 V5 模型。

## 2. 配置新数据

```bash
cp v6/configs/server.json v6/configs/local.json
```

编辑 `data.sources`，例如：

```json
{
  "glob": "/data/wind/*.csv",
  "columns": ["power"],
  "time_column": "timestamp",
  "time_kind": "datetime",
  "expected_interval": 300,
  "group": "wind"
}
```

- 一列作为一条单变量序列；多设备混排时增加 `device_column`。
- 数据必须按设备时间升序，无重复。时间间隔以秒计；数字时间戳可设 `time_kind=numeric`，间隔单位须与其一致。
- 建议明确填写 `expected_interval`，遇到缺失时间先补 NaN 或拆文件。没有该字段时仅检查递增，244 点不一定代表固定时长。
- NaN/Inf 打断连续区间，历史和未来都不跨越断点。不会把 NaN 填成零。
- 默认同一逻辑序列检索；`scope=group/all` 扩大范围，但本版未保存/核对跨文件日历截止时刻。严格因果回测应保持 `series`，或事先按共同时间边界处理数据。
- 每条序列要足够长，四个区间中均能容纳历史+最大未来。

默认划分：前 25% memory（检索库和概率模型训练），25–60% 编码器训练，60–80% validation，80–100% test。样本完整落在单一区间，不跨边界。

```bash
python -m v6 doctor --config v6/configs/local.json
python -m v6 estimate --config v6/configs/local.json --points 100000000
python -m v6 ingest --config v6/configs/local.json --output v6/runs/experiment/store
```

`estimate` 假定传入点数全部可入库，是容量上界近似；实际 memory 比例、无效值和边界会减少窗口。

### 复用服务器现有的 V5 store

不必重复导入 CSV。将后面命令中的 store 路径换成 `runs/v5_nrel/store` 或服务器实际目录。V6 配置的 `length`、`horizons`、`split` 必须与该 store 一致。

复用 store 时无需执行 `doctor`/`ingest`，也无需原 CSV 仍在旧路径；必须保留完整 catalog、config 和 values 文件。V5 checkpoint 不兼容，V6 两个阶段都要重新训练。

## 3. 训练未来概率网络

```bash
python -m v6 train --stage prior \
  --config v6/configs/local.json --store v6/runs/experiment/store \
  --output v6/runs/experiment/prior --device cuda
```

该阶段仅用 memory 样本拟合，validation 选择最低分布负对数似然的 `best.pt`。

## 4. 冻结概率网络，训练编码器

```bash
python -m v6 train --stage encoder \
  --config v6/configs/local.json --store v6/runs/experiment/store \
  --prior v6/runs/experiment/prior/best.pt \
  --output v6/runs/experiment/encoder --device cuda
```

编码器只在后续 train 区间更新。验证选择使用已固定的对比+辅助损失，不看 test。

中断后以相同命令加 `--resume`，配置和 prior 文件身份必须一致。恢复到最后一个完整 epoch；`last.pt` 包含优化器、混合精度缩放器和 RNG。不要把 `best.pt` 当作续训断点。

## 5. 建多层检索库

```bash
python -m v6 index --store v6/runs/experiment/store \
  --checkpoint v6/runs/experiment/encoder/best.pt \
  --output v6/runs/experiment/index --device cuda
```

- 默认每个合法起点一个向量，含 learned/history/belief 三种通道用于实验。
- 大规模部署可在训练前配置 `index.channels=["learned"]` 节省对照向量空间。
- `stride>1` 会漏掉未入库起点；不要将这种库的完整搜索称作原始全起点召回。
- 每个分片默认 10 万窗口，批量编码，不把全部库放入 GPU。
- 建库中断会保留 `complete=false`，拒绝查询。当前建库不支持断点续建，请选一个新的输出目录重新建；不要把未完成的 manifest 手工标为完成。

## 6. 先验证速度/精度，再固定设置测试

```bash
python -m v6 evaluate --store v6/runs/experiment/store \
  --checkpoint v6/runs/experiment/encoder/best.pt --index v6/runs/experiment/index \
  --output v6/runs/experiment/validation --split validation --device cuda \
  --leaf-budgets 0 32 128
python -m v6 analyze --results v6/runs/experiment/validation \
  --prior-training v6/runs/experiment/prior --encoder-training v6/runs/experiment/encoder
```

0 表示无限制完整搜索；32/128 是叶块访问预算。一次实验会报告所有通道的完整/预算查询耗时，包括各自的编码、搜索、候选读取与聚合；不计进程/模型加载。查询读入数组前的文件解析也不在计时中。评估会先计算概率基线，模型已有一次预热；库访问顺序也影响缓存，因此不是严格冷启动基准。

初次完整搜索可能慢，应先用少量验证查询确认规模可承受。`audit_queries` 默认 8，会额外做全库原始距离/真实未来 oracle 扫描，仅作离线对照，可能耗时较长；正式速度测试可在训练前将其设为 0。

查看 `analysis.json`：

- `timing`：p50/p95/p99/max、100ms 内比例、访问窗口比例、返回完整 k 的比例。
- `embedding_recall_against_full`：预算查询相对完整编码搜索的 ID Recall。
- `paired_comparisons`：相同查询上的 learned vs history、prior、persistence 等收益和置信区间。
- `calibration`：CRPS、边际 NLL、90% 覆盖率、区间宽度。
- `embedding_diagnostics`：表示方差和有效秩，帮助检查塌缩。
- `future_oracle_id_recall`：少量查询对“真实未来最近邻”的离线召回；并列与多解会影响精确起点指标。

根据 validation 固定预算。例如选择 128：

```bash
python -m v6 evaluate --store v6/runs/experiment/store \
  --checkpoint v6/runs/experiment/encoder/best.pt --index v6/runs/experiment/index \
  --output v6/runs/experiment/test --split test --device cuda --leaf-budgets 128
python -m v6 analyze --results v6/runs/experiment/test \
  --prior-training v6/runs/experiment/prior --encoder-training v6/runs/experiment/encoder
```

如果还要在 test 描述完整搜索与预算的差别，可固定报告 `--leaf-budgets 0 128`，但不要根据 test 再选择预算。

可用 `--oversample` 改变候选冗余，`--time-budget-ms` 添加软搜索时间上限，不必重训。修改只能依据验证集选择。小冗余可能返回不足 k；时间预算不含编码/候选读取，也不能中断正在发生的 I/O，不能保证总查询时间硬性不超过 100ms。

## 7. 证明概率引导是否有用：无 belief 消融

复制配置为 `local-no-belief.json`，仅把 `model.use_belief` 改为 `false`。保持 seed、数据、训练预算、窗口采样和损失其他部分相同。

```bash
python -m v6 train --stage encoder --config v6/configs/local-no-belief.json \
  --store v6/runs/experiment/store --prior v6/runs/experiment/prior/best.pt \
  --output v6/runs/no-belief/encoder --device cuda
python -m v6 index --store v6/runs/experiment/store \
  --checkpoint v6/runs/no-belief/encoder/best.pt --output v6/runs/no-belief/index --device cuda
python -m v6 evaluate --store v6/runs/experiment/store \
  --checkpoint v6/runs/no-belief/encoder/best.pt --index v6/runs/no-belief/index \
  --output v6/runs/no-belief/test --split test --device cuda --leaf-budgets 128
python -m v6 analyze --results v6/runs/no-belief/test
python -m v6 compare --first v6/runs/experiment/test --second v6/runs/no-belief/test \
  --output v6/runs/experiment/no_belief_comparison.json
```

比较中的正 improvement 表示第一个模型误差更低。结果使用相同查询配对，不用两个整体平均值替代配对分析。建议至少运行三个不同 seed，各自用相同 seed 的成对模型比较。

## 8. 单次真正查询

将纯历史数组保存为 `history.npy`，长度等于 `length`，不含未来。sid 是 catalog 中目标逻辑序列的编号，用于归一化尺度及默认搜索范围。

```bash
python -m v6 query --store v6/runs/experiment/store \
  --checkpoint v6/runs/experiment/encoder/best.pt --index v6/runs/experiment/index \
  --history /data/history.npy --sid 0 --leaf-budget 128 \
  --output v6/runs/query.json --device cuda
```

若查询本来就在同一库中，提供 `--start 原始起点`，这样会排除其上下文+未来重叠邻域。返回文件包含候选 sid/start/距离、诊断和预测。服务部署应复用 `Retriever` 对象，避免每次启动 Python 或加载模型。

## 9. 回传哪些结果

通常发送：

1. `test/analysis_bundle.zip`。
2. 相应的 `validation/analysis_bundle.zip`。
3. 无 belief 的结果包和 `no_belief_comparison.json`。

包中包含配对逐查询指标、候选位置、统计图和脱敏配置，不包含原始序列数组、CSV 路径、权重。候选 sid/start 仍是数据位置元信息。

本机另有 `example_private_*.png`（预测图）、`retrieved_private_*.png`（查询与候选历史/未来叠图）和 `examples_private.json`。它们包含实际曲线，没有自动放入回传包；如愿意分享具体曲线，可另附。

## 常见问题

- **CUDA OOM**：减小 training.batch_size / index.batch_size；批内对比需要至少 3 个样本，过小会削弱训练。
- **usable_fraction 很低**：检查同批样本是否来自过近的位置；增加样本间隔或更充分混合序列。
- **区间极宽或覆盖率低**：先检查概率模型训练/数据漂移，不能仅靠增大检索预算修复。
- **候选不足**：检查库中可用完整窗口数、排除条件和 oversample；预算截断不保证找到 k 个独立片段。
- **编码最近邻正确但预测更差**：说明表示/概率先验没有学好预测关系，不能用编码召回率掩盖预测失败。
- **模型和索引身份不匹配**：更换 checkpoint 必须重新建库，不能混用另一个 epoch 的向量。
- **磁盘不足**：先估算、只保留 learned 通道、缩小试验数据；stride 可节省空间但改变检索覆盖，必须标注。

V6 运行目录与本地配置已忽略，不建议把大规模数据/模型/索引加入 Git。
