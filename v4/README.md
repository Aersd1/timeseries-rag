# V4：检索相似过去，验证未来是否可预测

独立实验版本，位于仓库 `v4/` 目录，分支为 `v4-analog-forecasting`。保留 V2/V3，不训练神经预测器。本版实现参考网页一至六：历史案例 `(past, future)`、Analog Forecasting、三种映射、加权 KNN、过去距离与未来误差统计。

已完成的实验：[结论](docs/study/CONCLUSIONS.md)、[详细报告](docs/study/REPORT.md)、[统计结果](docs/study/summary.json)。这些是本机验证子集的历史结果，不代表新设备的性能。原始 CSV、大型索引、逐候选明细和编译产物不随仓库提供；新数据请先按根目录 [迁移说明](../MIGRATION_GUIDE.md) 建库。

参考：[用户提供的实验路线](https://chatgpt.com/s/t_6aaa38bfc3c4819185adf27f03fdf5d9)。网页中的百 GB / 100 ms 描述不是本项目已证实的结论。

## 数据与防泄漏

- 从既有原始数据索引读取数据，合并同文件/列/设备的连续分块，校验重叠值一致；有缺口则报错，不跨缺口拼接。
- 按逻辑序列长度分割：前 60% 是案例库；后 40% 是查询测试段。模型的 Fourier 频率与量化边界只在训练前缀拟合。
- 候选历史长度 `L=244`，未来长度 `H=24/96/244`，必须完整位于训练前缀。未来只用原始数组偏移读取，不额外保存所有窗口的 future 副本。
- 同序列要求 `candidate_start + L + H <= query_start`。这是半开区间约定：候选最后一个样本严格早于查询第一个样本。**原本的 Top1 自身和相邻重叠窗口从候选资格阶段排除**，不是事后删第一名。不同位置碰巧形状距离为零仍是合法历史。
- `same_series`：只检索相同文件、列、设备，保证严格过去。
- `pooled_train`：检索所有序列训练前缀，是离线外部案例库实验。不同文件没有统一时钟，不能宣称已经模拟全局在线时间因果性；实际部署要先按真实时间构建可用记忆库。
- 普通 Top100 的候选彼此可重叠；`episodes` 另加候选之间起点至少 `L+H` 的间隔。两个分支均排除了自身。查询采样也在各 H 内按 `L+H` 分隔。

## 结构

```text
现有原始索引 -> 合并逻辑序列 -> 训练前缀 -> 训练量化模型 + 层级符号索引
查询过去 -> 合法长块 -> 长块下界排序 -> 符号过滤 -> 精确距离
         -> 有界最近候选池 -> 普通 Top100 / episode 去重
         -> 读取候选 future -> 三种映射 -> 距离加权预测
验证程序另外读取 query future -> future MSE、相关性、随机对照
```

`forecast.py` 的检索、映射、聚合接口均不接收 query future。只有 `experiment.py` 的离线评分用它。

### 三种映射

1. `mean_std`: `mean(Q) + std(Q)/std(X) * (Y-mean(X))`
2. `endpoint`: `Q[-1] + std(Q)/std(X) * (Y-X[-1])`
3. `affine`: 仅在过去 244 点拟合 `Q≈aX+b`，预测 `aY+b`

常数/近常数历史（标准差 ≤1e-10）无法识别尺度，退回 query 最后一值，避免除零和巨大尺度倍率。

权重为 `softmax(-d²/tau)`，`tau=max(median(d²),1e-8)`；只用过去距离，不用真实未来调温度。实现加权均值和逐坐标加权中位数。全部报告 `K=1,4,16,32,100`；候选不足不补齐，也不把少于 K 的结果冒称 K。

## 使用

在本目录运行（Python、NumPy、SciPy、Matplotlib；C 编译器）：

```powershell
git switch v4-analog-forecasting
cd v4
python -m pip install -r requirements.txt
python build_native.py
python -m unittest -v test_forecast.py
python prepare.py --source-index ../results/validation_index --output results/memory
python experiment.py --memory results/memory --output results/study --horizons 24 96 244 --per-group 20
python analyze.py --results results/study
```

以上命令从仓库根目录开始；`--source-index` 必须替换为已经建立的实际原始索引目录，示例目录不会随仓库上传。输出目录必须是新目录，避免覆盖实验。Windows 编译器未在 PATH 时先设置 `$env:CC='编译器绝对路径'`。

其他设备先用 V2 数据导入流程将新 CSV 建成原始索引，再把 `--source-index` 换成新索引路径。当前准备程序要求相同逻辑序列的分块连续；包含缺失段的数据应预先按连续段分成不同逻辑序列。

本次生成的 cascade manifest 和 `experiment.json` 包含绝对路径，迁移后建议重新运行 prepare，而不是直接照搬旧机器绝对路径。原 CSV 无需与本机器目录相同，但 query `source/column/device` 必须与新索引一致。

### 单次预测

query JSON 包含 `values`（244 个有限数）、`source`、`column`、`device`、`start`（同设备/列的逻辑采样位置，不是混合设备 CSV 行号）。不需要 query future。

```powershell
python predict.py --memory results/memory --query-json query.json --horizon 96 --k 16 --scope same_series --selection episodes --transfer endpoint --aggregation mean --output results/prediction.json
```

返回预测、候选位置和 future 偏移、有效邻居数、检索及端到端预测时间。不含冷启动时间，首次加载应单独测量。案例库小或重复严重时，`returned_k` 可能小于请求值；检查 `episode_complete`，若为 false 表示有界池不足，不能宣称已找齐。

## 统计输出与解释

- `results/study/REPORT.md`：汇总报告。
- `results/study/CONCLUSIONS.md`：可读结论与基线对比。
- `summary.json`：分数据组、H、映射、候选类型的统计与时间。
- `pairs.jsonl`：每个查询的候选位置、过去距离、原始 future MSE、归一化 future MSE。
- `query_statistics.json`：每个 query 的 Spearman、分箱误差、随机对照。
- `forecasts.jsonl`：所有 K/聚合方式的预测误差、persistence 对照和有效邻居数。
- `progress.json`：每次检索时间、排除校验及不足标记。
- 图：`correlations.png`、`distance_future.png`、`forecast_skill.png`、`examples.png`。
- `past_future_scatter.png`：真正以历史欧氏距离为横轴的散点与条件均值曲线；跨查询混合曲线受查询难度影响，应结合 query 内相关性解读。

重点是第六项：计算每个 query 内的 `Spearman(d_past, MSE_future)`，再汇总；正值支持“过去更相似对应未来误差更低”。同时比较最近十个与随机历史，以及去重候选的统计。不同量纲的 MSE 不能直接混合，所以总体图除以 query 所属序列的训练方差；原始 MSE 仍保留在明细。

置信区间按逻辑序列整簇 bootstrap，平坦距离/误差导致的未定义相关性单独计数。列之间仍可能相关、文件少，因此属于探索性分析；不能只凭总体 rho 宣称因果或所有数据集有效。应同时看各组结果、去重结果与真实预测基线。

## 空间和时间

保留最多 `M=8192` 个候选位置/距离，块长 `B=16384`；不会持久保存 N 个已知 query 距离。核心距离缓存约 `O(M+B)`，加长块排序数组 `O(G)`，既有索引/均值方差仍为 `O(ND)` / `O(N)`。实验输出只保存选中候选，随查询数增长，可按需清理结果。

粗略最坏查询复杂度：`O(G log G + N*D + V*L + G*M log M + M*K)`，其中 D=48，V 为精确核验数，最后一项为 episode 贪心去重上界。未来映射/预测另外约 `O(K(L+H))`，加权中位数 `O(H*K log K)`。只检索同序列时 N/V/G 限于该序列候选，但块下界排序目前仍计算全部 G 个块。

这是为预测统计增加资格筛选和未来映射的版本，不是 10B 规模或每次 ≤100ms 的保证。
