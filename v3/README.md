# V3：不重叠的相似时间序列检索

## GitHub 仓库运行路径

本目录是仓库内的 `v3/`，根目录保留 V2。以下命令从仓库根目录开始：

```powershell
cd v3
python -m pip install -r requirements.txt
python build_native.py
python -m unittest -v test_diverse.py
```

测试会生成合成数据，不需要下载旧数据集。源码不附原始 CSV、完整查询数据、训练模型或大型索引。请先按根目录 [迁移指南](../MIGRATION_GUIDE.md) 创建自己的基础索引、级联索引与查询 JSON，然后在 `v3/` 内执行，例如：

```powershell
python diverse.py --index ../results/custom_cascade --query-json ../results/custom_query/query.json --strategy bounded --initial-candidates 512 --max-candidates 8192 --output results/bounded_query.json
```

默认旧实验路径已改为 `../results/cascade_index` 和 `../results/query.json`，克隆后这些文件并不存在，必须自行提供或显式传入路径。级联 manifest 内的基础索引绝对路径仍需按迁移指南修正。

小型历史结果已附在 [docs/benchmarks](docs/benchmarks)：只包含性能/准确性指标和对比图，不包含原始片段。本文其他 `results/...` 路径是本地实验生成物。

## 最新默认：冗余预取 + 有上限的距离缓存

CLI 默认已改为 `--strategy bounded`：首次取 512 个完整近邻，最多保留 8192 个候选，只存紧凑的距离/序列/起点数组和一个处理块；不保存全库查询距离。对照实验、内存测量与复杂度详见 [冗余候选与缓存限制报告](REDUNDANCY_REPORT.md)。下文 `prefix` 与 `hierarchy` 是仍保留的旧策略。

```powershell
python diverse.py --query-json ../results/query.json --strategy bounded --initial-candidates 512 --max-candidates 8192 --output results/bounded_query.json
```

Python 使用 `engine.search_bounded(query, initial_candidates=512, max_candidates=8192)`。达到上限仍不足 k 条且未穷尽数据库时，会返回 `candidate_limit_reached=true, complete=false, certified=false`，不突破上限或伪报完整结果。旧 `search` 层级模式可能保留大量已核验距离，要求限定距离缓存时应使用新方法。

## 1. 新的目标与排除规则

先找到当前最相似片段，再排除与它重叠的窗口，从剩余范围中寻找下一个最相似片段，直到获得 k 个结果或候选耗尽。

本次按用户确认的规则实现：查询长度 244，同一 **文件 + 数值列 + 设备** 内，任意两个返回片段的起点必须满足：

```text
abs(start_a - start_b) >= 244
```

若选中 `[s, s+244)`，禁止的整数窗口起点为 `[s-243, s+244)`。相隔恰好 244 点允许返回。同一个序列的不同内部 `sid` 也会共同排除，避免分块边界漏排。不同文件、列、设备独立处理，不按时间戳跨列去重。

这改变了 Top-k 的定义：现在是**逐次取剩余范围最近邻的贪心非重叠 Top-k**。不是普通 Top-k，也不是“让 k 个区间的距离总和最小”的全局组合优化。并列距离允许任取一个，选择不同并列项可能使后续排除区域不同。

查询本身若存在于库中，仍可作为第一个结果返回，然后排除其邻域。若希望连查询来源也预先排除，当前接口尚未提供该选项。

## 2. 两种实现

### 保留策略：`prefix`，自适应完整候选补足

1. 用 V2 的三级检索 + 全库安全补查，获得完整排序的前 128 个近邻。
2. 从近到远选择结果，每选一个，就合并它在同一逻辑序列上的禁止起点区间。
3. 不足 k 个时，将候选数量乘以 4，重新取得更长的**完整近邻列表**，再按同一规则选择。
4. 找到 k 个结果，或已取完数据库/距离阈值内候选后停止。

完整近邻列表中的第 k 个非重叠结果不可能被列表外更近的结果替代，因此不用盲目固定一个候选上限。它不同于“普通 Top-10 只过滤一次”。这一策略在首轮实测中较快，当时作为默认值；现在 CLI 默认使用增加缓存上限的 `bounded`。

此策略保证**输出和选择过程中排除重叠**，但底层生成完整候选列表时仍可能计算重叠位置；它并不承诺这些位置完全不消耗计算。

### 可选策略：`hierarchy`，搜索过程中直接排除

为每个 16,384 起点长段与 1,024 起点中段建立包围所有局部符号的安全区间摘要。使用全局优先队列按距离下界展开：

```text
长段摘要 → 中段摘要 → 局部符号下界 → 128 起点一批的精确核验
```

只有未展开区域都不可能产生更近结果时，才接受当前最优片段并排除邻域；后续展开和核验会跳过已禁止的位置。已核验距离保留供后续结果使用，不为每个排名从头重扫。零距离结果可直接证明当轮最优。

这是更直接的“找到一个就排除周围”实现，但当前长段区间下界较松，Python 队列开销较大，实测比 `prefix` 慢。两种策略均保留，可在其他数据集比较。

## 3. 使用

本目录包含独立的 V3 源码和本地内核，不修改 V2 代码。复用已经完成的 V2 基础/级联索引，不复制原始 CSV 或大型索引。

```powershell
python -m pip install -r requirements.txt
# GCC 加入 PATH，或设置 CC 为编译器可执行文件路径
python build_native.py
python -m unittest -v test_diverse.py

python diverse.py --index ../results/cascade_index --query-json ../results/query.json --k 10 --output results/my_query.json
```

显式使用搜索中直接排除策略：

```powershell
python diverse.py --index ../results/cascade_index --query-json ../results/query.json --strategy hierarchy --output results/my_query_hierarchy.json
```

Python 接口：

```python
from diverse import DiverseIndex

engine = DiverseIndex("../results/cascade_index")
try:
    # query 为 244 个有限数值
    result = engine.search_prefix(query, k=10, min_separation=244,
                                  initial_candidates=128, include_values=True)
    # 搜索中直接排除策略：engine.search(query, k=10, min_separation=244)
finally:
    engine.close()
```

注意：`DiverseIndex.search_prefix` 对应 CLI `prefix`；`DiverseIndex.search` 对应 `hierarchy`；`search_bounded` 对应当前默认 `bounded`。本目录继承的 `cascade.py`、`run.py query` 仍是普通检索接口；**需要非重叠结果时使用 `diverse.py`**。

参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `k` | 10 | 希望返回的非重叠结果数 |
| `min_separation` | 244 | 同一逻辑序列中起点最小间距，不能小于查询长度 |
| `max_distance` | 无 | 标准化欧氏距离上限；设置后可能不足 k 个结果 |
| `initial_candidates` | 128 | Python `search_prefix` 的初始完整近邻数量 |
| `include_values` | Python 默认 false；CLI true | 是否返回原始匹配数值 |

“最相似的十条”不意味着每条都非常相似；移除重叠邻居后，第十条通常更远。若只接受足够相似的片段，可设置经过自己数据验证的 `max_distance`，不要为了凑满十条放宽质量解释。

返回中 `certified=true` 表示当前已完成索引、标准化欧氏距离和保守浮点下界下，结果符合每轮最近邻的定义。`exhausted=true` 表示剩余满足约束的候选不足 k 条。没有 100 ms 强制截断，也不声称任意数据规模下硬实时。

## 4. 实测结果

库规模：1,023,529 个合法起点，约 103 万原始点，开发数据子集，不是全部 CSV 或百亿点。七类数据各两次随机原片段查询，长度 244，随机种子 20260921。计时包括常驻 API 与 JSON 序列化，排除启动、网络和穷举审计时间。

每个查询使用无索引剪枝的全量原始距离核验：逐轮检查返回距离是否为**排除之前选中邻域后**的全库最小值。同距离并列允许不同位置；这不是用普通 Top-10 的结果集合评估去重后的结果。

| 策略 | 中位 ms | P95 ms | 最慢 ms | 超过 100 ms | 每轮距离正确率 | 返回重叠对 |
|---|---:|---:|---:|---:|---:|---:|
| 优化后 `hierarchy` | 172.37 | 890.42 | 1138.53 | 12/14 | 100% | 0 |
| **`prefix`（默认）** | **104.80** | **156.65** | **183.80** | **7/14** | **100%** | **0** |

两种策略均在 14/14 个查询中补齐十条。旧普通 Top-10 过滤一次只能剩下 1～9 条。同次 `prefix` 实验的普通 V2 Top-10 中位约 47.21 ms；这两个任务约束不同，不能说去重必然更快。两种新策略分开运行，系统调度波动也会影响耗时。

测试通过：半开区间边界、恰好 244 点间距、跨内部块同源排除、不同列独立处理、超过百亿的逻辑起点、随机非原片段与常量查询、距离阈值和候选耗尽、从仅 1 个候选开始多轮补足。单元测试使用独立 NumPy 全量距离，真实库审计使用不带符号剪枝的原始距离核验。

## 5. GPU 查询的实际变化

对于之前起点 85018 的查询，仍然先返回该原片段。随后排除了：

- `temperature_gpu` 的 82212：与已选 82073 相隔 139 点，发生重叠。
- `temperature_memory` 的 84877：与已选 85018 相隔 141 点，发生重叠。
- `temperature_memory` 的 42041：与已选 42043 相隔 2 点，发生重叠。

默认策略补入 `temperature_memory` 的 78529、30989、59540，使最终仍有十条互不重叠的片段。

| 排名 | 列 | 起点 | 标准化距离 |
|---|---|---:|---:|
| 1 | temperature_memory | 85018 | 0.000 |
| 2 | temperature_gpu | 82073 | 11.894 |
| 3 | temperature_memory | 87515 | 12.223 |
| 4 | temperature_memory | 13712 | 12.499 |
| 5 | temperature_gpu | 57892 | 12.499 |
| 6 | temperature_memory | 70992 | 12.565 |
| 7 | temperature_memory | 42043 | 12.944 |
| 8 | temperature_memory | 78529 | 12.944 |
| 9 | temperature_memory | 30989 | 13.054 |
| 10 | temperature_memory | 59540 | 13.060 |

## 6. 时间复杂度

记 N 为全库合法起点数，D=48 为符号维数，m=244 为窗口长度，K 为返回数，B=16384 为长段起点数，b=1024 为中段起点数，G 为长段数（约 N/B，另受源序列边界影响）。

### 6.1 默认自适应补足策略

第 r 轮取得完整前 M_r 个近邻，M 从 128 开始按四倍增长。令 R 为轮数、V_r 为该轮精确核验次数、P_r 为触及的倒排项数、U_r 为命中的长段数。

主体计算为：

```text
T ≈ Σ_r [ O(N·D + V_r·m) + 倒排访问与候选排序 ]
```

具体实现还对触及长段排序、对块内候选排序，并逐块合并长度不超过 M_r 的结果。因此可写出包含这些开销的保守上界（固定三级候选参数项省略）：

```text
T = O(Σ_r [P_r + U_r log U_r + N·D + V_r·m
           + V_r log B + G·M_r log M_r
           + M_r log(K+1) + K²])
```

不要把它简化为 `O(log N)`。忽略排序时，最坏 V_r=O(N)，主要扫描/核验开销是 **O(R·N·(D+m))**；固定 m、D、少量补足轮数时，这部分随库规模近似线性。严格复杂度仍有排序项。

为什么一般不需要一轮只补一个？当同一逻辑序列每个起点只入库一次时，选中一条长度 m 的窗口，至多禁止 2m−1=487 个整数起点。若库中候选足够多，完整前 `(K−1)×487+1` 个近邻必能选出 K 个互不重叠结果。K=10 时该数为 4384，四倍增长会在 128→512→2048→8192 内覆盖它。代码仍保留增长到全库的兜底，以处理元数据重复、可用候选不足等情况；通用轮数上界为 O(log N)，而不是无条件固定四轮。

### 6.2 直接分层排除策略

设 S≤N 为实际展开并计算符号下界的起点数，C≤S 为实际核验原始距离的起点数。每个位置的符号下界和完整原始距离在单次查询中最多计算一次，结果被复用到后续排名。

主要开销为符号下界 O(SD)、原始距离 O(Cm)、中段局部排序 O(S log b)，以及优先队列与排除区间维护。最坏的保守上界可表示为：

```text
O(N·(D + m + log N + log(K+1)) + K²)
```

这是带排序/队列开销的全量最坏情形，不能认为只排除了邻域就自动变成次线性。优势是可按安全下界跳过区域，避免对每个新排名都从头做 K 次全库核验；实际取决于区域下界是否足够紧。

### 6.3 空间与启动

- 继承 V2 的逐起点符号、原始数据和约 16N 字节的均值/标准差缓存。
- V3 增加长/中段安全包围盒，约 O((N/b)D) 字节量级（包含上下界）。
- 层级查询最坏保留 O(N) 的符号排序与已核验候选；补足策略保留 O(M_r) 结果及原查询工作数组。
- 启动统计量准备仍为 O(Nm)，不计入常驻查询耗时。

所以目前实现满足新的非重叠精确检索规则，尚未解决百亿点、每次低于 100 ms 的扩展问题。

## 7. 复现实验与文件

```powershell
python benchmark_diverse.py --strategy prefix --output results/benchmark_prefix_retest
python benchmark_diverse.py --strategy hierarchy --output results/benchmark_hierarchy_retest
```

当前保存的结果：

- `results/benchmark_prefix/metrics.json`：默认策略逐条时延与审计。
- `results/benchmark_prefix/gpu_comparison.json`：GPU 普通结果和非重叠结果对照。
- `results/benchmark_prefix/gpu_result.json`：可直接用于绘图的非重叠结果。
- `results/benchmark_prefix/comparison.png`：耗时和去重后结果条数对比。
- `results/benchmark_optimized/metrics.json`：优化后的层级策略测试。
- `diverse.py`：两种精确非重叠策略及 CLI。
- `test_diverse.py`：独立距离与排除规则测试。

本目录没有新部署 HTTP 服务。新设备需重新编译本地库，并重新定位或重建基础/级联索引；运行时注意使用本目录的 Python 模块与匹配版本的本地库。
