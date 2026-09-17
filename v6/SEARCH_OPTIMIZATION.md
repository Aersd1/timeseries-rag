# V6 检索执行优化

基于服务器提交 `48dceda`，保留新增的 PatchTST 与联合编码器。此次沿用 V4 中经过实测的优化方向：有界候选堆、提前排除不合法位置、批量去重，以及减少重复数据访问。

## 直接使用

在仓库根目录运行：

```bash
python -m v6.build_native
python -m unittest discover -s v6/tests -v
```

之后原有 `python -m v6 query/evaluate` 调用无需修改。原索引和 checkpoint 可直接使用，不需要重建索引或重新训练。

编译器支持 GCC/Clang，或通过 `CC` 指定。应在目标设备编译；不要复制其他设备的 DLL/SO。没有编译器时仍可运行，自动使用 NumPy 候选池；返回的 `stats.candidate_backend` 会明确显示 `native` 或 `numpy`。首次编译后请重启已经导入 V6 的 Python 进程。

直接调用 `Library.search(..., candidate_backend='native')` 可要求使用 C 实现，此时缺少库会明确报错；`'numpy'` 强制备用实现；默认 `'auto'` 优先使用 C。

## 改动与不变的检索语义

- **原生有界候选堆**：以 `(distance, sid, start)` 排序，最多保留 M 个候选；避免每个叶节点重新分配并分区整个候选池。查询结束只排序一次。
- **保持距离算术**：编码距离仍由原来的 NumPy float64 平方差求和计算；C 只管理已计算的距离，避免更换浮点累加顺序改变并列候选。
- **小型节点包围盒缓存**：为已查询的通道缓存 float64 节点边界；数据向量仍按需映射，不复制全库 embedding，也不缓存全库的查询距离。
- **查询内复用数组引用**：每个实际访问到的分片只解析一次起点和向量数组；使用不复制数据的 ndarray 视图，避免逐叶重复执行内存映射对象处理。
- **资格提前检查**：先排除查询附近的非法窗口，再计算其余位置距离；完全非法的叶不读取向量距离。仍保留该叶的预算计数，因此固定叶预算的访问语义不变。
- **批量非重叠选择**：保持按距离顺序贪心选择和原长度排除规则，用向量掩码替代 Python 中逐候选逐已选项比较。
- **原始历史门槛批处理**：joint 仍使用相同历史标准化、NMSE 阈值、64 候选批次及选择顺序；批量维护重叠掩码，不放宽历史相似度门槛。
- **批量候选对象转换**：一次将有界候选数组转为 Python 数值，避免数千次 NumPy 标量索引转换。

没有修改模型、编码、候选容量、历史门槛、原始索引布局或叶预算。`scored` 现在只统计实际计算距离的合法窗口，旧版还计入被排除的窗口，比较该字段时应注意口径。

完整搜索保持原来的 TopM 前缀语义。joint 是在此候选池上应用历史门槛和非重叠规则：如果返回不足 K 个，仍可能需要更大的候选池，不能宣称全库符合历史门槛的 TopK 已经找齐。预算模式继续明确标记为近似。毫秒预算仍为按节点检查的软预算，不承诺绝对时限。

## 时间与空间

每次叶节点接受/替换候选的代价约为 O(log M)，查询末尾排序 O(M log M)。堆空间 O(M)，距离临时数组最多一叶，另外有树遍历队列。节点包围盒缓存大小与索引节点数和维度有关，为 O(TD)，不是 O(N) 个查询距离。

历史门槛每次最多读取 64 个长度 L 的候选历史。向量化去重仍有 O(MK) 上界，但主要计算从 Python 循环转为 NumPy。整体索引最坏情况仍可接近扫描，不保证 O(log N) 查询或 10B 点下的 100ms。

## 复现新旧版本同机对照

先保存已信任的服务器旧版本代码。下面的命令只是提取代码用于对照，不切换分支、不覆盖当前文件：

```bash
python -m v6.snapshot_search --ref 48dceda --output v6/runs/search_baseline
python -m v6.benchmark_search --store runs/v5_nrel/store --checkpoint runs/v6_nrel_patchtst/encoder/best.pt --index runs/v6_nrel_patchtst/index --baseline-dir v6/runs/search_baseline --output v6/runs/search_benchmark --queries 64 --repeats 3 --audit-queries 8
```

对联合卷积编码器，替换 checkpoint/index 路径为 `runs/v6_nrel_joint/...`。输出目录必须是新目录。如果使用其他服务器数据，首先按原流程准备该数据对应的 store、checkpoint、index，三者必须匹配。

每条查询使用相同的历史、模型和配置；各方法等量预热，然后随机交错重复。只读取查询历史，不使用查询真实未来或训练新模型。默认只用 CPU 单线程，不需要显存。

结果包括：

- `REPORT.md`：端到端 P50/P95/max、不超过 100ms 比例、召回与预测差异。
- `benchmark.json`：每条查询的完整指标、checkpoint/index 哈希、基线提交及环境。
- `latency.png`：速度对比图。

可用 `python -m v6.analyze_search --results v6/runs/search_benchmark --output v6/docs/search_audit` 从结果 JSON 重新生成便携报告与图像。

端到端时间包括查询编码、检索、历史过滤和候选未来读取；不含启动加载。不要把已经编码后的纯索引时间与这个端到端指标混淆。

报告区分三种召回：相同预算下的新旧一致性、相对旧版完整搜索的召回、对逐窗口暴力向量距离的审计召回。相同预算的 100% 只表示未因优化新增遗漏；不能证明原有近似预算本身就有全库 100% 召回。审计同样保留 TopM 后再应用历史门槛，不能掩盖候选池不足。

本机报告：[PatchTST](docs/search_optimization/patchtst/REPORT.md)、[联合编码器](docs/search_optimization/joint/REPORT.md)。这些是约 59.7 万个窗口的 NREL 案例库测试，不是全部原始 CSV 或 10B 规模的性能保证。
