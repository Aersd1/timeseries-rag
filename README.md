# 分层时间序列检索 V2

## GitHub 版本快速开始

此仓库包含源码、测试和文档，不包含原始 CSV、训练后的模型、生成索引、历史结果目录或平台相关动态库。下文 `results/...` 是本地实验示例路径，克隆后不会自动存在；请先按照 [迁移指南](MIGRATION_GUIDE.md) 第 6 节接入自己的数据并建库。历史报告保留测量记录，其图表和 JSON 路径需要本地生成。

```bash
git clone https://github.com/Aersd1/timeseries-rag.git
cd timeseries-rag
python -m pip install -r requirements.txt
python build_native.py
python -m unittest -v test_cascade.py test_index.py
```

需要 C11 GCC；将编译器加入 PATH 或设置 `CC` 为编译器可执行文件路径。当前仓库编译脚本已去掉旧设备的 E 盘回退路径。`CC` 只填可执行文件路径，不填额外参数。

跨设备迁移、模型结构与自定义 CSV 接入请先阅读 [迁移与使用指南](MIGRATION_GUIDE.md)，其中包含可直接复制的训练、建库、查询示例。

## 最新：完整召回补查（默认开启）

`Cascade.search`、命令行和三级 HTTP 服务现在默认 `verification="scan"`：先取得近似候选，再检查全库符号下界并精确核验无法排除的位置。完成后返回 `certified=true`，不再因长/中段配额而漏掉全库 Top-k。`verification="none"` 保留旧的近似模式，`"tree"` 使用原精确树补查。完整召回模式不因 100 ms 到时强制截断。

```powershell
C:\Python314\python.exe cascade.py query --index results/cascade_index --query-json ../rag-search/results/full_corpus/query.json --verification scan
C:\Python314\python.exe serve.py --index results/validation_index --cascade-index results/cascade_index --port 8770
C:\Python314\python.exe benchmark_cascade.py --certify --output results/cascade_certified_retest
C:\Python314\python.exe smoke_cascade_http.py --verification scan
C:\Python314\python.exe -m unittest -v test_cascade.py test_index.py
```

服务启动时额外缓存每个窗口的均值和标准差，约 16 字节/起点；启动成本不计入常驻查询时间。新报告见 `CERTIFIED_REPORT.md`。补查支持当前已完成索引的标准化欧氏距离；不是完整原始 CSV 或百亿点性能证明，重复片段也不能唯一确定来源。

## 最新：三级近似候选检索

已增加 `cascade.py`：符号倒排长段路由 → 中段摘要筛选 → 原始点位精确计算。默认选 12 个长段、16 个中段。最新同查询对照与 HTTP 测试见 `CASCADE_REPORT.md`；此前 `ORDINARY_REPORT.md` 是旧精确树的测试，不能混用其数字。

```powershell
# 已建好的开发索引可直接查询；不需要重新建库
C:\Python314\python.exe cascade.py query --index results/cascade_index --query-json ../rag-search/results/full_corpus/query.json
# 常驻三级检索服务
C:\Python314\python.exe serve.py --index results/validation_index --cascade-index results/cascade_index --port 8770
# 同查询对照、绘图和本机 HTTP 测试
C:\Python314\python.exe benchmark_cascade.py
C:\Python314\python.exe smoke_cascade_http.py
C:\Python314\python.exe -m unittest -v test_cascade.py
```

三级服务接受 `query` 数字数组、`k`、`long_candidates`（默认 12）、`mid_candidates`（默认 16）、`include_values` 和 `verification`（现在默认 `scan`）。不执行 `budget_ms` 截止。只有显式 `verification="none"` 才运行下文旧的近似模式并返回 `certified=false`；此时调大候选数也不保证单个查询召回单调增加。有限样本低于 100 ms 不是硬实时或百亿点承诺。

要为另一个已完成的 V2 基础索引建路由库，使用：

```powershell
C:\Python314\python.exe cascade.py build --base results/validation_index --output results/cascade_new
```

输出目录必须不存在，防止覆盖已有路由库。当前构建器在内存汇总符号和倒排项，适用于开发验证，百亿点须改为分片外排构建。基础原始数据沿用现有 V2 索引，不读取原始 CSV 执行在线查询。

当前目标：244 点普通形状检索（逐窗口标准化欧氏距离），100 ms 查询目标。DTW 已退出当前优化范围。

模型学习 Fourier 频率与量化分位点；全局目录 → 分片树 → 压缩符号区间 → 原始数据精确核验。相邻相同符号码连续起点合并存储，覆盖所有步长为 1 的合法起点。引导叶节点先产生经过核验的候选距离上限，后续仍用保守下界搜索，不能靠引导步骤证明区域不存在匹配。

## 运行

在本目录中运行（Python 需要 numpy、scipy、matplotlib）：

```powershell
C:\Python314\python.exe -m unittest -v test_index.py
C:\Python314\python.exe benchmark_100ms.py --output results/ordinary_final_100ms
C:\Python314\python.exe serve.py --index results/validation_index --port 8770
```

服务启动时预加载索引元数据。POST `http://127.0.0.1:8770/search`：

```json
{"query": [244个数值], "metric": "ed", "k": 10, "budget_ms": 80}
```

`query` 须替换成合法数字数组。省略预算时 ED 默认 80 ms；显式 `"budget_ms": null` 表示完成精确搜索。`certified=false` 表示搜索未完成，返回值只是当前候选，不能声称完整 Top-10。预算为协作式截止，Windows 调度、缺页和网络可能导致端到端超过 100 ms。服务为单进程串行原型，无并发 SLA。

命令行查询默认精确完成，进程启动和索引加载不能算作常驻查询延迟：

```powershell
C:\Python314\python.exe run.py query --index results/validation_index --query-json ../rag-search/results/full_corpus/query.json --metric ed --k 10
```

## 范围与限制

- 当前测试库约 103 万原始点、102 万合法窗口，包含项目 CSV 的开发子集和一个完整 GPU CSV；不是全部 4522 个文件。
- 原片段检索也可能存在常量或重复片段，距离相同不能唯一确定来源。Top-1、Top-10、精确距离召回和源位置命中须分别报告。
- 当前未达到“百亿点、完整 Top-10、每次 <100 ms”的联合要求。限时截断不能替代召回验证。
- `DESIGN.md` 为研究和设计；`results/ordinary_final_100ms/metrics.json` 为最新普通查询逐条测量；`benchmark_100ms.py` 可复现图表。
- 百亿点容量推算见 `results/capacity_10B.json`：96 维配置按此样本分布约 975 GB 完成索引、1.79 TB 建库峰值；这是容量外推，不是速度验证。普通 ED 的 48 维配置与其容量不同。
- `serve.py` / `coordinator.py` 提供同一索引分片的服务原型；当前要求共享原始存储，尚未经过真实集群容量和网络压测。

## 研究来源

[SOFA](https://arxiv.org/abs/2411.17483) 的频率选择和量化、[iSAX 2.0](https://www.cs.ucr.edu/~eamonn/iSAX_2.0.pdf) 的分层符号索引及 [ULISSE](https://helios2.mi.parisdescartes.fr/~themisp/ulisse/) 的子序列索引提供设计参考。此实现独立编写，不是上述系统官方实现，也不是已训练的神经网络检索 agent。
