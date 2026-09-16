# V5 — Unified Temporal Token Feasibility

同一个离散 Token 序列，同时用于压缩、形状检索、未来价值检索和直接预测。本分支新增 `v5/`，不改 V2/V3/V4 算法。**没有在开发机训练模型，也不包含预训练权重或真实服务器测试结果。**

- [服务器运行说明](SERVER_GUIDE.md)：换成你的 CSV，从导入到训练、评估、结果回传。
- [原文一到十二的实现对应](DESIGN.md)：模型、目标函数、边界和解释限制。
- [审查与验证记录](REVIEW.md)：本地/CI 的检查范围。
- [配置模板](configs/server.json)。所有命令从仓库根目录执行。

```text
CSV -> 按文件/列/设备保存连续数据 -> 固定历史记忆 + train/validation/test queries
                    |
                    +-> 冻结 V4 精确搜索 -> shape distance + affine future NMSE 标签
                    |
                    +-> 4/8点 primitive -> 局部标准化 -> 小 MLP -> 共享 VQ codebook
                                                           |
                                       Token=(code,duration,mean,std)
                                           /        |         \
                                    shape head   future head   causal GRU forecaster
                                         |           |             |
                                    cosine/BM25   候选重排       原始数值 future
```

训练损失：rate + reconstruction + shape KL + future KL + forecast NMSE，另外包含标准 VQ codebook/commitment 项。只在 `train` 命令启动训练。

## 运行路径

```bash
python -m v5 --help
python -m v5 doctor --config v5/server.local.json
python -m v5 ingest --config v5/server.local.json --output runs/v5/store
python v4/build_native.py
python -m v5 teacher --store runs/v5/store --output runs/v5/teacher
python -m v5 train --store runs/v5/store --teacher runs/v5/teacher --output runs/v5/train --device cuda
python -m v5 index --store runs/v5/store --checkpoint runs/v5/train/best.pt --output runs/v5/index --device cuda
python -m v5 evaluate --store runs/v5/store --teacher runs/v5/teacher --checkpoint runs/v5/train/best.pt --index runs/v5/index --output runs/v5/test --device cuda
python -m v5 analyze --results runs/v5/test --training runs/v5/train
```

最后把 **`runs/v5/test/analysis_bundle.zip`** 发回即可分析。它包含报告、指标、每查询误差和耗时、图、配置摘要及校验和，不包含原 CSV、模型权重或完整查询/未来数组。

## 重要限制

- 所有 self-match 在时间资格阶段排除，不能用原 Top1 给模型或评估“送答案”。固定记忆库前 25%，训练 query 为 25%～60%，验证 60%～80%，测试 80%～100%；past+future 必须完整位于自己的划分。
- 服务器模板只是起点，不是已验证最佳超参数。默认 H=24/96/244、context=244，全部长度均以采样点计。
- 形状教师是同逻辑序列的 stride-1 搜索；Token 库默认 stride=16。报告会把未入库的真实起点计为未召回，并给出覆盖上限。
- cosine 基线是分块全扫描，不是 ANN；BM25 是真实 unigram/bigram 倒排检索。此版不承诺 10B 扩展性或 100ms。
- 4 点/Token 不等于磁盘压缩 4 倍：当前 token 记录包含 11 bytes 元数据/ID，4 点 float32 是 16 bytes；不含倒排表时只有约 1.45 倍负载压缩。8 点 primitive 对 L=244 是 31 tokens，负载约 2.86 倍。
- V4 的符号与统计仍消耗内存，教师按一条逻辑序列处理；`max_points_per_series` 触发时会报错，不默默截断数据。详细估算见服务器说明。
