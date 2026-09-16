# 原文一到十二与实现对应

来源：[用户提供的 V5 方案](https://chatgpt.com/s/t_6aaa52305f308191976d4924c6a435dd)。范围是固定 primitive 的统一表示可行性，不含后续动态边界、BPE、Mamba、自回归 future Token。

| 原文 | 实现 | 关键约定 |
|---|---|---|
| 一：V4 作为 Teacher | `teacher.py` 调用未修改的 `v4/forecast.py` | 候选是 query 之前的固定记忆；保存 exact past distance 和三 H 的 affine future NMSE |
| 二：统一 Token 可行性 | `model.py` + `evaluate.py` | 压缩、召回、预测分别评估，不声称 SOTA |
| 三：4/8 点 primitive | `UnifiedTokens.encoder` | 小 MLP 只读取已完成的当前 primitive；非整除尾部用 mask 和真实 duration |
| 四：真正离散 codebook | `nn.Embedding` + nearest-code VQ | Token 记录 `(uint16 code, uint8 duration, float32 mean, float32 std)`；hard IDs 用于部署 |
| 五：多目标联合 | `objective` | 五个任务损失加 VQ 辅助项，权重均可配置 |
| 六：Rate | 可学习 categorical prior + soft-assignment expected bits | 可微代理，不是已实现熵编码器；还报告实际字节及 hard-ID entropy |
| 七：Reconstruction | codebook decoder + 已知 level/scale | 按窗口已知尺度归一化重构误差；默认较低权重；另报告使用码数/困惑度检查 collapse |
| 八：Shape distillation | `KL(p_shape || p_token)` | teacher softmax 用实际欧氏距离 d，不用 d²；student 是共享 Token 的 shape readout cosine |
| 九：Future distillation | `KL(p_future || p_predictive)` | 每个 H 分别构造分布、平均 KL；query future 只用于训练标签和离线评分 |
| 十：两个 Head | `shape_head` / `predictive_head` | 共用同一组 Token IDs/codebook；前者 duration-weighted pooling，后者 causal GRU 最后状态 |
| 十一：Token forecasting | 小单向 GRU + raw-future head | 直接读取共享 hard Token embedding 和元数据；训练预测三 H 的数值 future |
| 十二：同一表示三用途 | Token forecaster、cosine/BM25 index、predictive rerank + affine analog | 同一 checkpoint 建库；checkpoint SHA 校验防止混用不同码本 |

## 标准化和因果性

局部 primitive mean/std 只由该 primitive 的有效点得到。该 primitive 的 Token 在段末才可获得：这是**段级因果性**，不是段内每个点实时可见。预测发生在整个 query 结束以后。GRU 单向，模型接口只接收 query past 和来自历史记忆的尺度下限。

Token side features 为 `(mu_j-mu_query)/scale_query`、`sigma_j/scale_query`、`duration/p`。query mean/variance 由 Token 元数据的加权矩计算，因此预测器没有绕过 Token 直接读取 raw future/隐藏原始点。重构只能靠 hard code 和各段 mean/std。

硬量化前向用最近 code，训练用 straight-through 将任务梯度送入 encoder。码本通过 codebook/commitment 项以及 rate prior 学习。Rate 采用 soft assignment 期望编码成本，forward 的 code 仍然离散；评估记录 hard-ID entropy 和实际 11-byte 记录，二者不混淆。

query NMSE 分母为 `max(std(query), memory_std*1e-4, 1e-6)^2`。memory_std 只用固定记忆前缀估计。没有用 query future 的方差。与 V4 按训练序列方差的 NMSE 不同，不能直接拿 62.1% 旧数字比较。

## 教师和时间划分

每条逻辑序列的前 25% 固定为 memory，后面依次为训练查询、验证查询、测试查询。每个 query 的上下文和最长 H 都在所属划分。三个 H 共用能容纳 max(H) 的候选集合，因此教师的精确性相对于这个共同合法集合成立。每个查询的原片段及相邻重叠片段均不可能进入 memory。

CSV 多设备按 device 列分开；非有限值分成连续段，禁止跨段取 future。可配置 timestamp 校验和 expected_interval；不排序、不插值、不跨缺口伪造连续性。

教师生成只存 query/candidate 位置和距离/标签。训练随机保留前 8 个、部分 hard negatives（past 位于较近一半且平均 future NMSE 位于较差四分位），以及剩余随机候选；默认每次 64/256，降低显存。KL 是这个子集内重新归一化的分布，不冒称全库分布。验证使用确定的前 64 个；checkpoint 根据 validation 的直接预测误差选择，不看 test。

episode_id 是固定 `(L+maxH)` 时间桶标识，**不是严格贪心非重叠证明**。V5 默认评估普通 TopK 以隔离 Token 表示变量；没有把它称为独立历史 episode TopK。

## 检索范围与复杂度

- V4 教师：按单条逻辑序列构建/加载符号索引，使用冻结的精确合法前缀搜索。有限数值容差和零距离并列沿用 V4。
- Token cosine：从同序列采样 memory 库全量分块扫描向量，O(Nd)；候选 top-K 内存有界，不是 ANN。
- Token BM25：unigram/bigram SQLite 倒排，成本取决于命中 posting 数；SQL 聚合可能扫描大量文档并落临时磁盘。不能保证 O(log N)。窗口 Token 长度恒定，BM25 的长度归一化项为 1。
- Query 首先用 shape cosine 或 Token BM25 取粗候选，再比较 exact past distance 与 predictive-head 两种排序。只在选中候选内做精确核验，不能证明粗检索未漏掉全局近邻。
- Token 索引默认采样 stride=16，教师 stride=1；严格 Top10 位置召回会包含网格遗漏。需要较高真实起点 Recall 时将 stride 设为 1 并承担存储/时间成本；不得以 top-256 pool 内重排代替全库检索评估。
- 所有 teacher index、Torch checkpoint、tokens 都属于离线产物；不上传 GitHub。store 和 Token 索引内部路径相对，可整体迁移；旧 V4 oracle 的 manifest 含绝对路径，迁移时重建该缓存。

## 评估和回传

`evaluate` 按固定验证/测试查询输出：Token 直接预测、persistence、raw-shape analog、random analog、cosine/BM25 各自的 exact/predictive analog。记录分组 MSE/MAE/NMSE、future usefulness、strict teacher Top10 Recall、library coverage、码本使用、实际 bytes、P50/P95。

`analyze` 做配对 persistence 比较和按逻辑序列簇的 bootstrap，生成图、Markdown、CSV/JSON、带 SHA256 清单的 ZIP。相邻列可能相关，CI 只用于探索；不同 K 的有效查询集合可能不同，应看 queries 列。

本阶段没有实现同参数量固定 Patch 预测器对照（原文第十五项），也没有把训练未完成的合成测试结果当作实测收益。
