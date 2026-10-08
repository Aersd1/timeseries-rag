# Moirai 2.0：两个独立的时间序列检索版本

本目录实现用户指定的两种方案。V4/V6 原代码不改动；A 复用 V6 的历史骨干、投影与损失结构、分层向量索引、历史门槛及类比预测；B 直接调用原始 V4 精确检索，再重排一次返回的 20 个候选。

**这两个版本都已实现。预训练 Moirai 推理与端到端正确性已验证；尚未证明替换后预测更好，也没有无损压缩结论。** 验证范围见 [VALIDATION.md](VALIDATION.md)。

## 1. 为什么替换 CDF 不一定更好

V6 的 CDF 支路把预测分布压到 9 个数值阈值 × 8 个时间桶，加均值/方差摘要共 88 维，会丢失细节。Moirai 的 384 维隐藏表示来自大规模预训练，有机会保留更有预测价值的历史信息，但它也是有损表示，且预训练目标不是本项目的最近邻排序。

A 的改变同时涉及「原概率网络换成预训练 Moirai」和「CDF 摘要换成隐藏表示」。因此收益不能直接归因于 CDF；即便 A 胜过 V6，也需要额外控制实验才能分离模型能力与摘要方式。Moirai 最后 token、32 维投影也都存在信息瓶颈。

这里说的压缩是检索表示压缩，不是可重建原始时间序列的文件压缩。默认每个 learned 向量 32×4=128 bytes、joint 64×4=256 bytes；还要加原始数组、位置映射、包围盒和模型权重。默认三个索引通道合计 128 维，每窗口 512 bytes，仅向量部分。不同版本的索引不能混用。

官方模型是 **Salesforce/moirai-2.0-R-small**：384 维、6 层、patch=16，输出 0.1～0.9 的九个分位数；单次预测 4 个 patch，长预测由官方接口递归完成。它没有 V6 混合高斯的显式密度，B 不伪造高斯 NLL。

## 2. A：用 Moirai 隐藏表示替换 CDF

```text
244 点历史 X
  ├─ 官方 Moirai 2（冻结、eval）→ 最后历史 token 的 384 维表示 r
  └─ V6 历史归一化 → 原历史骨干 → 64 维特征 u
[u,r] 448 维 → Linear(448,64) → GELU → Linear(64,32) → L2归一化 e
历史归一化 → 池化并归一化 → 32 维 h
joint = [sqrt(1-α)e, sqrt(α)h]，α=0.5
→ V6 分层索引 → Top-M → 完整历史门槛 → 非重叠 Top-K → 类比预测
```

- 采用受控替换：保留 V6 历史支路；将「FuturePrior + 88维CDF」整个概率特征支路替换为 Moirai 384维表示。
- `configs/nrel.json` 延续 PatchTST 历史支路；`configs/server.json` 延续 CNN 支路。Moirai 编码均参与二者的检索表示。
- 不经过 CDF 网格或未来时间桶；不会把训练样本真实未来送入 Moirai 特征提取。
- 历史左侧补零到 patch 整数倍，补的位置标为未观测。244点形成16个token；使用最后一个因果token，不直接平均所有token。
- 缩放、投影和 Transformer 运算调用官方 module；测试与官方 Forecast 输入转换产生的上下文表示对齐。
- Moirai 使用官方历史缩放；V6 支路、训练标签和类比映射保留 memory 标准差下限。两种缩放各有用途。
- 预训练参数不更新，始终 eval；同时明确将函数式注意力 dropout 置零。训练的是历史支路、448→32投影与辅助头。

### A 的训练

沿用 V6 的 chronological split：memory 0～25%，train 25～60%，validation 60～80%，test 80～100%。完整历史+未来必须在自己的区间内。

Moirai 已有预训练权重，因此没有本地 `train prior` 阶段。train 区间训练投影，validation 选择最佳 checkpoint。全程 float32；不接续 V6 旧投影权重，不支持本版训练断点恢复。checkpoint 内保存 Moirai 架构和权重，后续加载 A 不需要重新联网下载。

复用 `v6.model.encoder_loss`：

1. 真实未来在 H=24/96/244 上的平均距离构造软对比目标。
2. 原 `belief` 教师/重建位置改为 **Moirai latent**：0.2权重混入隐藏表示相似度；这已不是概率分布距离。
3. 乘历史相似度核，排除自己与同序列重叠的历史+未来样本对。
4. 未来预测、latent重建、历史摘要重建、表示方差和协方差辅助约束。
5. 历史近邻采样约半批，另外半批随机补充。

默认损失为 pair + 1.0·future + 0.2·latent + 0.5·history + 1.0·variance + 0.05·covariance。新 latent 的尺度与 CDF 不同，权重只是沿用起点，应在 validation 上选择。

配置保留 `version=6` 及 `components/belief_bins/belief_signature/prior_epochs`，是为了复用 V6 数据/配置格式；A 模型不使用这些高斯/CDF字段。A checkpoint 独立标记 version=7、variant=a。

### A 的索引与查询

默认 stride=1，memory 的每个合法历史起点编码一次。分片内空间重排；256向量一叶、16叉包围盒；复用 V6 距离下界和有界 Top-M，M=K×512。

`joint` 距离为 `(1-α)||e_q-e_c||² + α||h_q-h_c||²`。再对完整244点标准化历史计算 MSE，仅保留 ≤0.5 的候选；按距离贪心去除历史重叠，候选间隔至少244点。`learned/history`通道作为消融，不执行joint硬门槛。

无预算截断只证明存储向量距离下的候选前缀。历史门槛之后可能不足K，且Top-M之外仍可能有合格候选；不自动放宽阈值或重新扫描。预算模式为近似，软时间预算不包括整个模型延迟。

只有 memory 构建索引；查询强制同序列且 start≥memory_end。返回候选已发生的未来，通过 V6 均值/尺度映射，再按检索距离加权。没有候选时回退到最后值预测。

## 3. B：V4一次检索20个，再按未来重排

```text
查询历史 X ── V4 原始检索一次 ── 20个历史案例 (X_c,Y_c)
      └────── Moirai官方预测 ──── 9×244分位数 Q
候选未来 Y_c → 根据两段历史做均值/尺度映射 → Y'_c
Q 与 Y'_c 的多horizon pinball兼容分数 → 排序 → 前K个（默认5）
→ 按重排分数加权 → 预测
```

### 召回

- 直接使用 `v4.forecast.ForecastIndex` 的原始算法：存在 `search_baseline` 时调用它，否则调用原版 `search`。不依赖工作区尚未提交的 V4 加速文件。
- `k=20` 且每条查询只调用一次。内部 capacity 默认20000，是为了非重叠去重预留候选；返回给 Moirai 重排的最多就是20个。
- 默认 `selection=episodes`：候选完整历史+未来间隔至少488点，避免20个候选都是相邻窗口；可在配置中改成 `ordinary`，它可能高度重叠。
- 候选的未来必须完整处于同序列memory前缀，且早于查询历史起点。
- B 独立从相同 Store 的前25%构建 V4案例库，避免直接拿旧 V4 默认60%库与 V6 的25%库比较。
- 原始 V4 符号库仍有内存成本；保留每序列点数上限，超出则报错，不悄悄截断数据。

### 未来排序分数

Moirai仅预测查询未来；候选未来直接读取已经发生的案例后续。无需再预测20条候选的未来。

先映射候选未来：

`Y'_c = μ_q + s_q (Y_c - μ_c) / s_c`

`s=max(std(history), 0.1·memory_std, 1e-6)`。然后候选未来与 Moirai 分位数都转到查询历史归一化坐标。

对分位数α定义 pinball：`ρ_α(u)=max(αu,(α-1)u)`。默认分数：

`D_future(c) = mean_{h∈[24,96,244]} mean_{t≤h,α} ρ_α(Y'_c(t)-Q_α(t))`

低分表示该已知候选未来与模型给出的未来区间/中心更兼容。这里是分位数兼容性分数，不是概率密度、完整路径似然，也不能保证真实未来相似。通常会偏好接近中心的轨迹；不表达跨时间的完整联合概率。

分位数交叉时，仅逐时刻排序修复，不进行时间池化。另提供 `score=median_mse`，用于与中位数预测的MSE对照。

默认只按未来分数重排（`history_weight=0`）。可在validation上设置β：

`score = (1-β) D_future / median(D_future) + β D_history / median(D_history)`

两项中位数均有1e-6下限。并列时按历史距离、位置确定顺序。

### 前K个预测

默认 `top_k=5`，只从这20个中取前5个。权重 `softmax(-score/T)`，`T=max(median(selected_scores),1e-6)`，聚合已映射的候选未来。

同步返回两个对照：同一20候选池保留原 V4 顺序、相同K/映射的预测；Moirai直接中位数预测。输出20个重排候选的原排名、历史距离、未来分数、最终分数及选中权重。不足20或不足K明确标记，无候选时最后值回退，不重新召回。

## 4. 安装和执行

在仓库根目录创建独立 Python 3.11/3.12 环境。官方Uni2TS约束torch<2.5，不能与本项目V6原requirements中torch>=2.5合并安装。

```bash
python -m venv .venv-moirai
# Linux: source .venv-moirai/bin/activate
# Windows PowerShell: .venv-moirai/Scripts/Activate.ps1
python -m pip install -r v7/requirements.txt
python v4/build_native.py
python -m v7 doctor --config v7/configs/nrel.json --download
```

V4需要C编译器；Windows可设置CC为gcc完整路径。A的V6原生候选堆可选：`python -m v6.build_native`，无该库时自动NumPy。

依赖固定官方Uni2TS提交 `cfd46d4510ed8896f263116f32928eede05b0a75`，模型固定提交 `30f43ff08c8494f4943ae1521e9d4e94a0fbb389`。权重只下载到忽略的`.moirai-cache`，不上传到Git。离线机器先传缓存并设置`moirai.local_files_only=true`；不要把修改后的配置混配到旧索引。

已有 `runs/v5_nrel/store` 可直接复用。新数据请复制 `configs/server.json` 修改CSV配置后执行：

```bash
python -m v7 ingest --config v7/configs/server.json --output v7/runs/my_store
```

现有NREL数据的完整A流程：

```bash
python -m v7 train-a --config v7/configs/nrel.json --store runs/v5_nrel/store --output v7/runs/a/train --device cpu
python -m v7 index-a --store runs/v5_nrel/store --checkpoint v7/runs/a/train/best.pt --output v7/runs/a/index --device cpu
```

B不训练新神经网络：

```bash
python -m v7 index-b --config v7/configs/nrel.json --store runs/v5_nrel/store --output v7/runs/b/index
```

可将`--device cpu`改为`--device cuda`，但需先安装对应torch2.4.1的CUDA环境。当前验证在CPU完成。

单条查询历史保存为长度244的`.npy`。sid是Store逻辑序列ID；start是查询历史第一个点在该逻辑序列中的位置，不能填CSV混合行号。输出路径须不存在。

```bash
python -m v7 query-a --store runs/v5_nrel/store --checkpoint v7/runs/a/train/best.pt --index v7/runs/a/index --history query.npy --sid 0 --start 408808 --output v7/runs/a/query.json
python -m v7 query-b --store runs/v5_nrel/store --index v7/runs/b/index --history query.npy --sid 0 --start 408808 --output v7/runs/b/query.json
```

A可选 `--channel learned` / `history` / `joint`，默认joint；`--leaf-budget 128`启用近似搜索，0为完整。不要用`python -m v6 evaluate`读取A checkpoint：原评估需要高斯分布字段，本版使用独立评估。

## 5. 怎样判断是否更好

先用validation选择top_k、历史权重、A投影维度/损失权重，再固定配置跑test。查询采样完全复用V6，按同一(sid,start)配对；修改模型/索引配置需新建相应产物。

```bash
python -m v7 evaluate --config v7/configs/nrel.json --store runs/v5_nrel/store --checkpoint-a v7/runs/a/train/best.pt --index-a v7/runs/a/index --index-b v7/runs/b/index --output v7/runs/comparison_validation --split validation
python -m v7 evaluate --config v7/configs/nrel.json --store runs/v5_nrel/store --checkpoint-a v7/runs/a/train/best.pt --index-a v7/runs/a/index --index-b v7/runs/b/index --checkpoint-v6 runs/v6_nrel_patchtst/encoder/best.pt --index-v6 runs/v6_nrel_patchtst/index --output v7/runs/comparison_test --split test
```

也支持只传A或只传B。输出`REPORT.md`、`summary.json`、逐查询`metrics/timing/retrieval.jsonl`、带数据与模型/索引哈希的`run.json`。真实查询未来只出现在evaluate评分阶段。

- A vs 原V6：相同K、历史门槛、入库stride与预算才适合做归因。当前配置A的K=10，B的K=5；不要把跨版本K差异当模型收益。
- B vs `b_v4_original_order`：相同20候选、K、映射，观察重排本身收益。再与`moirai_direct`比较，检验检索是否提供额外价值。
- 主NMSE用memory方差；另存query历史尺度NMSE，不能混用。
- 配对变化及按序列簇bootstrap置信区间写入summary；序列数量很少时不确定性估计很弱。
- 记录p50/p95/max、是否返回满K；耗时不含模型初始化，B评估在每序列查询前加载V4引擎；未预热，首查询单独标识。
- 检索向量的压缩质量还应独立测相似性保持、维度/存储/速度权衡，不能以一次预测MSE替代。
- 预训练数据来源不由本项目控制；需要额外检查目标数据是否与基础模型预训练语料重合。

## 6. 文件与测试

|文件|职责|
|---|---|
|moirai.py|固定版本官方模型加载、历史隐藏表示、完整分位数预测|
|model.py / train.py|A模型、未来监督投影训练、离线checkpoint加载|
|retrieval.py|A索引构建与V6查询逻辑复用|
|rerank.py|B的V4建库、单次召回、未来评分与预测|
|evaluate.py|同查询配对评估与统计|
|__main__.py|doctor/ingest/train/index/query/evaluate入口|
|tests/test_variants.py|接口对齐、无未来输入、冻结、训练/索引/查询、重排与回退|

```bash
python -m unittest discover -s v7/tests -v
```

单元测试用随机小型官方Moirai结构，无需下载权重；这不替代真实预训练模型验证。CI在Linux编译仓库原V4并运行本版测试。

官方参考：[模型卡](https://huggingface.co/Salesforce/moirai-2.0-R-small)、[固定版本module](https://github.com/SalesforceAIResearch/uni2ts/blob/cfd46d4510ed8896f263116f32928eede05b0a75/src/uni2ts/model/moirai2/module.py)、[固定版本forecast](https://github.com/SalesforceAIResearch/uni2ts/blob/cfd46d4510ed8896f263116f32928eede05b0a75/src/uni2ts/model/moirai2/forecast.py)。模型权重遵循官方模型卡的CC-BY-NC-4.0，Uni2TS代码遵循其Apache-2.0许可；本仓库不复制模型权重。
