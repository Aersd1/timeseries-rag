# 设计依据与 V5 复查

## 已阅读的相关工作

|工作|相关思路|本实现采用的部分与区别|
|---|---|---|
|[Mixture Density Networks, Bishop 1994](https://www.microsoft.com/en-us/research/publication/mixture-density-networks/)|神经网络输出条件混合分布，表达同一输入可能对应多个输出|用于 `p(Y,Z|X)`；没有复制论文代码，不宣称对所有未来分布都充分|
|[Predictive-State Decoders, NeurIPS 2017](https://arxiv.org/abs/1709.08520)|通过未来预测辅助监督，让内部状态携带未来信息|使用未来预测和概率特征重建辅助损失；这里是用于检索的卷积编码器|
|[Contrastive Predictive Coding, 2018](https://arxiv.org/abs/1807.03748)|在表示空间中用预测与对比学习提取信息|使用未来相似度软对比监督；不是 CPC 原始的时间步 InfoNCE 复现|
|[Deep State Space Models for Time Series Forecasting, NeurIPS 2018](https://papers.nips.cc/paper/8004-deep-state-space-models-for-time-series-forecasting.pdf)|结合神经网络和概率状态空间模型预测|支持“历史推断潜在未来状态”的设计方向；V6 是条件混合 DAG，不是该论文的线性状态空间递推|
|[Probabilistic Contrastive Learning, ICML 2023](https://proceedings.mlr.press/v202/kirchhof23a.html)|对比表征可以表达模糊输入的不确定性|启发不确定性诊断；V6 没有复现其球面后验或继承其理论定理|

这是基于上述思路的工程研究组合，不是某篇论文方法的原样复现。论文不能证明本项目的编码一定改善未来检索，必须用配对实验回答。

## 用户上传的 V5 结果

基线提交：`438c327`，分支 `v5-unified-temporal-tokens`。

检查了 `runs/v5_nrel/test/REPORT.md`、训练 history、NREL 配置、服务器代码改动。

- 测试查询 64，4 个逻辑序列；训练每序列目标 48 个查询，8 个 epoch，属于小规模实验。
- Token 码本仅使用 16/256，困惑度 5.10，需要关注表示退化。
- stride=16 的库对教师 Top-k 精确起点覆盖上限约 0.0688；这不是“未来正确率 6.88%”，也不能完全归因于编码质量。
- H=244 时 cosine_predictive 平均 NMSE 44311，而 persistence 195.91；H=96 也有异常放大。小样本下不能断言所有查询都变差，但不支持稳定有效。
- V5 对检索片段未来进行无约束线性映射，且历史尺度下限极小，可能放大近常值历史后的变化；这是代码机制上的风险判断，未单独实验确认异常值全部由此产生。
- 预测向量主要在历史候选中重排，历史粗选丢失的候选无法由未来重排恢复。
- 用户更新的 V5 加入 checkpoint 非安全加载回退。V6 独立保存纯 tensor/基础类型 checkpoint，使用 `weights_only=True`，不继承该回退。

## V6 对应的变化

1. 不再使用 VQ Token 作为唯一入口；直接检索 learned/belief 向量。
2. 默认 stride=1，以覆盖所有合法 memory 起点；显式报告窗口数和存储代价。
3. 拟合未来多模式分布，并监督编码保存它。
4. memory 标准差的 0.1 倍作为默认归一化下限；不使用无约束仿射斜率。NMSE 主指标按 memory 方差归一化，不能直接和 V5 上述数值比较。
5. 原片段不入评估查询的 memory 库；进一步检查自身及上下文+未来重叠排除。
6. 限制搜索时同时报告精度损失和速度，而不把预算搜索标记为精确。

## 仍存在的研究限制

- 高斯模式内的条件独立假设偏强，模式也可能塌缩；需要查看概率校准、模式熵和编码有效秩。
- 当前概率网络只用 memory 前缀训练，避免直接看过编码器训练目标；若数据漂移严重，其分布可能失准。
- 对比正负样本来自批内，不是已经做了全库 hard-negative mining。
- 32 维向量是有损表征；这里没有证明其距离等价于真实未来距离。
- 时间相邻的 embedding 包围盒可能很松；最坏无法有效剪枝。
- memory 前缀和其他划分按每序列点数比例切分。默认只检索同序列；跨序列/跨文件时没有日历时间对齐保证，不能当作严格在线因果回测。
- stride>1 只对采样库完整，不覆盖原始所有起点；完整模式也不保证库外候选。
- 当前单机 numpy/PyTorch 原型没有分布式路由、SSD 预取调度、产品量化和 10B 负载测试。
