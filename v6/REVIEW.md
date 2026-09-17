# V6 实现审查与验证

## 本轮更新验证

在服务器提交 `3a91cda` 基础上直接优化 V6。新增 CDF 模式交换不变性、批量 Top-M 等距稳定排序、空间重排/重复重排一致性、仅 validation 融合、历史近邻采样隔离、独立布局通道对齐、原始历史门槛核验；共 15 个 V6 测试方法。实际服务器权重完成纯 learned 与联合检索的推理评估，不更新神经网络权重。见 [联合方案](JOINT_RETRIEVAL.md) 和 [优化对照](OPTIMIZATION.md)。

以下保留初始 V6 审查记录，初版的时间布局/候选堆描述由优化报告中的空间布局/批量有界候选实现更新。

## 范围

以用户上传服务器结果后的 V5 提交 `438c327` 为父版本，新增 `v6/` 与独立 CI。没有改写 V4/V5 代码、既有结果或主分支。

## 已检查的关键逻辑

- **真实未来隔离**：生产接口 `Retriever.retrieve(x, ...)` 不接收 y；真实查询未来只出现在训练监督和离线 evaluate/oracle 中。
- **时间划分**：prior 仅 fit memory；encoder 仅 fit 后续 train；validation 选择 checkpoint；test 不更新权重。所有窗口及其未来完整落于单一区间、同一有限值连续段。
- **自身排除**：默认查询与 memory 时间区间分离；另外按 sid/start 排除查询上下文+未来重叠邻域。返回候选之间排除历史窗口重叠。
- **索引证明边界**：下界针对存储 float32 embedding，用外扩包围盒与 float64 距离。没有将原始序列下界误用于学习向量，也不将未来相似度称为可保证指标。
- **冗余候选**：只维护固定容量堆及单叶块距离，没有全库距离缓存。一次搜索结束后不补扫；不足 k 明确报告。
- **预算截断**：访问/时间预算提前结束时不报告 certified；完整 top-k 与 certified 分开记录。
- **身份检查**：store data_id、checkpoint hash、配置窗口/划分必须一致；换模型不能复用旧索引。
- **保存与恢复**：checkpoint 原子替换；训练恢复检查 config/data/prior 身份并加载优化器/RNG。索引 complete=false 不能用于查询。
- **统计**：按逻辑序列簇 bootstrap；比较在同一 query ID 上配对。原始值、私有曲线及源路径不自动进入分析包。
- **数值稳定**：memory 尺度下限；避免无约束仿射斜率；混合似然 logsumexp；概率/损失使用 float32；AMP 掩码前转换为 float32。

## 本地验证

环境：Windows、conda `tslib`、Python 3.11.15、PyTorch 2.5.1+cu124、CUDA 可用。

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& 'F:\Anaconda\envs\tslib\python.exe' -m unittest discover -s v6/tests -v
```

测试覆盖：

1. 概率混合似然、贝叶斯后验与独立解析结果一致。
2. 单高斯 CRPS 和 90% 分位区间与解析结果一致。
3. CPU 前后向梯度有限；冻结 prior 没有梯度；不执行 optimizer.step。
4. CUDA AMP 前后向梯度有限；不执行 optimizer.step。
5. 改变未来目标不改变历史编码。
6. 训练对比排除重叠 episode；所有对被排除时损失仍有限。
7. 多层搜索与穷举一致，包括多分片、等距并列和 query 邻域排除。
8. 多尺度数值下，包围盒下界不大于真实最近距离；预算截断不误报精确。
9. 含 NaN 的 CSV → store → 随机权重索引 → 评估 → 统计/图/ZIP 的小型完整流程；验证无未来跨界、错误权重拒绝、回传包无私有数组/CSV 路径、同一运行配对比较为零差异。

上面逻辑组织为 9 个 unittest 测试方法。测试 checkpoint 标记 epoch=-1，任何相应指标仅是代码冒烟测试，不能作为模型效果证据。

另检查 CLI 帮助、10B 容量估算、Python 编译、Git whitespace；CI 在 Linux CPU 再执行相同测试，CUDA 项会跳过。

## 未验证的内容

- 没有在本地运行正式训练或优化器更新；训练收敛、校准、未来检索收益需服务器实验。
- 没有在真实 10B 库运行，也没有证明每次查询 <100ms。
- 未进行正式 epoch 级长训练/中断恢复压力测试；checkpoint 格式、分支和恢复状态代码已审查。
- 未支持跨序列日历时刻严格因果对齐、分布式/增量建库、跨 checkpoint 索引迁移。
- 没有保证任意配置/任意浮点极值下的理论实数精确性；实际检验针对有限有界归一化向量。
