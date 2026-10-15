# 5+5统一残差与独立训练消融

这个实验保留原来的100候选重排选5，以及按原始历史数值选5的第二路。
TRAIN、VAL和TEST均沿用已保存的十案例选择结果，不扩大候选数或改写重排权重。
历史和预测长度都是96。实验模块与原来的5+5实现分开，需明确指定新实验目录。

## 统一哪个参考

记冻结的Moirai 2.0预测器为M。查询与旧案例均使用相同模型、相同96点输入、
相同96点预测和相同中位数输出：

```
query_target = query_true_future - M(query_history)
case_residual = old_true_future - M(old_history)
```

旧案例不再减去自训练未来状态编码器的预测。旧案例Moirai预测优先从已校验的
历史缓存读取，缺失部分使用同一个固定版本模型补算。缓存记录来源与哈希；
实际GPU核验重算旧案例预测。Moirai骨干不参与本实验训练。

本实验没有用当前数据集TRAIN拟合M的参数，因此不会把一个TRAIN内拟合的适配器
输出当成参考预测，参考本身不需要数据集内的OOF重训。
但原入选案例所用的上游编码器曾在整个TRAIN上拟合：**整个检索流水线不是
前向OOF**，不能将本实验描述为完全前向OOF。新增适配器还做一次有清除间隔的
TRAIN前推留出诊断；该诊断不用于选择模型、轮数或TEST策略。

## 五组独立训练的输入对照

所有组使用同一网络结构、参数量、三个固定种子、优化器和训练轮数。每组独立
拟合自己的参数，不把正常模型在推理时简单置零当作充分消融。

| 组 | 输入给新增网络的信息 |
|---|---|
| query_only | 当前历史、Moirai分位数、已观测历史的64→32回测误差 |
| history_only | 上述信息，加原5+5案例历史及由旧历史得到的Moirai预测特征；不提供旧真实未来 |
| real | 上述信息，加十个旧案例的真实Moirai预测残差 |
| shuffled | 保持真实案例历史/键不变，在各自五案例组内打乱已经对齐的残差/值 |
| matched_random | 用仅根据历史状态匹配的随机旧案例替代原案例，并使用其真实Moirai残差 |

query_only与history_only使用当前已观测32点回测误差的重复序列激活同一值分支，
不是把真实旧未来假装为已观测，也不是给两个对照提供旧真实未来。历史案例对照
中的旧Moirai预测只依赖旧历史，属于历史可得信息。

随机组匹配真实案例的水平、幅度、趋势、日/周相位与年龄，从最近32个可行
历史状态中按固定种子抽样。排除与真实十案例相同的开始位置；十个随机案例
不重复，第二组五个随机案例内部间隔至少192点。随机案例与真实案例允许
部分历史区域相交，不能把它称为独立案例库。选择函数不读取任何真实未来。

打乱对照在历史拟合缩放之后打乱残差，保留每个五案例组的残差集合，避免
仅因换到不同数值尺度而人为制造坏对照。它检测历史与残差的配对价值；
若打乱后仍有效，不代表整个检索集合没有信息。

另有独立重训的`deranged`补充对照：在各五案例组内拒绝所有含固定点的排列，
确保没有一个案例保留自己的残差，仍保持组内残差集合不变。普通随机排列可能
保留少量正确配对，因此不能把普通打乱对照当成完全错配。这项加强对照在复核
普通打乱设计后补充，预算与种子相同，结果单独列出，不替换原主输出。

## 网络与预算

外部网络使用32维表示。查询按8点分为12个预测patch，每个patch读取旧历史键，
并只读取对应预测时段的旧残差。十案例加空案例形成11个注意力位置。
案例历史拟合的Wiener斜率同时缩放残差，但不对残差增加水平平移项。
输入保留历史水平、幅度、趋势、日/周状态、案例年龄，以及旧Moirai预测状态。

网络包括查询修正头与有门控的残差迁移。两者初始化输出均为零，初始预测等于M。
修正上限为Moirai分位跨度四分之一与当前历史尺度两倍中的较小者。

每组固定种子13、41、73，各训练16轮，批量512，AdamW学习率0.001、
权重衰减0.001。目标等权混合TRAIN尺度归一化的总体原始MSE与逐变量标准化MSE，
另加加权MAE和修正能量惩罚。不根据TEST选择种子、轮数或组别。
三个种子的预测取平均。原有分数文件的NMSE仍沿用源数据中的std分母，
而本实验训练与总体验证保护使用TRAIN标准差；报告必须区分这两种标准化。

## 确认输出与容量匹配

原始网络输出是基本消融结果。每组还使用相同的验证规则得到确认输出：
每变量前两VAL时间块从0.1、0.25、0.5、1中选择强度，最后一块只确认或拒绝。
随后要求三个VAL块中的总体原始MSE、TRAIN标准化MSE与MAE都不退化。
验证不能保证TEST时期不退化。

`own_real`将真实案例网络的修正加在M上；与`query_confirmed`比较时，
两边都只有三个新增网络。`extra_real`以已确认的查询网络输出为基础，
再验证是否加入真实案例网络的差异。拒绝时保留查询网络，不重置为M。
history_only、shuffled和matched_random也采用同样流程。

`extra_real`可能同时使用两组三种子网络，因此另有`query_query`容量对照：
基础查询网络加第二组独立查询网络，种子固定为原种子加100，采用完全相同的
16轮预算、强度网格与验证保护。两边都是六个新增网络。这项容量对照是在
发现混合输出的容量问题后补充的，不将其称为预注册盲测。

报告同时比较三对三的原始/own输出、六对六的确认输出，并单列不增加网络的
固定残差均值迁移。不能只用extra_real相对单组查询网络的改善证明RAG有效。

## 时间边界、核验与局限

所有真实/随机旧案例完整192点都必须在查询历史开始前观测完，且不超过冻结库。
TRAIN查询的真实未来也必须完全落在TRAIN内。输入函数不读取查询raw_y。
核验覆盖物理数据来源、参考预测、随机选择规则、打乱映射、每个策略和指标重建，
并在实际推理路径上检查篡改当前及同步其他变量未来不会改变预测。

这批TEST时期已在先前实验中查看过。上游TRAIN检索选择也不是前向OOF。
结论属于开发期探索，不是新盲测或全流水线无偏因果估计。原VAL也参与过上游策略选择，本轮末块只确认新增层，不是全流水线全新的独立确认集。原5+5与本实验的
差别包含参考、外部网络、训练目标和确认规则，不能把全部差值仅归因于统一残差。
Illness原初始库不支持同样的100候选预算，不给它伪造同条件结果。

## 复现与部署

```
python -m v7.prepare_unified_residual --source <frozen-ten-feature-root> --output <new-root> --datasets <names>
python -m v7.benchmark_unified_residual --root <new-root> --datasets <names>
python -m v7.benchmark_unified_capacity --root <new-root> --datasets <names>
python -m v7.benchmark_unified_derangement --root <new-root> --datasets <names>
python -m v7.verify_unified_residual --root <new-root> --datasets <names>
python -m v7.verify_unified_residual --root <new-root> --datasets <names> --live
python -m v7.verify_unified_residual --root <new-root> --datasets <names> --capacity-only
python -m v7.verify_unified_residual --root <new-root> --datasets <names> --capacity-only --live
python -m v7.verify_unified_residual --root <new-root> --datasets <names> --derangement-only
python -m v7.verify_unified_residual --root <new-root> --datasets <names> --derangement-only --live
```

部署入口为`query_unified_residual.UnifiedResidualPredictor`，默认主输出字段
`forecasts['extra_real']`；`own_real`作为明确的另一种已固定输出保留。
`used_cases=0`表示本次新增案例修正被验证规则拒绝，不能据检索数量宣称案例有效。
复用旧运行目录需要全部源文件、模型、案例缓存与策略哈希通过校验。

只发布代码、测试和机制说明。数据、历史预测库、训练权重、案例轨迹、指标、
实验报告都留在本地忽略目录。
