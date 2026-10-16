# 原5+5上的闭式局部适应：快速筛查

本实验复用原固定5+5的案例及同一冻结Moirai预测，默认入口仍为
[CURRENT_SCHEME.md](CURRENT_SCHEME.md)。它用于快速判断局部适应有没有信号，
不是完整元训练模型，也尚未实现当前历史前缀上的回测。

## 模型

将完整原始96点历史按自身末16点水平和历史尺度标准化，保留全部96点形状，
并加入原始水平、幅度和日/周相位。特征标准化与PCA8仅拟合TRAIN。
加截距后，历史描述为9维。训练标签为同一个冻结Moirai的96步预测残差，
压缩为12个8点块。TRAIN拟合9×12的全局岭回归头W0，正则项10。
所有数据集使用相同设置。

每条查询依然使用原100候选重排五个，加原始历史数值检索五个。
十案例的实际历史经同一个PCA映射得到Phi；实际已知旧未来减去对应旧Moirai
预测，按旧历史尺度归一化得到R。当前查询的真实未来不参与输入。

```
delta_W = (Phi.T @ Phi + lambda*I)^-1 @ Phi.T @ (R - Phi @ W0)
W_local = W0 + delta_W
forecast = Moirai(query) + decoded(phi_query @ W_local)
```

修正恢复查询历史尺度并插值到96点，保留分位跨度与历史尺度限制。
它通过完整的旧历史—旧未来配对拟合当前查询的小预测头，不直接拼接案例到Moirai。
所有头均为闭式求解，无GPU训练、无Moirai微调、无新时序编码器元训练。

## 对照与轻量检查

- no_rag：相同全局PCA及W0，不进行案例适应。
- real：原十案例做局部适应。
- checked（预先固定主输出）：real加支持集内部迁移检查。
- deranged：保留历史输入，在各五案例组内错排归一化残差，保证无固定点。
- random：相同预算，用原已准备的历史状态匹配随机案例及其真实残差适应。
- mean：直接平均十案例Moirai残差，不拟合局部关系。
- restored：原固定增量迁移5+5，作为固定比较对象。

支持集内部检查逐个留出旧案例，同时排除与其完整192点区间重叠的其他案例。
用剩余至少三个案例适应后，比较对留出案例的预测与全局头预测。至少三个
留出案例可评估、平均损失下降且至少一半改善时才放行。所有标签均已在当前
查询历史开始前观测完。它测的是所选支持集内的迁移性，尚不能直接证明
这些规律适用于当前查询，也不是全流水线的前向OOF：全局头已在整个TRAIN拟合。

## 冻结协议

局部正则项固定候选1、10、100，融合强度候选0.1、0.25、0.5、1。
前两VAL块选择一次，清除跨入最后VAL块历史区间的完整192点查询；最后一块
只确认或拒绝。选择和确认要求原始MSE下降且MAE不增加。拒绝局部适应时保留
已确认的无检索分支。无检索分支也用相同VAL确认相对直接Moirai是否启用。
验证保护不能保证TEST不退化，且本轮不声称NMSE也受到硬约束。

所有七个数据集策略先冻结，再统一打开本实验TEST。各消融单独按同一规则
选择参数，另报告统一lambda=10、完整强度的原始诊断，不按TEST挑赢家。
因为受确认规则影响，最终控制输出相同并不等于它们的原始适应行为相同。

输入缓存和策略核对哈希，旧案例完整192点必须结束于查询历史开始之前，
真实案例必须与原5+5保存选择一致。验证脚本回放原输出、检查物理旧历史/未来，
并篡改当前目标验证预测与支持集检查均不变。

## 复现

```powershell
.venv-moirai/Scripts/python.exe -m v7.benchmark_local_pair_adaptation --output v7/runs/local_pair_quick_new
.venv-moirai/Scripts/python.exe -m v7.verify_local_pair_adaptation --root v7/runs/local_pair_quick_new
.venv-moirai/Scripts/python.exe -m unittest v7.tests.test_local_pair_adaptation -q
```

另有明确标记为事后补充的零先验对照：`benchmark_local_pair_zero_prior`。
第一轮的Weather无检索头明显退化，因此该补充去掉TRAIN监督拟合的全局修正，
令W0严格为零、无检索预测严格等于Moirai，仅保留从TRAIN历史拟合的PCA。
所有局部修正只能来自已观测旧案例。它复用相同局部求解器、消融与VAL规则，
七组策略同样先全部冻结，再统一评测；不替换第一轮主结果或按TEST混合数据集。
该分支也尚未实现元训练编码器与当前历史前缀回测。

```
python -m v7.benchmark_local_pair_zero_prior --output v7/runs/local_pair_zero_new
python -m v7.verify_local_pair_adaptation --root v7/runs/local_pair_zero_new
```

输出必须为新目录。数据、权重、预测、指标和报告均保持本地，不上传。
原检索器在全TRAIN拟合且使用过旧VAL，这批TEST也已多轮查看。因此本轮是
快速可行性筛查，不能用来宣称论文创新成立、稳定泛化或已证明恢复局部动力学。
