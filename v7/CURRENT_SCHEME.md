# 当前方案：恢复固定增量迁移 5+5

按用户确认，当前预测入口恢复早期固定增量迁移版。统一复用
`v7/runs/dual_five_distinct_96_20261014/<dataset>` 的冻结模型与案例库，
读取原输出 `fixed_increment_transport`。七个数据集使用同一个版本，
不按 TEST 逐数据集拼接实验分支。

历史长度 96、预测长度 96。原分支从 100 个候选重排选 5，保留 0.8 未来兼容与
0.2 历史相似评分；第二路按原始历史数值 MSE 选另外 5 个。第二路内部案例
间隔至少 192 点，十案例去重，所有完整历史加未来均在当前查询历史开始前
已经观测完。案例库保持原冻结范围。

预测采用原来的 `F5 + 0.25 * bounded_increment`。增量来自两组对齐后的旧编码器
预测残差差异，并保留原来的分歧收缩、历史支持度及修正上限。它不是后续统一
Moirai 残差实验，也不是 `guarded_dual` 验证保护输出。十个案例参与固定增量
计算；收缩可能使个别修正为零，不能把案例数当作十个独立有效信号。

Moirai 保持冻结，仅输入当前原始 96 点历史。检索案例通过原外部修正路径使用。
本次恢复不新增网络、不重训，也不声称解决 Electricity 的既有退化。

## 使用

从仓库根目录运行，`history.npy` 为该变量已观测的原始 96 点数值，`sid` 和
`start` 使用已有 Store 的变量编号与历史起点坐标：

```powershell
.venv-moirai-gpu/Scripts/python.exe -m v7 query-current --dataset ETTm1 --history history.npy --sid 0 --start <history_start> --output v7/runs/current_query.json --device cuda
```

支持 ETTh1、ETTh2、ETTm1、ETTm2、weather、electricity、traffic。Illness 原库
不足以复用相同 100 候选协议，不自动换用其他条件。

Python 接口为 `v7.current_forecast.CurrentPredictor`，提供 `predict`、
`predict_cached` 和 `close`，也支持上下文管理器。返回的 `forecast` 是当前主预测，
`primary` 固定为 `fixed_increment_transport`。`forecasts` 中其他输出仅供诊断。
入口仍执行旧实现的来源哈希及时间边界校验；本地冻结文件缺失或变更时直接报错，
不会自动改用最近一次实验或重新拟合。

后续统一残差、选择器和成对历史增量实验保留原独立入口及目录，当前入口不加载
这些网络。已有 `query-a`、`query-b` 等显式实验接口保持各自原来的语义。
模型权重、原始数据、预测与评测报告留在本地，不随代码发布。
