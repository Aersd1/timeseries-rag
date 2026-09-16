# V5 审查与验证记录

## 开发机检查

- 语法编译通过；CLI `--help` 正常。
- 5 项纯 CPU/NumPy 测试通过：有限段跨块边界、时间分割/affine 映射、CSV 导入与数据 SHA/标签隔离、乱序时间拒绝、统计/图像/回传 ZIP。
- 本机未安装 PyTorch；3 项模型测试显式跳过，没有安装大型依赖，没有调用正式训练命令。
- 测试教师使用小型合成序列的独立 NumPy 暴力 Oracle，不构建真实项目数据索引，不拟合神经模型。

## 模型与集成检查

`test_model.py` 检查 primitive 尾部 duration、段级因果性、平坦输入、五类任务及 VQ 梯度、KL 方向、随机未训练权重的序列化/建库/评估/打包。仅做 backward 梯度校验，**没有 optimizer.step**。随机权重生成的报告被标记为未训练，不能视为效果实验。

分支包含 GitHub Actions CPU 检查 `.github/workflows/v5-tests.yml`，安装 CPU PyTorch 后运行全部测试。实际状态以该分支 Actions 记录为准；未完成的 CI 不等于通过。GPU、CUDA AMP、完整 epoch 训练和大数据吞吐必须在目标服务器验证。

## 已审查并修复的风险

- 自身/未来泄漏：query 段与 memory 分离，候选完整 maxH future ≤ memory_end ≤ query_start。
- 训练/验证/测试混用：checkpoint 选择只读取 validation；test 只在 evaluate 使用。
- 混合精度：quantization distance 显式用 float32；KL 的 -1e9 mask 在 float32 logits 上执行。
- 填充样本：KL、重构、rate 和 VQ 均按 valid mask 去除重复 padding；不让补齐副本影响统计。
- 数据身份：所选数值字节 SHA256 进入 catalog ID；checkpoint、labels、Token index 检查同一 data_id，index 还检查 checkpoint SHA。
- 文件与内存：分块 CSV、有限运行区间、mmap LRU、原始位置引用；显式关闭句柄。没有全库 query 距离持久表。
- 指标口径：strict teacher-ID recall、grid coverage、并列标记分开；actual token bytes 与点数压缩分开；cosine scan 不命名为 ANN。
- 回传：ZIP 使用文件白名单和 SHA 清单，不含原始序列/权重；配置摘要去掉数据路径。

## 尚未验证

没有训练效果、泛化收益、GPU 内存实测、100ms 保证或 10B 扩展性证明；没有完整 epoch/中断续训的服务器实测。请先在服务器测试，再从小规模数据开始。缺少这些结果不会被合成检查替代。
