# V5 审查与验证记录

## 开发机检查

- 语法编译通过；CLI `--help` 正常。
- 5 项纯 CPU/NumPy 测试通过：有限段跨块边界、时间分割/affine 映射、CSV 导入与数据 SHA/标签隔离、乱序时间拒绝、统计/图像/回传 ZIP。
- 最初默认 Python 没有 PyTorch，3 项模型测试跳过；随后按用户要求切换到已有 conda `tslib`，全部 8 项 CPU/集成测试通过，再单独通过 1 项 CUDA AMP 测试。共 9 项通过，没有安装大型依赖，没有调用正式训练命令。
- 测试教师使用小型合成序列的独立 NumPy 暴力 Oracle，不构建真实项目数据索引，不拟合神经模型。

`tslib` 实测环境：Windows，Python 3.11.15，PyTorch 2.5.1+cu124，NumPy 2.1.2，SciPy 1.16.3，Pandas 2.3.3，Matplotlib 3.10.8。CUDA 可用。原 8 项测试用时约 8.84 秒，新增 CUDA 检查约 6.19 秒（不含 Python/PyTorch 启动）。

## 模型与集成检查

`test_model.py` 检查 primitive 尾部 duration、段级因果性、平坦输入、五类任务及 VQ 梯度、KL 方向、随机未训练权重的序列化/建库/评估/打包。仅做 backward 梯度校验，**没有 optimizer.step**。随机权重生成的报告被标记为未训练，不能视为效果实验。

GitHub Actions CPU 检查已成功完成：[运行 35078485343](https://github.com/Aersd1/timeseries-rag/actions/runs/35078485343)，对应代码提交 `d93f4b4571f6229145338b513a42c7e1c85b43e2`。检查安装了 CPU PyTorch，运行原 8 项测试，包括前向/反向和完整合成推理链路；没有正式训练。后来增加的 CUDA 测试在 CPU runner 上允许跳过，已在本机 tslib 的 CUDA 环境通过。

`test_cuda.py` 只用两个 64 点合成 query 和 8 个候选，检查 float16 autocast、GradScaler 反向和梯度有限性；不执行 optimizer step。可用显存低于 512 MiB 时主动跳过。

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

没有训练效果、泛化收益、正式训练 GPU 内存实测、100ms 保证或 10B 扩展性证明；没有完整 epoch/中断续训的服务器实测。CUDA AMP 仅做小型数值正确性检查。请先在服务器测试，再从小规模数据开始。缺少这些结果不会被合成检查替代。
