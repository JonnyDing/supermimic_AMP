# WMRL / SuperMimic

当前训练入口为 `scripts/train.py Unitree-G1-AMP-Flat`，详细命令见根目录 [README_zh.md](../README_zh.md) 和 [README.md](../README.md)。

默认使用 4090 环境、seed 42、BF16 网络计算、WM 整批 2048、actor 整批 1024、32 步想象。网络从随机初始化开始，不加载 PPO 策略或 teacher。真实采集、世界模型拟合、想象 BPTT、critic 和 AMP 判别器交替更新。

- `train_fast.py`、`cli.py`：训练循环及命令行配置。
- `core/`：策略、世界模型、归一化、序列 replay、想象更新及检查点。
- `mdp/`、`adapters/`：可微运动学、AMP、恢复目标及 mjlab 采集。
- `evaluation/`：固定速度和跌倒姿态评估、指标及导出工具。

`model_<iteration>.pt` 和 `last.pt` 保存网络、优化器及统计量，续训时重新采集 replay。训练指标写入当前运行目录的 TensorBoard 文件。

当前仅保留一个训练实现和固定场景评估。旧阶段训练器、验证/探针/报告生成模块、未使用的配置与接口骨架、teacher 分支和重复 rollout 已移除。播放与 sim2sim 导出使用根目录 `scripts/play.py`、`scripts/export_sim2sim.py`、`scripts/sim2sim.py`。
