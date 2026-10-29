# WMRL / SuperMimic

四个 G1 任务共用 `scripts/train.py <任务 ID>` 入口。完整安装、训练、续训、播放和导出说明见根目录 [中文 README](../README_zh.md) / [English README](../README.md)。

| 任务 | 默认环境数 | 想象任务头 | AMP |
|---|---:|---|---|
| `Unitree-G1-AMP-Flat` | 4090 | 解析观测、奖励与恢复逻辑 | 开启 |
| `Unitree-G1-AMP-Rough` | 512 | 学习观测、原生奖励、终止与恢复计数 | 开启 |
| `Unitree-G1-Flat` | 4090 | 学习观测、原生奖励与终止 | 关闭 |
| `Unitree-G1-Rough` | 512 | 学习观测、原生奖励与终止 | 关闭 |

训练流程：真实采集 → 世界模型拟合 → 想象 BPTT actor 更新 → GAE critic 拟合。AMP 任务另行更新判别器。Rough 保留地形扫描与地形课程，普通速度任务保留步态相位；四个 actor 均使用按时间排列的四帧历史。

```bash
python scripts/train.py Unitree-G1-Rough \
  --agent.max-iterations=14000 --agent.run-name=wmrl_scratch
```

从仓库根目录、按主 README 完成环境配置后执行。默认 BF16、开启编译、32 步想象，首次编译可能需要数分钟；`--supermimic.compile=False` 可关闭编译。默认总轮数为 100001，包含 500 轮模型预训练。新训练随机初始化，不加载预训练策略或 teacher。`rsl_rl` 仍承担网络、归一化和导入兼容，并未完全移除。

| 模块 | 职责 |
|---|---|
| `cli.py`、`agent_config.py` | 四个任务的默认设置与 WMRL agent 配置 |
| `train_fast.py` | 共用训练循环、保存和续训 |
| `core/task_model.py` | 新任务的学习式任务头、地形观测及想象更新 |
| `core/fast.py`、`mdp/recovery.py` | AMP Flat 的解析想象路径 |
| `core/` 其余模块 | 策略、动力学、归一化、replay、梯度累积与 checkpoint |
| `mdp/`、`adapters/` | 可微运动学、AMP、恢复奖励与真实仿真采集 |
| `evaluation/` | AMP Flat 固定速度及跌倒恢复场景评估 |

输出位于 `logs/rsl_rl/<任务实验目录>/<时间戳>_<run_name>/`。`model_<轮数>.pt`、`last.pt` 和 `actor_final.pt` 保存网络、优化器及统计量，不包含 replay；续训恢复学习状态后启动新 episode 并重新采集。播放使用匹配任务的 `scripts/play.py`。现有固定恢复评估、ONNX sim2sim 和机器人部署导出仅支持 AMP Flat。

迁移回归测试不依赖 pytest：

```bash
python supermimic/tests/test_task_model.py
```

已通过四个任务的 GPU 小规模训练、三个新增任务续训及 AMP Rough BF16 编译更新；这些检查不代表已验证长期收敛。低资源训练检查命令和完整参数表见主 README。
