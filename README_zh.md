# SuperMimic AMP — 面向 Unitree G1 的 WMRL

[English README](README.md)

基于 mjlab 的 G1 世界模型强化学习项目，目标是速度跟踪与跌倒恢复。当前训练后端为 WMRL：真实仿真采集、动力学学习、想象轨迹和通过时间反向传播（BPTT）更新策略；AMP 提供可微的运动风格奖励。

## 当前实现

- 真实 transition 写入序列 replay buffer，用于拟合世界模型。
- Actor 在想象轨迹上优化任务与 AMP 奖励，结合 critic 自举，不使用 teacher 锚定。
- 判别器通过真实策略/专家 transition 独立训练，采用最小二乘目标和梯度惩罚；想象阶段冻结其参数，保留输入梯度。
- 策略接收四帧观测（384 维），输出 29 个关节动作，控制频率为 50 Hz。
- 当前训练支持 `Unitree-G1-AMP-Flat`、单张 CUDA GPU 和 TensorBoard；默认 BF16 网络计算、开启编译、想象时域为 32。

PPO 优化器和训练 Runner 已移除。保留的 `rsl_rl` 组件用于网络和归一化兼容，`OnPolicyRunner` 仅是满足 mjlab 导入的占位类。新训练随机初始化所有网络，不加载预训练策略、世界模型、判别器或 teacher checkpoint。

## 环境准备

使用 Linux、Python 3.11、支持 CUDA 的 PyTorch 和可用的 GPU 驱动。以下命令在仓库根目录、已有的 mjlab 环境中执行。`setup.py` 固定了 mjlab 1.2.0、mujoco-warp 3.8.1、warp-lang 1.12.0 和 SciPy 1.17.0。

```bash
conda activate mjlab
python -m pip install -e .
python -m pip install onnx onnxruntime tensorboard
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

将仓库根目录加入 `PYTHONPATH`，确保使用本地兼容模块。G1 观测配置依赖仓库提供的**按时间排列观测历史补丁**；如果当前 mjlab 尚未包含该补丁，先备份再替换：

```bash
python - <<'PYTHON'
from pathlib import Path
import shutil
import mjlab
source = Path('mjlab_patch/mjlab/managers/observation_manager.py')
target = Path(mjlab.__file__).resolve().parent / 'managers/observation_manager.py'
shutil.copy2(target, target.with_suffix('.py.bak'))
shutil.copy2(source, target)
print(target)
PYTHON
```

不要通过删除 `history_ordering="time"` 绕过补丁：观测顺序必须与 checkpoint 一致。

运动数据目录：

- `src/assets/motions/g1/amp/WalkandRun/`
- `src/assets/motions/g1/amp/Recovery/`

## Checkpoint 与训练前提

已提交、可用于播放和导出的模型：

```text
logs/rsl_rl/g1_supermimic_locomotion/2026-10-27_11-13-57_wmrl_amp/model_14000.pt
```

该历史模型仍可播放和导出。新训练不依赖它，也不依赖任何 `logs/` 下的文件；AMP 和恢复重置仍需要参考动作数据。

## 从零训练（4090 环境）

训练只使用一个 seed，默认值为 **42**；可选评估沿用相同 seed。命令行默认使用 **4090 个并行环境**，完整训练流程如下：

- 世界模型：2048×3 隐藏层，8 步预测损失；每轮进行 10 次、每次 2048 个窗口的优化。
- Actor：32 步 BPTT，每次 1024 个想象起点。Critic：在想象轨迹上计算 GAE，训练 5 个 epoch、每个 4 个 minibatch。
- 使用梯度累积保持等效 batch：WM microbatch 为 2048，actor microbatch 为 1024，默认均整批更新；网络矩阵计算默认使用 BF16，仿真和几何计算保持 FP32；每个等效 batch 只更新一次参数。
- Actor 观测及 WM 输入/输出统计随训练分区的真实数据更新；critic 有独立的运行输入归一化。BPTT 期间固定统计。
- 每个想象观测帧使用真实任务的噪声配置；ONNX 推理保持确定性。
- 先用当前随机初始化策略加探索噪声采集 64 步，再进行 500 个模型预训练迭代，每个迭代包含 10 次 WM 更新；这 500 轮计入总轮数，之后同时训练 actor 和 critic，不再额外预热 critic。不加载预训练策略或 teacher。
- Actor、WM、critic 初始学习率为 `2e-4`，按参考实现使用余弦调度，250000 个总迭代后降至初值的 0.03；探索标准差为 0.1。

当前 GPU 配置保留 256 步 replay，tracking 参考实现为 1000 步。此任务保留速度控制与 AMP 接口，不引入 tracking 专用目标编码器。世界模型按终止掩码学习有效前缀；想象起点只需一个有效真实状态，不要求后续存活 32 步，避免排除失败数据。这些修改尚不代表修改后的任务已验证收敛。

```bash
python scripts/train.py Unitree-G1-AMP-Flat \
  --env.scene.num-envs=4090 \
  --agent.max-iterations=14000 \
  --agent.run-name=wmrl_scratch \
  --agent.save-interval=1000
```

BF16＋1024 actor 整批更新已写入默认配置，新训练和续训都不需要追加精度或微批次参数。旧检查点中的 TF32、WM microbatch 1024、actor microbatch 256 会采用当前默认值；网络、优化器及其他学习配置仍按检查点恢复。

训练直接使用策略动作及探索噪声，不再执行动作幅值门限、异常动作重试重置、零动作替换、想象轨迹剔除或运行时参数审计。真实 episode 的终止掩码、参考实现的梯度裁剪和 AMP 梯度惩罚仍属于训练算法。


日志写入 `logs/rsl_rl/g1_supermimic_locomotion/<时间戳>_<run_name>/`。查看 TensorBoard：

```bash
tensorboard --logdir logs/rsl_rl/g1_supermimic_locomotion
```

通过 `--supermimic.wm-microbatch`、`--supermimic.actor-microbatch` 调整显存占用，不改变等效 batch。`--supermimic.utd=0.4` 平均每次更新采集 2.5 个向量环境步。默认选择 GPU 0；显式列表写作 `--gpu-ids '[0]'`。通过 `--help` 查看参数。`--agent.*` 保留现有命令行结构，WMRL 专用选项位于 `--supermimic.*`；不支持的旧算法参数覆盖会报错。`--backend supermimic` 是默认 `wmrl` 后端的兼容别名。

| 文件 | 用途 |
|---|---|
| `model_<update>.pt` | 网络、归一化和学习状态，不含 replay；可用于播放和导出 |
| `actor_final.pt` | 最终 checkpoint，与最终编号模型采用相同载荷结构，并非仅 actor 权重 |
| `last.pt` | 最新网络、优化器和统计量；不含 replay，恢复时重新采集 |

续训使用新从零训练产生的 `last.pt`（`fast_schema=3`），需保持原运行的环境和算法配置，并将总更新次数设为更大的值：

```bash
python scripts/train.py Unitree-G1-AMP-Flat \
  --env.scene.num-envs=4090 \
  --agent.max-iterations=20000 \
  --agent.resume=True \
  --agent.load-run=YOUR_RUN_DIRECTORY \
  --agent.load-checkpoint=last.pt
```

续训恢复学习状态、归一化、探索进度和随机状态，跳过已完成的模型预热，然后重新开始仿真 episode 并采集至少 64 步填充 replay；旧 schema 3 文件若包含 replay 仍可读取。不是模拟器内部状态的逐位续训。Schema 1/2 的训练文件因统计、critic 输入坐标和网络结构不同，不作为新流程的续训来源；旧 actor 仍可播放和导出。从零收敛效果需要使用修正后的采样流程重新训练验证。

## 播放、导出与 sim2sim

```bash
python scripts/play.py Unitree-G1-AMP-Flat \
  --checkpoint-file logs/rsl_rl/g1_supermimic_locomotion/2026-10-27_11-13-57_wmrl_amp/model_14000.pt \
  --num-envs 1 --viewer native
```

播放入口只负责策略预览。单独导出可移植 ONNX 包：

```bash
python scripts/export_sim2sim.py \
  --checkpoint-file logs/rsl_rl/g1_supermimic_locomotion/2026-10-27_11-13-57_wmrl_amp/model_14000.pt \
  --output-dir deploy/wmrl

python scripts/sim2sim.py Unitree-G1-AMP-Flat \
  --policy deploy/wmrl/policy.onnx \
  --viewer native --duration 60 --command 0.5 0 0
```

导出包包含 `policy.onnx`、`policy.json` 和 `scene.zip`，观测归一化已包含在 ONNX 中。导出时使用 100 组输入检查数值一致性。无界面 sim2sim 使用 `--viewer none`；添加 `--output /tmp/sim2sim.json` 保存轨迹测量摘要。

## 真机部署

板端采用 Python ONNX 推理与 Unitree SDK2 原生 C++ 通信。按照 [deploy/README.md](deploy/README.md)，使用板端 SDK 和匹配的 DDS 库编译扩展，再在机器人计算机上执行：

```bash
python deploy/deploy_g1_cpp.py --net eth0 --policy deploy/wmrl/policy.onnx
```

操作顺序：START → 默认姿态 → A → 策略；B/SELECT 停止。运行时 CRC、状态超时和看门狗阻尼保留，独立部署验证脚本已移除。软件测试及 sim2sim 结果不能替代真机闭环性能验证。

## 代码与验证

| 路径 | 职责 |
|---|---|
| `scripts/train.py`、`supermimic/train_fast.py` | 当前训练入口与 WMRL 训练循环 |
| `supermimic/core/` | 动力学、replay、想象、策略加载与 checkpoint |
| `supermimic/mdp/`、`supermimic/adapters/` | 奖励、AMP、恢复目标及仿真适配 |
| `supermimic/evaluation/` | 固定场景评估与公共指标 |
| `src/tasks/amp_loco/` | G1 任务与环境配置 |
| `deploy/` | ONNX 资源及原生机器人通信 |

添加 `--supermimic.evaluate=True` 可在训练后执行固定场景评估；AMP 回报使用当前判别器，不用于跨 checkpoint 比较。单元测试、临时性能对照工具及历史诊断报告已清理，训练、播放、导出和固定场景评估入口继续保留。
