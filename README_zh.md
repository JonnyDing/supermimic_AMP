# SuperMimic AMP — 面向 Unitree G1 的 WMRL

[English README](README.md)

基于 mjlab 的 G1 世界模型强化学习项目，目标是速度跟踪与跌倒恢复。当前训练后端为 WMRL：真实仿真采集、动力学学习、想象轨迹和通过时间反向传播（BPTT）更新策略；AMP 提供可微的运动风格奖励。

## 当前实现

- 真实 transition 写入序列 replay buffer，用于拟合世界模型。
- Actor 在想象轨迹上优化任务与 AMP 奖励，结合 critic 自举和冻结 teacher 锚定。
- 判别器通过真实策略/专家 transition 独立训练，采用最小二乘目标和梯度惩罚；想象阶段冻结其参数，保留输入梯度。
- 策略接收四帧观测（384 维），输出 29 个关节动作，控制频率为 50 Hz。
- 当前训练支持 `Unitree-G1-AMP-Flat`、单张 CUDA GPU 和 TensorBoard；默认 FP32、开启编译、想象时域为 8。

PPO 优化器和训练 Runner 已移除。保留的 `rsl_rl` 组件用于网络和归一化兼容，`OnPolicyRunner` 仅是满足 mjlab 导入的占位类。当前实现依赖已有策略初始化和 teacher 锚定，不是从零训练。

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

当前训练器还依赖以下固定路径中的初始化文件：

- `logs/rsl_rl/g1_amp_locomotion/2026-10-24_13-24-33/model_66600.pt`
- `logs/supermimic/p3_online_conservative_20261026/actor_final.pt`

第一个提供 teacher 和原始 AMP 状态，第二个提供初始策略及世界模型学习状态。这些历史文件不包含在单个已发布 checkpoint 中，启动新训练前需要恢复。续训也会检查记录的源文件。仅有 `model_14000.pt` 可以加载策略和导出，但不能满足完整训练链路的依赖。

## 训练

恢复上述依赖后运行：

```bash
python scripts/train.py Unitree-G1-AMP-Flat \
  --gpu-ids 0 \
  --env.scene.num-envs=512 \
  --agent.max-iterations=14000 \
  --agent.seed=2031 \
  --agent.run-name=wmrl_amp \
  --agent.save-interval=1000
```

日志写入 `logs/rsl_rl/g1_supermimic_locomotion/<时间戳>_<run_name>/`。查看 TensorBoard：

```bash
tensorboard --logdir logs/rsl_rl/g1_supermimic_locomotion
```

通过 `--help` 查看参数。`--agent.*` 保留现有命令行结构，WMRL 专用选项位于 `--supermimic.*`；不支持的旧算法参数覆盖会报错。`--backend supermimic` 是默认 `wmrl` 后端的兼容别名。

| 文件 | 用途 |
|---|---|
| `model_<update>.pt` | 网络、归一化和学习状态，不含 replay；可用于播放和导出 |
| `actor_final.pt` | 最终 checkpoint，与最终编号模型采用相同载荷结构，并非仅 actor 权重 |
| `last.pt` | 包含 replay 的完整训练恢复文件 |

续训需保持原运行的环境和算法配置，并将总更新次数设为更大的值：

```bash
python scripts/train.py Unitree-G1-AMP-Flat \
  --env.scene.num-envs=512 --agent.seed=2031 \
  --agent.max-iterations=20000 \
  --agent.resume=True \
  --agent.load-run=YOUR_RUN_DIRECTORY \
  --agent.load-checkpoint=last.pt
```

续训恢复学习状态、replay 和随机状态，然后重新开始仿真 episode；不是模拟器内部状态的逐位续训。

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

```bash
python -m unittest discover -s supermimic/tests -v
```

代码清理后 26 项测试通过，包含 ONNX 一致性检查；已发布 checkpoint 的加载与推理正常。清理环境缺少 pybind11，未验证原生 C++ 重新编译。早期阶段训练脚本和报告保留用于历史实验；[supermimic/README.md](supermimic/README.md) 记录这些历史，不代表另一个当前支持的训练后端。
