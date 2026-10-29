# SuperMimic — Unitree G1 世界模型强化学习

[English](README.md)

本项目基于 mjlab，使用 SuperMimic/WMRL 训练 G1 速度跟踪与跌倒恢复策略。学习流程为：真实仿真采集 → 世界模型拟合 → 想象轨迹 BPTT 更新 actor → GAE 目标拟合 critic。AMP 任务额外训练运动风格判别器；普通速度任务不使用 AMP。

当前训练不使用 PPO 优化器、概率比裁剪、KL 调度或 teacher。`rsl_rl` 仍提供网络、归一化及 mjlab 导入兼容，尚未从依赖中删除。新训练随机初始化网络，不需要预训练 checkpoint。

## 支持的任务

| 任务 ID | 地形 | AMP | 默认环境数 | 实验目录名 |
|---|---|---|---:|---|
| `Unitree-G1-AMP-Flat` | 平地 | 开启 | 4090 | `g1_supermimic_locomotion` |
| `Unitree-G1-AMP-Rough` | 随机崎岖地形 | 开启 | 512 | `g1_supermimic_amp_rough` |
| `Unitree-G1-Flat` | 平地 | 关闭 | 4090 | `g1_supermimic_velocity_flat` |
| `Unitree-G1-Rough` | 随机崎岖地形 | 关闭 | 512 | `g1_supermimic_velocity_rough` |

四个任务均使用四帧、按时间排列的 actor 观测，输出 29 个关节动作，默认控制频率为 50 Hz。AMP Flat 输入为 384 维，其余任务的网络和归一化尺寸按任务构建。

- **AMP Flat**：想象阶段使用现有解析观测、奖励和延迟恢复逻辑；观测噪声来自任务配置。
- **其余三个任务**：保留原生环境奖励及终止条件，用真实转移监督观测、奖励、终止及恢复计数模型头；终止/截断边界不提供观测监督。想象阶段通过学习的观测头预测后续观测，命令、执行动作和普通速度任务的步态相位使用明确的更新规则。
- **Rough**：保留地形扫描与地形课程，想象中的后续扫描由观测头预测。
- **普通速度任务**：保留速度跟踪和步态奖励，不加载 AMP 专家动作、不训练判别器。

训练使用单张 CUDA GPU 和 TensorBoard。固定恢复评估、现有 ONNX sim2sim 及机器人部署导出仅支持 **AMP Flat**；四个任务均支持对应 checkpoint 的 mjlab 播放。已有验证覆盖小规模训练链路，尚未验证新增任务的长期收敛。

## 环境准备

使用 Linux、Python 3.11、可用的 NVIDIA 驱动和支持 CUDA 的 PyTorch。以下命令均在仓库根目录、已有的 `mjlab` 环境中执行。`setup.py` 固定 mjlab 1.2.0、mujoco-warp 3.8.1、warp-lang 1.12.0 和 SciPy 1.17.0。

```bash
conda activate mjlab
python -m pip install -e .
python -m pip install tensorboard
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

导出 ONNX 或运行 sim2sim 时另装：

```bash
python -m pip install onnx onnxruntime
```

仓库根目录需要位于 `PYTHONPATH`，以加载本地任务与兼容模块。四帧观测需要仓库提供的 **time-ordered history 补丁**。如果当前 mjlab 尚未应用该补丁，执行以下命令；首次执行会保留原文件备份：

```bash
python - <<'PYTHON'
from pathlib import Path
import shutil
import mjlab
source = Path('mjlab_patch/mjlab/managers/observation_manager.py')
target = Path(mjlab.__file__).resolve().parent / 'managers/observation_manager.py'
backup = target.with_suffix('.py.bak')
if not backup.exists():
    shutil.copy2(target, backup)
shutil.copy2(source, target)
print(target)
PYTHON
```

不要删除 `history_ordering="time"` 来绕过补丁，观测顺序必须与策略一致。AMP 训练和恢复重置需要以下动作目录，普通速度任务不需要这些数据：

```text
src/assets/motions/g1/amp/WalkandRun/
src/assets/motions/g1/amp/Recovery/
```

## 从零训练

选择一个任务运行。以下命令均训练到总计 14000 轮，每 1000 轮保存 checkpoint；环境数采用任务默认值：

```bash
python scripts/train.py Unitree-G1-AMP-Flat --agent.max-iterations=14000 --agent.run-name=wmrl_scratch --agent.save-interval=1000
python scripts/train.py Unitree-G1-AMP-Rough --agent.max-iterations=14000 --agent.run-name=wmrl_scratch --agent.save-interval=1000
python scripts/train.py Unitree-G1-Flat --agent.max-iterations=14000 --agent.run-name=wmrl_scratch --agent.save-interval=1000
python scripts/train.py Unitree-G1-Rough --agent.max-iterations=14000 --agent.run-name=wmrl_scratch --agent.save-interval=1000
```

只执行所需任务的那一行。省略 `--agent.max-iterations` 时默认总轮数为 **100001**。默认 seed 为 **42**，训练 episode 为 5 秒，间隔推扰关闭；Rough 保留地形课程。`--backend supermimic` 与默认的 `wmrl` 等价。

AMP Flat 的 14000 轮启动脚本也可使用：

```bash
bash supermimic/run_fast_14000.sh
```

该脚本默认使用本机 `/home/ubuntu/USE/Miniconda/envs/mjlab/bin/python`；其他环境请指定 `PYTHON_BIN`，例如 `PYTHON_BIN=python bash supermimic/run_fast_14000.sh`。`RUN_NAME` 控制运行名，额外 CLI 参数可追加到脚本后。

### 低资源运行检查

下面使用普通 Flat、16 个环境和小网络，运行 1 轮模型预训练及 1 轮 actor/critic 更新，用于检查安装和训练链路。它仍需要 CUDA，不用于学习可部署策略。可替换为其他三个任务 ID 检查对应路径；AMP 任务需要动作数据。

```bash
python scripts/train.py Unitree-G1-Flat \
  --env.scene.num-envs=16 \
  --agent.max-iterations=2 --agent.save-interval=1 --agent.run-name=smoke \
  --supermimic.wm-width=32 \
  --supermimic.batch-size=8 --supermimic.wm-batch-size=8 \
  --supermimic.actor-microbatch=8 --supermimic.wm-microbatch=8 \
  --supermimic.buffer-steps=16 --supermimic.warmup-steps=8 \
  --supermimic.horizon=2 --supermimic.wm-batches=1 \
  --supermimic.wm-pretrain-updates=1 \
  --supermimic.precision=fp32 --supermimic.compile=False --supermimic.log-every=1
```

### 默认学习参数与调节

| 项目 | 默认值 / 用法 |
|---|---|
| 世界模型 | 2048×3 隐藏层，8 步预测损失，每轮 10 次 WM 更新 |
| WM / actor 等效 batch | 2048 / 1024 |
| WM / actor microbatch | 2048 / 1024；缩小可减少训练显存，保留等效 batch |
| 想象长度 | `--supermimic.horizon=32` |
| 初始采集 / replay 容量 | 64 / 256 个向量步 |
| 模型预训练 | 500 轮，计入 `--agent.max-iterations`；正常训练总轮数须大于此值 |
| Critic | 想象 GAE 目标，5 个 epoch、4 个 minibatch |
| 初始 actor / WM / critic 学习率 | `2e-4`；余弦调度到 250000 轮时降至初值的 0.03 |
| AMP 判别器学习率 | `--supermimic.discriminator-lr=1e-5`，仅 AMP 任务使用 |
| 探索噪声标准差 | 0.1 |
| 采集比例 | `--supermimic.utd=0.4`，平均每轮更新采集 2.5 个向量步 |
| 精度 / 编译 | BF16 / 开启；仿真和几何计算仍为 FP32 |
| GPU / 日志 | GPU 0 / TensorBoard |

常用调整：`--env.scene.num-envs=256` 调整并行环境数；`--supermimic.wm-microbatch=512 --supermimic.actor-microbatch=256` 减少网络训练显存；`--supermimic.compile=False` 跳过首次编译；`--supermimic.precision=fp32` 改用 FP32。首次编译可能需要数分钟。环境数、网络宽度及 batch 都会影响显存，默认环境数并不保证适合所有 GPU。

Actor 和 WM 的运行统计来自训练分区的真实数据；critic 使用独立输入归一化，BPTT 期间固定这些统计。策略动作叠加探索噪声后直接执行。训练保留终止掩码、梯度裁剪及 AMP 梯度惩罚。`--agent.algorithm.*` 仅包含当前 WMRL 使用的字段，PPO 的 clip、KL 和 entropy 参数已移除。

查看完整参数：

```bash
python scripts/train.py Unitree-G1-Rough --help
```

## 日志与 checkpoint

各任务输出到：

```text
logs/rsl_rl/<上表实验目录名>/<时间戳>_<run_name>/
```

`rsl_rl` 是当前日志路径的兼容命名，不代表训练运行 PPO。启动时终端会打印实际输出目录。查看全部任务：

```bash
tensorboard --logdir logs/rsl_rl
```

| 文件 | 内容 |
|---|---|
| `model_<轮数>.pt` | 网络、归一化、优化器等学习状态，不包含 replay；用于播放或续训，AMP Flat 也可导出 |
| `last.pt` | 最近保存的完整学习状态；按保存间隔及训练结束时更新，续训后重新采集 replay |
| `actor_final.pt` | 最终 checkpoint，包含完整学习状态，并非仅 actor 权重 |
| `params/env.yaml`、`params/agent.yaml` | 通过 `scripts/train.py` 启动时保存的配置 |
| `config.json`、`training.json` | 训练设置与完成/失败摘要 |
| TensorBoard event 文件 | 训练和预训练指标 |

播放和导出的示例使用**你自己训练得到的 checkpoint**，不依赖已删除的历史模型。

## 续训

使用相同任务、环境数及原始环境/算法参数，并将总轮数设为大于已完成轮数。`--agent.load-run` 填写实验目录下的运行文件夹名，不是 checkpoint 完整路径。例如继续普通 Rough 的 14000 轮训练：

```bash
python scripts/train.py Unitree-G1-Rough \
  --agent.max-iterations=20000 \
  --agent.resume=True \
  --agent.load-run=YOUR_RUN_DIRECTORY \
  --agent.load-checkpoint=last.pt
```

若初次训练覆盖过环境数、网络宽度、batch、horizon 等参数，续训时也需提供相同设置；microbatch、精度和编译开关允许调整。使用其他 `--agent.experiment-name` 时也要保持一致，程序会在该实验目录查找运行。

续训要求从零训练生成的 `fast_schema=3` checkpoint，校验任务和环境配置，并恢复网络、模型头统计、优化器、探索进度及 RNG。它跳过已完成的预训练，启动新的仿真 episode，重新填充 replay；这不是仿真器状态的逐位延续。旧 schema 1/2 不支持训练恢复，旧策略的播放/导出还需满足对应任务的观测与权重结构。

## 播放

将 `YOUR_RUN_DIRECTORY` 替换为你训练后的运行文件夹名，checkpoint 的任务必须与命令中的任务 ID 匹配。例如普通 Rough：

```bash
python scripts/play.py Unitree-G1-Rough \
  --checkpoint-file logs/rsl_rl/g1_supermimic_velocity_rough/YOUR_RUN_DIRECTORY/model_14000.pt \
  --num-envs=1 --viewer=native
```

四个任务都可以播放，替换任务 ID 和对应实验目录即可。`--viewer=viser` 使用浏览器查看，`--viewer=auto` 自动选择查看器。播放录像使用 `--video=True`，训练入口不支持录像。

无需 checkpoint 检查环境：

```bash
python scripts/play.py Unitree-G1-Flat --agent=zero --num-envs=1 --viewer=viser
```

## 评估、ONNX 与 sim2sim（仅 AMP Flat）

训练结束后执行固定速度/恢复场景评估：

```bash
python scripts/train.py Unitree-G1-AMP-Flat \
  --agent.max-iterations=14000 --agent.run-name=wmrl_eval \
  --supermimic.evaluate=True --supermimic.seconds=20
```

评估使用相同 seed，写入运行目录的 `evaluation.json`；AMP 返回使用候选策略自身的判别器，不能作为跨 checkpoint 的统一评分。其他任务当前使用对应的播放入口检查表现。

安装 ONNX 依赖后，导出自己训练的 AMP Flat checkpoint：

```bash
python scripts/export_sim2sim.py \
  --checkpoint-file logs/rsl_rl/g1_supermimic_locomotion/YOUR_RUN_DIRECTORY/model_14000.pt \
  --output-dir deploy/wmrl

python scripts/sim2sim.py Unitree-G1-AMP-Flat \
  --policy deploy/wmrl/policy.onnx \
  --viewer native --duration 60 --command 0.5 0 0
```

导出包含 `policy.onnx`、`policy.json` 与 `scene.zip`；ONNX 内嵌归一化，导出器用 100 组输入核对 PyTorch/ONNX 数值一致性。无窗口运行使用 `--viewer none`，可加 `--output /tmp/sim2sim.json` 保存指标。现有导出器和 sim2sim 的 384 维 AMP Flat 输入协议不适用于其他三个任务。

## 机器人部署

AMP Flat 导出的部署包通过 Python ONNX 推理和原生 Unitree SDK2 C++ 通信运行。在机器人计算机上按 [deploy/README.md](deploy/README.md) 编译 SDK/DDS 扩展后执行：

```bash
python deploy/deploy_g1_cpp.py --net eth0 --policy deploy/wmrl/policy.onnx
```

操作顺序：START → 默认姿态 → A → 策略；B/SELECT 停止。保留 CRC、状态超时检查和 watchdog 阻尼保护。小规模训练测试和 sim2sim 不代表已验证机器人闭环效果。

## 代码与验证

| 路径 | 用途 |
|---|---|
| `scripts/train.py`、`supermimic/cli.py` | 推荐训练入口及四个任务的默认配置 |
| `supermimic/train_fast.py` | 共用 WMRL 训练循环 |
| `supermimic/agent_config.py` | WMRL agent 配置，不包含 PPO 专属参数 |
| `supermimic/core/task_model.py` | 新增任务的观测、奖励、终止模型头及想象更新 |
| `supermimic/core/` | 世界模型、replay、归一化、checkpoint 与解析 AMP Flat 路径 |
| `supermimic/mdp/`、`supermimic/adapters/` | 可微奖励、AMP、恢复与真实物理采集 |
| `src/tasks/amp_loco/`、`src/tasks/velocity/` | 四个任务的原生环境配置 |
| `supermimic/evaluation/` | AMP Flat 固定场景评估 |
| `deploy/` | ONNX 资产与机器人通信 |

无需 pytest，可运行迁移回归测试：

```bash
python supermimic/tests/test_task_model.py
```

当前验证包括 5 项回归测试、四个任务在 16 个环境与缩小网络上的采集/预训练/actor/critic 更新和策略重载、三个新增任务的续训，以及 AMP Rough 的 BF16 编译更新。此结果验证运行链路，不建立长期训练收敛或实机可用性结论。
