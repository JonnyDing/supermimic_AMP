# SuperMimic — World-model reinforcement learning for Unitree G1

[中文说明](README_zh.md)

Built on mjlab, this project trains G1 velocity tracking and fall recovery with SuperMimic/WMRL: real simulation collection → world-model fitting → imagined BPTT actor updates → critic fitting with GAE targets. AMP tasks additionally train a motion-style discriminator; ordinary velocity tasks do not use AMP.

Training uses no PPO optimizer, probability-ratio clipping, KL schedule or teacher. `rsl_rl` remains a dependency for networks, normalization and mjlab import compatibility. New runs initialize networks from scratch and require no pretrained checkpoint.

## Supported tasks

| Task ID | Terrain | AMP | Default environments | Experiment directory |
|---|---|---|---:|---|
| `Unitree-G1-AMP-Flat` | Plane | Enabled | 4090 | `g1_supermimic_locomotion` |
| `Unitree-G1-AMP-Rough` | Generated rough terrain | Enabled | 512 | `g1_supermimic_amp_rough` |
| `Unitree-G1-Flat` | Plane | Disabled | 4090 | `g1_supermimic_velocity_flat` |
| `Unitree-G1-Rough` | Generated rough terrain | Disabled | 512 | `g1_supermimic_velocity_rough` |

All four actors use four time-ordered observation frames and produce 29 joint actions at a default control rate of 50 Hz. AMP Flat has 384 inputs; network and normalization sizes for other tasks are derived from their observations.

- **AMP Flat:** imagination uses the existing analytic observation, reward and delayed-recovery logic. Observation noise follows the task configuration.
- **The other three tasks:** retain their native environment rewards and termination conditions. Real transitions supervise observation, reward, termination and recovery-counter heads; terminal/truncated boundaries do not supply observation targets. During imagination, learned heads predict subsequent observations, while commands, executed actions and velocity-task gait phase follow explicit update rules.
- **Rough tasks:** retain terrain scans and terrain curriculum. The observation head predicts subsequent scans during imagination.
- **Ordinary velocity tasks:** retain velocity and gait rewards, load no AMP motion experts, and train no discriminator.

Training supports one CUDA GPU and TensorBoard. Fixed recovery evaluation, the current ONNX sim2sim bundle and robot deployment export support **AMP Flat only**. All four tasks support mjlab playback with a matching checkpoint. Small training checks have passed; long-run convergence of the new tasks has not been established.

## Environment

Use Linux, Python 3.11, a working NVIDIA driver and CUDA-capable PyTorch. Run the following commands from the repository root in an existing `mjlab` environment. `setup.py` pins mjlab 1.2.0, mujoco-warp 3.8.1, warp-lang 1.12.0 and SciPy 1.17.0.

```bash
conda activate mjlab
python -m pip install -e .
python -m pip install tensorboard
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

For ONNX export or sim2sim, also install:

```bash
python -m pip install onnx onnxruntime
```

Keep the repository root on `PYTHONPATH` to load local tasks and compatibility modules. Four-frame observations require the supplied **time-ordered history patch**. If your mjlab installation does not already include it, apply it below; the original file is backed up on the first invocation:

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

Do not remove `history_ordering="time"` to bypass the patch: observation ordering must match the policy. AMP training and recovery resets require the motion directories below; ordinary velocity tasks do not require them:

```text
src/assets/motions/g1/amp/WalkandRun/
src/assets/motions/g1/amp/Recovery/
```

## Train from scratch

Choose one task. Each command trains to 14000 total iterations and saves every 1000 iterations, using the task's default environment count:

```bash
python scripts/train.py Unitree-G1-AMP-Flat --agent.max-iterations=14000 --agent.run-name=wmrl_scratch --agent.save-interval=1000
python scripts/train.py Unitree-G1-AMP-Rough --agent.max-iterations=14000 --agent.run-name=wmrl_scratch --agent.save-interval=1000
python scripts/train.py Unitree-G1-Flat --agent.max-iterations=14000 --agent.run-name=wmrl_scratch --agent.save-interval=1000
python scripts/train.py Unitree-G1-Rough --agent.max-iterations=14000 --agent.run-name=wmrl_scratch --agent.save-interval=1000
```

Run only the line for your chosen task. Without `--agent.max-iterations`, the default total is **100001**. The default seed is **42**, training episodes last 5 seconds, interval pushes are disabled, and Rough retains terrain curriculum. `--backend supermimic` aliases the default `wmrl` backend.

The AMP Flat 14000-iteration launcher is also available:

```bash
bash supermimic/run_fast_14000.sh
```

It defaults to this machine's `/home/ubuntu/USE/Miniconda/envs/mjlab/bin/python`. For another environment, set `PYTHON_BIN`, for example `PYTHON_BIN=python bash supermimic/run_fast_14000.sh`. `RUN_NAME` sets the run name; extra CLI arguments can follow the script path.

### Small installation check

This ordinary Flat run uses 16 environments and a small network for one model-pretraining iteration and one actor/critic iteration. It still requires CUDA and is not intended to learn a deployable policy. Replace the task ID to check any other task; AMP tasks require motion data.

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

### Learning defaults and tuning

| Setting | Default / usage |
|---|---|
| World model | Three 2048-wide hidden layers, eight-step prediction loss, 10 WM updates per iteration |
| WM / actor effective batch | 2048 / 1024 |
| WM / actor microbatch | 2048 / 1024; smaller values reduce training memory while retaining effective batches |
| Imagination horizon | `--supermimic.horizon=32` |
| Initial collection / replay capacity | 64 / 256 vector steps |
| Model pretraining | 500 iterations, included in `--agent.max-iterations`; normal training must exceed this count |
| Critic | Imagined GAE targets, five epochs and four minibatches |
| Initial actor / WM / critic learning rate | `2e-4`; cosine schedule reaches 0.03 of the initial rate at 250000 iterations |
| AMP discriminator learning rate | `--supermimic.discriminator-lr=1e-5`, AMP tasks only |
| Exploration standard deviation | 0.1 |
| Collection ratio | `--supermimic.utd=0.4`, averaging 2.5 vector steps collected per update |
| Precision / compilation | BF16 / enabled; simulation and geometry remain FP32 |
| GPU / logger | GPU 0 / TensorBoard |

Common overrides: `--env.scene.num-envs=256` changes parallelism; `--supermimic.wm-microbatch=512 --supermimic.actor-microbatch=256` reduces network-training memory; `--supermimic.compile=False` skips initial compilation; `--supermimic.precision=fp32` selects FP32. Initial compilation can take several minutes. Environment count, network width and batches all affect memory; the default environment count is not a guarantee of fitting every GPU.

Actor and WM running statistics use real data from the training partition. The critic has independent input normalization; these statistics stay fixed during BPTT. Policy actions execute directly with exploration noise. Termination masks, gradient clipping and AMP gradient penalties remain. `--agent.algorithm.*` exposes fields consumed by WMRL; PPO clip, KL and entropy options are absent.

Inspect all options:

```bash
python scripts/train.py Unitree-G1-Rough --help
```

## Logs and checkpoints

Each task writes to:

```text
logs/rsl_rl/<experiment directory from the table>/<timestamp>_<run_name>/
```

The `rsl_rl` directory name is retained for path compatibility; it does not select PPO. The terminal prints the actual output directory. View all task runs with:

```bash
tensorboard --logdir logs/rsl_rl
```

| File | Contents |
|---|---|
| `model_<iteration>.pt` | Networks, normalization, optimizers and learning state, without replay; playback/resume for all tasks, export for AMP Flat |
| `last.pt` | Latest saved learning state, updated at checkpoint intervals and training completion; replay is recollected on resume |
| `actor_final.pt` | Final checkpoint with full learning state, not actor-only weights |
| `params/env.yaml`, `params/agent.yaml` | Configuration saved when launched through `scripts/train.py` |
| `config.json`, `training.json` | Training settings and completion/failure summary |
| TensorBoard event files | Training and pretraining metrics |

Playback and export examples use **your own trained checkpoint** and do not depend on removed historical models.

## Resume

Use the same task, environment count and original environment/algorithm settings, with a total iteration count greater than the completed count. `--agent.load-run` is the run folder name under the experiment directory, not a checkpoint's full path. For example, continue an ordinary Rough run trained to iteration 14000:

```bash
python scripts/train.py Unitree-G1-Rough \
  --agent.max-iterations=20000 \
  --agent.resume=True \
  --agent.load-run=YOUR_RUN_DIRECTORY \
  --agent.load-checkpoint=last.pt
```

Repeat any original overrides for environment count, model width, batch size, horizon and other learning settings. Microbatch sizes, precision and compilation can change. Keep any custom `--agent.experiment-name` so the runner searches the same experiment directory.

Resume requires a scratch-training checkpoint with `fast_schema=3`, validates task and environment configuration, and restores networks, task-head statistics, optimizers, exploration progress and RNG. It skips completed pretraining, starts fresh simulator episodes and refills replay; this is not bitwise simulator continuation. Schema 1/2 files cannot resume training. Legacy-policy playback/export also requires compatible task observations and weight structure.

## Playback

Replace `YOUR_RUN_DIRECTORY` with your trained run folder. The checkpoint must match the requested task ID. For example, ordinary Rough:

```bash
python scripts/play.py Unitree-G1-Rough \
  --checkpoint-file logs/rsl_rl/g1_supermimic_velocity_rough/YOUR_RUN_DIRECTORY/model_14000.pt \
  --num-envs=1 --viewer=native
```

All four tasks support playback: substitute the task ID and matching experiment directory. Use `--viewer=viser` for browser viewing or `--viewer=auto` for automatic selection. `--video=True` records playback; the training entrypoint does not support recording.

Inspect an environment without a checkpoint:

```bash
python scripts/play.py Unitree-G1-Flat --agent=zero --num-envs=1 --viewer=viser
```

## Evaluation, ONNX and sim2sim (AMP Flat only)

Run fixed velocity/recovery scenarios after training:

```bash
python scripts/train.py Unitree-G1-AMP-Flat \
  --agent.max-iterations=14000 --agent.run-name=wmrl_eval \
  --supermimic.evaluate=True --supermimic.seconds=20
```

Evaluation uses the same seed and writes `evaluation.json` to the run directory. AMP returns use the candidate's own discriminator and are not a shared score across checkpoints. Use matching-task playback to inspect other tasks.

After installing ONNX dependencies, export your trained AMP Flat checkpoint:

```bash
python scripts/export_sim2sim.py \
  --checkpoint-file logs/rsl_rl/g1_supermimic_locomotion/YOUR_RUN_DIRECTORY/model_14000.pt \
  --output-dir deploy/wmrl

python scripts/sim2sim.py Unitree-G1-AMP-Flat \
  --policy deploy/wmrl/policy.onnx \
  --viewer native --duration 60 --command 0.5 0 0
```

The bundle contains `policy.onnx`, `policy.json` and `scene.zip`. Normalization is embedded in ONNX; export checks PyTorch/ONNX parity on 100 inputs. For headless execution, use `--viewer none`; optionally add `--output /tmp/sim2sim.json` to save measurements. The current exporter and sim2sim use the 384-input AMP Flat contract and do not support the other three tasks.

## Robot deployment

The AMP Flat deployment bundle uses Python ONNX inference and native Unitree SDK2 C++ transport. On the robot computer, build the SDK/DDS extension as described in [deploy/README.md](deploy/README.md), then run:

```bash
python deploy/deploy_g1_cpp.py --net eth0 --policy deploy/wmrl/policy.onnx
```

Operator sequence: START → nominal pose → A → policy; B/SELECT stops. CRC, stale-state checks and watchdog damping remain. Small training checks and sim2sim do not establish real-robot closed-loop performance.

## Code and validation

| Path | Responsibility |
|---|---|
| `scripts/train.py`, `supermimic/cli.py` | Recommended training entrypoint and defaults for all four tasks |
| `supermimic/train_fast.py` | Shared WMRL training loop |
| `supermimic/agent_config.py` | WMRL agent configuration without PPO-specific options |
| `supermimic/core/task_model.py` | Observation, reward and termination heads and imagination for the new tasks |
| `supermimic/core/` | Dynamics, replay, normalization, checkpoints and analytic AMP Flat path |
| `supermimic/mdp/`, `supermimic/adapters/` | Differentiable rewards, AMP, recovery and real physics collection |
| `src/tasks/amp_loco/`, `src/tasks/velocity/` | Native environment configurations for the four tasks |
| `supermimic/evaluation/` | AMP Flat fixed-scenario evaluation |
| `deploy/` | ONNX assets and robot transport |

Run migration regression tests without pytest:

```bash
python supermimic/tests/test_task_model.py
```

Validation covers five regression tests; collection, pretraining, actor/critic updates and policy reload for all four tasks with 16 environments and reduced networks; resume for the three new tasks; and an AMP Rough BF16 compiled update. These checks establish execution paths, not long-run convergence or deployment readiness.
