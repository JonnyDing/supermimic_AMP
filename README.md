# SuperMimic AMP — WMRL for Unitree G1

[中文说明](README_zh.md)

World-model reinforcement learning for G1 velocity tracking and fall recovery, built on mjlab. The active training backend is WMRL: real simulation collection, learned dynamics, imagined rollouts, and actor updates through backpropagation through time (BPTT). AMP supplies a differentiable motion-style reward.

## Current implementation

- Real transitions populate a sequence replay buffer and train the world model.
- The actor optimizes imagined task and AMP rewards with critic bootstrapping and no teacher anchor.
- The discriminator trains separately on real policy/expert transitions with a least-squares objective and gradient penalty. During imagination its parameters are frozen while input gradients remain available.
- One policy produces 29 joint actions from four observation frames (384 inputs), at 50 Hz.
- Training currently supports `Unitree-G1-AMP-Flat`, one CUDA GPU, and TensorBoard. Default settings use BF16 network math, compilation, and an imagination horizon of 32.

PPO optimizers and training runners have been removed. The remaining `rsl_rl` code provides compatible networks and normalization; its `OnPolicyRunner` placeholder exists only for mjlab imports. New runs initialize all networks from scratch. No pretrained policy, world model, discriminator or teacher checkpoint is loaded.

## Environment

Use Linux, Python 3.11, a CUDA-capable PyTorch installation, and a working GPU driver. Run commands from the repository root in an existing mjlab environment. `setup.py` pins mjlab 1.2.0, mujoco-warp 3.8.1, warp-lang 1.12.0, and SciPy 1.17.0.

```bash
conda activate mjlab
python -m pip install -e .
python -m pip install onnx onnxruntime tensorboard
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

The repository root on `PYTHONPATH` selects the local compatibility modules. The G1 observation configuration requires the supplied **time-ordered observation history patch**. If your mjlab installation does not already include it, back up and replace the manager in the active environment:

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

Do not remove `history_ordering="time"` to bypass the patch: observation ordering must match the checkpoint.

Motion assets are expected under:

- `src/assets/motions/g1/amp/WalkandRun/`
- `src/assets/motions/g1/amp/Recovery/`

## Checkpoint and training prerequisites

The committed inference/export checkpoint is:

```text
logs/rsl_rl/g1_supermimic_locomotion/2026-10-27_11-13-57_wmrl_amp/model_14000.pt
```

This historical checkpoint remains usable for playback/export. New training does not depend on it or any other file under `logs/`. Motion reference data is still required for AMP and recovery resets.

## Train from scratch (4090 environments)

Training uses one seed, **42** by default; optional evaluation uses the same seed. The default CLI configuration uses **4090 parallel environments** and the reference learning flow:

- World model: three 2048-wide hidden layers, eight-step prediction loss, 10 optimizer updates of 2048 windows per iteration.
- Actor: 32-step BPTT, 1024 imagination starts per update. Critic: imagined GAE targets, five epochs and four minibatches.
- Effective batches use gradient accumulation: WM microbatch 2048 and actor microbatch 1024, both full batches by default. Network matmuls default to BF16; simulation and geometry remain FP32. Parameters update once per effective batch.
- Actor observation and WM input/output statistics update from real transitions in the training partition. The critic has independent running input normalization. Statistics stay fixed during BPTT.
- Every imagined observation frame uses the real task's observation-noise configuration. ONNX inference remains deterministic.
- Initial replay: 64 steps from the random initial policy plus exploration noise; 500 model-only iterations of 10 WM updates each count toward the total iteration budget, followed by joint actor/critic updates without separate critic warmup. No pretrained policy or teacher is loaded.
- Actor, WM and critic learning rates follow the reference cosine schedule: initial `2e-4`, final ratio 0.03 after 250000 total iterations. Exploration standard deviation is 0.1.

Replay retains 256 steps for this GPU configuration; the tracking reference uses 1000. Tracking-specific goal encoders are not part of this velocity/AMP task. These settings restore the learning mechanisms but do not establish convergence on the modified task.

```bash
python scripts/train.py Unitree-G1-AMP-Flat \
  --env.scene.num-envs=4090 \
  --agent.max-iterations=14000 \
  --agent.run-name=wmrl_scratch \
  --agent.save-interval=1000
```

BF16 and full 1024-sample actor updates are now the defaults for both scratch training and resume. No precision or microbatch options are required. Older checkpoints' TF32/1024-WM/256-actor runtime settings use the new defaults, while their networks, optimizers and other learning settings are restored.

Training uses policy actions plus exploration noise directly, with no action magnitude threshold, numerical reset retries, zero substitutions, imagination trajectory rejection or runtime parameter audits. Episode termination masks, the reference gradient clipping and AMP gradient penalty remain part of the learning algorithm.


Logs are written to `logs/rsl_rl/g1_supermimic_locomotion/<timestamp>_<run_name>/`. View them with:

```bash
tensorboard --logdir logs/rsl_rl/g1_supermimic_locomotion
```

Use `--supermimic.wm-microbatch` and `--supermimic.actor-microbatch` to adjust memory while retaining the effective batches. `--supermimic.utd=0.4` collects 2.5 vector steps per update on average. GPU 0 is selected by default; an explicit list uses `--gpu-ids '[0]'`. Use `--help` for options. `--agent.*` retains the existing CLI schema; WMRL options use `--supermimic.*`. Unsupported legacy algorithm overrides raise an error. `--backend supermimic` is an alias for the default `wmrl` backend.

| Artifact | Purpose |
|---|---|
| `model_<update>.pt` | Networks, normalization and learning state, without replay; usable for playback/export |
| `actor_final.pt` | Final checkpoint with the same payload structure as the final numbered checkpoint; not actor-only |
| `last.pt` | Latest networks, optimizers and statistics; replay is recollected on resume |

To resume, use a new scratch run’s `last.pt` (`fast_schema=3`) and retain the original environment/algorithm settings and specify a larger total update count:

```bash
python scripts/train.py Unitree-G1-AMP-Flat \
  --env.scene.num-envs=4090 \
  --agent.max-iterations=20000 \
  --agent.resume=True \
  --agent.load-run=YOUR_RUN_DIRECTORY \
  --agent.load-checkpoint=last.pt
```

Resume restores learning state, normalization, exploration progress and RNG, skips completed model warmup, then starts fresh simulator episodes and recollects at least 64 steps of replay. Older schema 3 files with replay remain loadable. It is not bitwise simulator continuation. Schema 1/2 training files are rejected: their frozen statistics, critic coordinates and architecture do not describe the corrected learning state. Their actor policies remain loadable for playback/export. Scratch convergence must be assessed with a fresh run using the corrected failure sampling.

## Playback, export and sim2sim

```bash
python scripts/play.py Unitree-G1-AMP-Flat \
  --checkpoint-file logs/rsl_rl/g1_supermimic_locomotion/2026-10-27_11-13-57_wmrl_amp/model_14000.pt \
  --num-envs 1 --viewer native
```

Playback only previews the policy. Export the portable ONNX bundle separately:

```bash
python scripts/export_sim2sim.py \
  --checkpoint-file logs/rsl_rl/g1_supermimic_locomotion/2026-10-27_11-13-57_wmrl_amp/model_14000.pt \
  --output-dir deploy/wmrl

python scripts/sim2sim.py Unitree-G1-AMP-Flat \
  --policy deploy/wmrl/policy.onnx \
  --viewer native --duration 60 --command 0.5 0 0
```

The bundle contains `policy.onnx`, `policy.json`, and `scene.zip`. Normalization is embedded in ONNX. Export checks numerical agreement on 100 inputs. For headless sim2sim, use `--viewer none`; add `--output /tmp/sim2sim.json` to save rollout measurements.

## Robot deployment

The board entrypoint uses Python ONNX inference and native Unitree SDK2 C++ transport. Build the extension against the board SDK and matching DDS libraries as described in [deploy/README.md](deploy/README.md), then run on the robot computer:

```bash
python deploy/deploy_g1_cpp.py --net eth0 --policy deploy/wmrl/policy.onnx
```

Operator sequence: START → nominal pose → A → policy; B/SELECT stops. Runtime CRC, stale-state checks and watchdog damping remain. Standalone deployment verification scripts have been removed. Software tests and sim2sim do not establish real-robot closed-loop performance.

## Code and validation

| Path | Responsibility |
|---|---|
| `scripts/train.py`, `supermimic/train_fast.py` | Supported training CLI and WMRL training loop |
| `supermimic/core/` | Dynamics, replay, imagination, policy loading and checkpoint handling |
| `supermimic/mdp/`, `supermimic/adapters/` | Rewards, AMP, recovery and simulation adapter |
| `supermimic/evaluation/` | Fixed-scenario evaluation and shared metrics |
| `src/tasks/amp_loco/` | G1 task and environment configuration |
| `deploy/` | ONNX assets and native robot transport |

Use `--supermimic.evaluate=True` for fixed-scenario evaluation after training. AMP returns use the candidate discriminator and are not comparable across checkpoints. Unit tests, temporary benchmark tools and historical diagnostic reports have been removed; the training, playback, export and fixed-scenario evaluation entry points remain available.
