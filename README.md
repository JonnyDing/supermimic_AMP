# SuperMimic AMP — WMRL for Unitree G1

[中文说明](README_zh.md)

World-model reinforcement learning for G1 velocity tracking and fall recovery, built on mjlab. The active training backend is WMRL: real simulation collection, learned dynamics, imagined rollouts, and actor updates through backpropagation through time (BPTT). AMP supplies a differentiable motion-style reward.

## Current implementation

- Real transitions populate a sequence replay buffer and train the world model.
- The actor optimizes imagined task and AMP rewards with critic bootstrapping and a frozen teacher anchor.
- The discriminator trains separately on real policy/expert transitions with a least-squares objective and gradient penalty. During imagination its parameters are frozen while input gradients remain available.
- One policy produces 29 joint actions from four observation frames (384 inputs), at 50 Hz.
- Training currently supports `Unitree-G1-AMP-Flat`, one CUDA GPU, and TensorBoard. Default settings use FP32, compilation, and an imagination horizon of 8.

PPO optimizers and training runners have been removed. The remaining `rsl_rl` code provides compatible networks and normalization; its `OnPolicyRunner` placeholder exists only for mjlab imports. This implementation uses existing policies for initialization and teacher anchoring, rather than training from scratch.

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

The current trainer also expects these initialization assets at fixed paths:

- `logs/rsl_rl/g1_amp_locomotion/2026-10-24_13-24-33/model_66600.pt`
- `logs/supermimic/p3_online_conservative_20261026/actor_final.pt`

The first supplies the teacher and original AMP state; the second supplies the initial policy/world-model learning state. These historical assets are not supplied by the single published checkpoint and must be restored before starting a new training run. Resume also checks its recorded source files. `model_14000.pt` alone supports policy loading and export, not the complete training setup.

## Train

After restoring the required assets:

```bash
python scripts/train.py Unitree-G1-AMP-Flat \
  --gpu-ids 0 \
  --env.scene.num-envs=512 \
  --agent.max-iterations=14000 \
  --agent.seed=2031 \
  --agent.run-name=wmrl_amp \
  --agent.save-interval=1000
```

Logs are written to `logs/rsl_rl/g1_supermimic_locomotion/<timestamp>_<run_name>/`. View them with:

```bash
tensorboard --logdir logs/rsl_rl/g1_supermimic_locomotion
```

Use `--help` for options. `--agent.*` retains the existing CLI schema; WMRL options use `--supermimic.*`. Unsupported legacy algorithm overrides raise an error. `--backend supermimic` is an alias for the default `wmrl` backend.

| Artifact | Purpose |
|---|---|
| `model_<update>.pt` | Networks, normalization and learning state, without replay; usable for playback/export |
| `actor_final.pt` | Final checkpoint with the same payload structure as the final numbered checkpoint; not actor-only |
| `last.pt` | Full trainer resume checkpoint, including replay |

To resume, retain the original environment/algorithm settings and specify a larger total update count:

```bash
python scripts/train.py Unitree-G1-AMP-Flat \
  --env.scene.num-envs=512 --agent.seed=2031 \
  --agent.max-iterations=20000 \
  --agent.resume=True \
  --agent.load-run=YOUR_RUN_DIRECTORY \
  --agent.load-checkpoint=last.pt
```

Resume restores learning state, replay and RNG, then starts fresh simulator episodes; it is not bitwise simulator continuation.

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

```bash
python -m unittest discover -s supermimic/tests -v
```

The cleanup passed 26 tests, including ONNX parity, plus loading/inference of the published checkpoint. Native C++ rebuild was not verified in the cleanup environment because pybind11 was unavailable. Earlier stage trainers and reports remain for historical experiments; [supermimic/README.md](supermimic/README.md) documents that history, not an additional supported training backend.
