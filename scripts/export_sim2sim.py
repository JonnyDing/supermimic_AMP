"""Export WMRL actor + normalization and a portable MuJoCo sim2sim bundle."""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
import torch
import onnx
import onnxruntime as ort
from mjlab.entity import Entity
from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_env_cfg
from supermimic.core.policy import load_policy


class Policy(torch.nn.Module):
    def __init__(self, actor, normalizer):
        super().__init__()
        self.actor = actor
        self.normalizer = normalizer

    def forward(self, observation):
        return self.actor(self.normalizer(observation))


def export(checkpoint, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    actor, normalizer = load_policy(checkpoint, 'cpu')
    policy = Policy(actor.actor, normalizer).eval()
    path = output / 'policy.onnx'
    torch.onnx.export(policy, torch.zeros(1, 384), str(path), input_names=['observation'],
                      output_names=['action'], opset_version=17, dynamo=False)
    onnx.checker.check_model(str(path))
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    generator = torch.Generator().manual_seed(42)
    max_error = 0.
    with torch.inference_mode():
        for _ in range(100):
            obs = torch.randn(1, 384, generator=generator)
            expected = policy(obs).numpy()
            actual = session.run(['action'], {'observation': obs.numpy()})[0]
            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
            max_error = max(max_error, float(np.max(np.abs(actual - expected))))
    cfg = g1_amp_flat_env_cfg()
    robot_cfg = cfg.scene.entities['robot']
    robot = Entity(robot_cfg)
    spec = robot.spec
    spec.worldbody.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[100, 100, .1], friction=[1., .005, .0001], rgba=[.25, .3, .35, 1])
    model = spec.compile()
    cfg.sim.mujoco.apply(model)
    with (output / 'scene.zip').open('wb') as scene_file:
        spec.to_zip(scene_file)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    joints = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    names = [model.joint(j).name for j in joints]
    motors = [next(a for a in robot_cfg.articulation.actuators
                   if any(re.fullmatch(p, name) for p in a.target_names_expr)) for name in names]
    fields = ['stiffness', 'damping', 'effort_limit', 'X1', 'X2', 'Y1', 'Y2', 'Fs', 'Fd', 'Va']
    option_names = ['timestep', 'iterations', 'ls_iterations', 'integrator', 'solver', 'cone',
                    'ccd_iterations', 'impratio', 'tolerance', 'ls_tolerance', 'noslip_iterations',
                    'noslip_tolerance', 'disableflags', 'enableflags']
    options = {name: getattr(model.opt, name) for name in option_names}
    metadata = dict(schema_version=1, scene='scene.zip', joint_names=names,
                    default_q=data.qpos[model.jnt_qposadr[joints]].tolist(),
                    action_scale=[next(v for p, v in cfg.actions['joint_pos'].scale.items()
                                       if re.fullmatch(p, n)) for n in names],
                    motors=[{f: getattr(a, f) for f in fields} for a in motors],
                    decimation=cfg.decimation, physics_options=options,
                    source_checkpoint=str(Path(checkpoint).resolve()), normalization_in_onnx=True,
                    onnx_max_abs_error=max_error, mujoco_version=mujoco.__version__)
    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Exported {path}; 100-input ONNX parity max abs error: {max_error:.8g}')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-file', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('deploy/wmrl'))
    args = parser.parse_args()
    export(args.checkpoint_file, args.output_dir)


if __name__ == '__main__':
    main()
