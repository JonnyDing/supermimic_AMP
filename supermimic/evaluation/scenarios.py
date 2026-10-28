"""Fixed-scenario, headless PPO evaluation. No training, renderer or logger uploads."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

from supermimic.adapters.mjlab import CaptureEnv, PhysicsCollector
from supermimic.evaluation.baseline import sha256
from supermimic.mdp.kinematics import RobotKinematics

from supermimic.core.policy import ROOT, load_policy
COMMANDS = [[0., 0., 0.], [0.5, 0., 0.], [1.5, 0., 0.], [0., 0.5, 0.], [0., 0., 0.75]]



def fixed_recovery_frames():
    path = ROOT / 'src/assets/motions/g1/amp/Recovery/fallAndGetUp1_subject1.npz'
    data = dict(np.load(path))
    quat = data['body_quat_w'][:, 0]
    gravity_z_abs_upright = 1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    fallen = (data['body_pos_w'][:, 0, 2] < 0.5) | (gravity_z_abs_upright < math.cos(math.radians(70)))
    candidates = np.flatnonzero(fallen)
    if len(candidates) < 8:
        raise ValueError('Need at least eight fallen reference frames')
    frames = candidates[np.linspace(0, len(candidates)-1, 8, dtype=int)].tolist()
    return path, data, frames


def initialize_scenarios(env, data, frames):
    robot = env.scene['robot']
    device = env.device
    nv = len(COMMANDS)
    ids = torch.arange(env.num_envs, device=device)
    pose = robot.data.default_root_state[:, :7].clone()
    pose[:, :3] += env.scene.env_origins
    velocity = torch.zeros(env.num_envs, 6, device=device)
    q, qd = robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel)
    to_tensor = lambda a: torch.as_tensor(a, device=device, dtype=torch.float32)
    pose[nv:, 2] = to_tensor(data['body_pos_w'][frames, 0, 2]) + env.scene.env_origins[nv:, 2]
    pose[nv:, 3:] = to_tensor(data['body_quat_w'][frames, 0])
    velocity[nv:, :3] = to_tensor(data['body_lin_vel_w'][frames, 0])
    velocity[nv:, 3:] = to_tensor(data['body_ang_vel_w'][frames, 0])
    limits = robot.data.soft_joint_pos_limits
    q[nv:] = to_tensor(data['joint_pos'][frames]).clamp(limits[nv:, :, 0], limits[nv:, :, 1])
    qd[nv:] = to_tensor(data['joint_vel'][frames])
    robot.write_root_link_pose_to_sim(pose, env_ids=ids)
    robot.write_root_link_velocity_to_sim(velocity, env_ids=ids)
    robot.write_joint_state_to_sim(q, qd, env_ids=ids)
    env.episode_length_buf.zero_()
    env.action_manager.reset(ids)
    env.termination_manager._delay_counters.zero_()
    env.termination_manager._delay_env_mask[:] = torch.arange(env.num_envs, device=device) >= nv
    command = env.command_manager.get_term('twist')
    command.command.zero_()
    command.command[:nv] = to_tensor(COMMANDS)
    command.time_left.fill_(1e9)
    command.is_heading_env.zero_(); command.is_standing_env.zero_()
    env.scene.write_data_to_sim()
    env.sim.forward(); env.sim.sense()
    env.observation_manager.reset(ids)
    env.obs_buf = env.observation_manager.compute(update_history=True)


@torch.no_grad()
def run_seed(seed, policy, norm, data, frames, device):
    from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_env_cfg
    cfg = g1_amp_flat_env_cfg()
    cfg.seed = seed
    cfg.scene.num_envs = len(COMMANDS) + len(frames)
    cfg.events.pop('push_robot', None)
    cfg.curriculum = {}
    cfg.observations['actor'].enable_corruption = False
    cfg.episode_length_s = 20.0
    env = CaptureEnv(cfg=cfg, device=device)
    try:
        env.reset(seed=seed)
        initialize_scenarios(env, data, frames)
        collector, kin = PhysicsCollector(env), RobotKinematics(env)
        torso = list(env.scene['robot'].body_names).index('torso_link')
        n, nv, dt = env.num_envs, len(COMMANDS), env.step_dt
        active = torch.ones(n, dtype=torch.bool, device=device)
        steps = torch.zeros(n, device=device)
        xy_error = torch.zeros(n, device=device); yaw_error = torch.zeros_like(xy_error)
        action_cost = torch.zeros_like(xy_error); slip_sum = torch.zeros_like(xy_error)
        contact_count = torch.zeros_like(xy_error)
        consecutive = torch.zeros(n, device=device, dtype=torch.long)
        recovery_time = torch.full((n,), float('nan'), device=device)
        failure = torch.zeros(n, dtype=torch.bool, device=device)
        control_steps = round(20.0 / dt)
        start = time.perf_counter()
        for step in range(control_steps):
            actions = policy.act_inference(norm(env.obs_buf['actor']))
            if not torch.isfinite(actions).all():
                raise RuntimeError('Policy produced non-finite actions')
            tr = collector.step(actions)
            _, rotations, linear, angular = kin(tr.next_state)
            local_velocity = quat_apply_inverse(yaw_quat(rotations[:, torso]), linear[:, torso])
            xy_error += (local_velocity[:, :2] - tr.task['command'][:, :2]).square().sum(-1) * active
            yaw_error += (angular[:, torso, 2] - tr.task['command'][:, 2]).square() * active
            action_cost += (actions - tr.task['previous_action']).square().sum(-1) * active
            speed = tr.labels['feet_velocity_xy'].norm(dim=-1)
            contact = tr.labels['feet_in_contact']
            slip_sum += (speed * contact).sum(-1) * active
            contact_count += contact.sum(-1) * active
            steps += active
            gravity = torch.zeros(n, 3, device=device); gravity[:, 2] = -1
            gravity_b = quat_apply_inverse(tr.next_state.root_quat_wxyz, gravity)
            upright = (tr.next_state.root_pos[:, 2] >= 0.5) & (-gravity_b[:, 2] >= math.cos(math.radians(70)))
            consecutive = torch.where(active & upright, consecutive + 1, torch.zeros_like(consecutive))
            reached = (consecutive >= round(1.0 / dt)) & recovery_time.isnan() & active & ~tr.terminated
            recovery_time[reached] = (step + 1) * dt
            failure |= active & tr.terminated
            active &= ~(tr.terminated | tr.truncated)
            if step + 1 >= round(5.0 / dt):
                active[nv:] = False
            if step % 250 == 0:
                print(f'seed={seed} step={step}/{control_steps}', flush=True)
        torch.cuda.synchronize() if str(device).startswith('cuda') else None
        elapsed = time.perf_counter() - start
        rows = []
        for i in range(n):
            count = max(float(steps[i]), 1)
            row = {'seed': seed, 'kind': 'velocity' if i < nv else 'recovery',
                   'command': COMMANDS[i] if i < nv else [0, 0, 0],
                   'frame': None if i < nv else frames[i-nv],
                   'observed_seconds': float(steps[i]) * dt, 'terminated': bool(failure[i]),
                   'velocity_xy_rmse_m_s': math.sqrt(float(xy_error[i]) / count),
                   'yaw_rate_rmse_rad_s': math.sqrt(float(yaw_error[i]) / count),
                   'action_delta_squared_sum_mean': float(action_cost[i]) / count,
                   'contact_foot_speed_mean_m_s': float(slip_sum[i]) / max(float(contact_count[i]), 1)}
            if i >= nv:
                row['recovery_success'] = bool(torch.isfinite(recovery_time[i]))
                row['recovery_time_s'] = float(recovery_time[i]) if row['recovery_success'] else None
            else:
                row['survived_full_horizon'] = not row['terminated'] and float(steps[i]) >= control_steps
            rows.append(row)
        return rows, elapsed
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'supermimic/reports/baseline_manifest.json').read_text())
    checkpoint = ROOT / manifest['checkpoint']['path']
    assert sha256(checkpoint) == manifest['checkpoint']['sha256'], 'Frozen checkpoint changed'
    path, data, frames = fixed_recovery_frames()
    protocol = {'version': 1, 'seeds': args.seeds, 'commands': COMMANDS,
                'velocity_duration_s': 20, 'recovery_duration_s': 5,
                'recovery_hold_s': 1, 'minimum_height_m': 0.5, 'maximum_tilt_degrees': 70,
                'recovery_frames': frames, 'motion_path': str(path.relative_to(ROOT)),
                'motion_sha256': sha256(path), 'checkpoint': manifest['checkpoint'],
                'actor_noise': False, 'pushes': False, 'startup_domain_randomization': True,
                'initial_velocity_state': 'default root pose and joint pose, zero velocities',
                'recovery_state': 'NPZ pose and velocity, xy relocated to environment origin, joints clamped to soft limits',
                'metric_frame': 'torso heading XY linear velocity; world torso yaw rate',
                'error_aggregation': 'while active including terminal transition; report survival alongside RMSE',
                'action_clipping': None, 'activation': 'elu', 'normalizer_epsilon': 0.01,
                'termination': 'velocity: immediate; recovery: original 250-step delay; first episode only'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Write frozen scenarios BEFORE evaluating, including exact recovery frame IDs.
    args.output.with_suffix('.protocol.json').write_text(json.dumps(protocol, indent=2))
    policy, norm = load_policy(checkpoint, args.device)
    rows, durations = [], []
    for seed in args.seeds:
        seed_rows, elapsed = run_seed(seed, policy, norm, data, frames, args.device)
        rows.extend(seed_rows); durations.append(elapsed)
        args.output.write_text(json.dumps({'status': 'running', 'protocol': protocol, 'episodes': rows}, indent=2))
    velocities = [r for r in rows if r['kind'] == 'velocity']
    recoveries = [r for r in rows if r['kind'] == 'recovery']
    mean = lambda rs, key: sum(r[key] for r in rs) / len(rs)
    successes = [r for r in recoveries if r['recovery_success']]
    report = {'status': 'completed', 'protocol': protocol, 'episodes': rows,
              'summary': {'velocity_episodes': len(velocities), 'recovery_episodes': len(recoveries),
                          'velocity_survival_rate': mean(velocities, 'survived_full_horizon'),
                          'velocity_xy_rmse_episode_mean_m_s': mean(velocities, 'velocity_xy_rmse_m_s'),
                          'yaw_rate_rmse_episode_mean_rad_s': mean(velocities, 'yaw_rate_rmse_rad_s'),
                          'recovery_success_rate': mean(recoveries, 'recovery_success'),
                          'recovery_time_success_mean_s': mean(successes, 'recovery_time_s') if successes else None,
                          'evaluation_rollout_wall_seconds': sum(durations)},
              'limitations': ['One checkpoint and eight frames from one recovery clip; not broad generalization evidence',
                              'No pushes, observation noise, terrain or training-speed comparison',
                              'Errors are conditional on first-episode survival; inspect per-command results',
                              'Seeds vary startup randomization; actor is deterministic']}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
