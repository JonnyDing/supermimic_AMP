"""Real mjlab evaluation for P3, with the same post-state task head as BPTT.

Initial states, domain randomization and horizon are paired explicitly by hashes.
The return is the P3 four-term ablation, not the full AMP training objective.
"""
import hashlib
import math
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat
import time
from supermimic.adapters.mjlab import CaptureEnv, PhysicsCollector, read_physics
from supermimic.mdp.kinematics import RobotKinematics
from supermimic.mdp.velocity import VelocityHeads
from supermimic.core.state import pack
from supermimic.evaluation.scenarios import COMMANDS, initialize_scenarios

@torch.no_grad()
def run_trial(seed, policy, norm, data, frames, device, horizon_seconds=20, full_amp=None, push_step=None):
    from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_env_cfg
    cfg = g1_amp_flat_env_cfg()
    cfg.seed = seed
    cfg.scene.num_envs = len(COMMANDS) + len(frames)
    cfg.events.pop('push_robot', None)
    cfg.curriculum = {}
    cfg.observations['actor'].enable_corruption = False
    cfg.episode_length_s = float(horizon_seconds)
    env = CaptureEnv(cfg=cfg, device=device)
    try:
        env.reset(seed=seed)
        collector, kin = PhysicsCollector(env), RobotKinematics(env)
        heads = VelocityHeads(env)
        # Warm TorchScript observation/geometry kernels before installing scored states.
        # Lazy profiling/fusion otherwise changes floating-point rounding mid-comparison.
        for _ in range(6):
            warm_state=pack(read_physics(env))
            heads.reward(warm_state,env.action_manager.action,env.action_manager.prev_action,
                         env.command_manager.get_command('twist'))
            env.observation_manager.compute(update_history=False)
        initialize_scenarios(env, data, frames)
        if full_amp is not None:
            from supermimic.mdp.amp import AMPFeatures
            amp_features = AMPFeatures(env)
            full_return = torch.zeros(env.num_envs, device=device)
            style_total = torch.zeros_like(full_return)
        context_digest = hashlib.sha256(collector.context().cpu().numpy().tobytes()).hexdigest()
        # Hash simulator coordinates, not a quaternion-derived world angular velocity.
        # Equivalent derived transforms can differ by a few ULPs after JIT warmup.
        initial_digest = hashlib.sha256(env.sim.data.qpos.cpu().numpy().tobytes()+env.sim.data.qvel.cpu().numpy().tobytes()).hexdigest()
        torso = list(env.scene['robot'].body_names).index('torso_link')
        n, nv, dt = env.num_envs, len(COMMANDS), env.step_dt
        active = torch.ones(n, dtype=torch.bool, device=device)
        task_return = torch.zeros(n, device=device)
        steps = torch.zeros(n, device=device)
        xy_error = torch.zeros(n, device=device); yaw_error = torch.zeros_like(xy_error)
        action_cost = torch.zeros_like(xy_error); slip_sum = torch.zeros_like(xy_error)
        contact_count = torch.zeros_like(xy_error)
        consecutive = torch.zeros(n, device=device, dtype=torch.long)
        recovery_time = torch.full((n,), float('nan'), device=device)
        failure = torch.zeros(n, dtype=torch.bool, device=device)
        control_steps = round(horizon_seconds / dt)
        start = time.perf_counter()
        for step in range(control_steps):
            if push_step is not None and step == push_step:
                ids = env.scene['robot'].indexing.free_joint_v_adr.long()[:2]
                env.sim.data.qvel[:nv, ids] += torch.tensor([0.6, 0.3], device=device)
                env.sim.forward(); env.sim.sense()
            actions = policy.act_inference(norm(env.obs_buf['actor']))
            if not torch.isfinite(actions).all():
                raise RuntimeError(f'Policy produced non-finite actions at step {step}; envs={(~torch.isfinite(actions).all(-1)).nonzero().flatten().tolist()}; active={active.tolist()}')
            tr = collector.step(actions)
            step_reward = heads.reward(pack(tr.next_state), actions, tr.task['previous_action'], tr.task['command'])
            task_return += torch.where(active, step_reward - 200 * dt * tr.terminated.float(), 0.)
            if full_amp is not None:
                amp_return, score = full_amp.reward(amp_features(pack(tr.state)), amp_features(pack(tr.next_state)), tr.reward)
                style = full_amp.discriminator.amp_reward_coef * (1-.25*(score-1).square()).clamp_min(0)
                full_return += torch.where(active, amp_return, 0.)
                style_total += torch.where(active, style, 0.)
            _, rotations, linear, angular = kin(tr.next_state)
            local_velocity = quat_apply_inverse(yaw_quat(rotations[:, torso]), linear[:, torso])
            xy_error += torch.where(active, (local_velocity[:, :2] - tr.task['command'][:, :2]).square().sum(-1), 0.)
            yaw_error += torch.where(active, (angular[:, torso, 2] - tr.task['command'][:, 2]).square(), 0.)
            action_cost += torch.where(active, (actions - tr.task['previous_action']).square().sum(-1), 0.)
            speed = tr.labels['feet_velocity_xy'].norm(dim=-1)
            contact = tr.labels['feet_in_contact']
            slip_sum += torch.where(active, (speed * contact).sum(-1), 0.)
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
                   'p3_task_return': float(task_return[i]), 'observed_seconds': float(steps[i]) * dt, 'terminated': bool(failure[i]),
                   'velocity_xy_rmse_m_s': math.sqrt(float(xy_error[i]) / count),
                   'yaw_rate_rmse_rad_s': math.sqrt(float(yaw_error[i]) / count),
                   'action_delta_squared_sum_mean': float(action_cost[i]) / count,
                   'contact_foot_speed_mean_m_s': float(slip_sum[i]) / max(float(contact_count[i]), 1)}
            if full_amp is not None:
                row['full_amp_task_return'] = float(full_return[i])
                row['style_reward_mean'] = float(style_total[i]) / count
            if i >= nv:
                row['recovery_success'] = bool(torch.isfinite(recovery_time[i]))
                row['recovery_time_s'] = float(recovery_time[i]) if row['recovery_success'] else None
            else:
                row['survived_full_horizon'] = not row['terminated'] and float(steps[i]) >= control_steps
            rows.append(row)
        return {'episodes': rows, 'elapsed_seconds': elapsed, 'context_sha256': context_digest, 'initial_state_sha256': initial_digest, 'initial_hash_schema': 'raw_qpos_qvel_v2', 'math_warmup_calls': 6, 'push_step': push_step, 'push_velocity_xy': [0.6, 0.3] if push_step is not None else None}
    finally:
        env.close()


def summarize(trial):
    velocity = [r for r in trial['episodes'] if r['kind'] == 'velocity']
    recovery = [r for r in trial['episodes'] if r['kind'] == 'recovery']
    mean = lambda rows, key: sum(r[key] for r in rows) / len(rows)
    return {'mean_task_return': mean(velocity, 'p3_task_return'),
            'velocity_survival_rate': mean(velocity, 'survived_full_horizon'),
            'xy_rmse': mean(velocity, 'velocity_xy_rmse_m_s'),
            'yaw_rmse': mean(velocity, 'yaw_rate_rmse_rad_s'),
            'recovery_success_rate': mean(recovery, 'recovery_success')}


def main():
    """Re-evaluate one candidate versus its frozen PPO initialization."""
    import argparse
    import json
    from pathlib import Path
    from supermimic.core.policy import ROOT, load_policy
    from supermimic.evaluation.scenarios import fixed_recovery_frames
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--seconds', type=int, default=20)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_float32_matmul_precision('highest')
    manifest = json.loads((ROOT/'supermimic/reports/baseline_manifest.json').read_text())
    policy, norm = load_policy(ROOT/manifest['checkpoint']['path'], args.device)
    baseline = {k:v.clone() for k,v in policy.state_dict().items()}
    saved = torch.load(args.candidate, map_location=args.device, weights_only=False)
    for key,value in norm.state_dict().items():
        if not torch.equal(value, saved['obs_norm_state_dict'][key]):
            raise ValueError(f'Candidate observation normalizer differs: {key}')
    _, data, frames = fixed_recovery_frames()
    result = []
    for seed in args.seeds:
        pair = {'seed':seed}
        for name, state in [('baseline',baseline),('candidate',saved['model_state_dict'])]:
            policy.load_state_dict(state); policy.eval()
            trial = run_trial(seed, policy, norm, data, frames, args.device, args.seconds)
            pair[name] = {'summary':summarize(trial), 'trial':trial}
        for key in ('context_sha256','initial_state_sha256'):
            if pair['baseline']['trial'][key] != pair['candidate']['trial'][key]:
                raise AssertionError(f'Unpaired evaluation: {key}')
        result.append(pair)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__': main()
