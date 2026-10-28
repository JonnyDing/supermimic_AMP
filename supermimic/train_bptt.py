"""P3 controlled short-horizon BPTT, developmental selection, held-out real trials.

The PPO actor is warm-started; the P2 WM and observation normalizer are frozen.
This is an offline velocity-task ablation, not joint WMRL or AMP/recovery training.
"""
import argparse
import copy
import json
from pathlib import Path

import torch
from supermimic.core.imagination import ValueModel, critic_input, imagine, update_critic, freeze_parameters
from supermimic.core.replay import SequenceReplay
from supermimic.core.state import pack
from supermimic.core.world_model import WorldModel
from supermimic.core.policy import ROOT, load_policy
from supermimic.evaluation.scenarios import fixed_recovery_frames
from supermimic.evaluation.bptt import run_trial, summarize
from supermimic.evaluation.baseline import sha256
from supermimic.mdp.velocity import VelocityHeads
from supermimic.adapters.mjlab import CaptureEnv, read_physics

from supermimic.core.sampling import eligible_starts, sample_starts


def check_gradient(wm, policy, norm, critic, heads, starts, horizon):
    # Directional derivative on actor output bias; anchor excluded because it is a stop-gradient target.
    parameter = policy.actor[-1].bias
    direction = torch.randn_like(parameter); direction /= direction.norm()
    original = parameter.detach().clone()
    with freeze_parameters(wm, critic):
        objective = imagine(wm, policy.actor, norm, critic, heads, starts, horizon)['objective']
        derivative = (torch.autograd.grad(objective, parameter)[0] * direction).sum().item()
        with torch.no_grad():
            epsilon = 0.002
            parameter.copy_(original + epsilon * direction)
            plus = imagine(wm, policy.actor, norm, critic, heads, starts, horizon)['objective'].item()
            parameter.copy_(original - epsilon * direction)
            minus = imagine(wm, policy.actor, norm, critic, heads, starts, horizon)['objective'].item()
            parameter.copy_(original)
    finite_difference = (plus - minus) / (2 * epsilon)
    error = abs(derivative - finite_difference)
    if error > 5e-5 + 0.05 * abs(derivative):
        raise AssertionError((derivative, finite_difference, error))
    return {'autograd': derivative, 'finite_difference': finite_difference, 'absolute_error': error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--p2-dir', type=Path, default=ROOT/'logs/supermimic/p2_diagnostic_20261026')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--horizon', type=int, default=8)
    parser.add_argument('--updates', type=int, nargs='+', default=[10, 25, 50])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--actor-lr', type=float, default=3e-6)
    parser.add_argument('--anchor-weight', type=float, default=1.0)
    parser.add_argument('--bootstrap-weight', type=float, default=1.0)
    parser.add_argument('--critic-warmup', type=int, default=60)
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--development-seed', type=int, default=14201)
    parser.add_argument('--test-seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--development-seconds', type=int, default=10)
    parser.add_argument('--test-seconds', type=int, default=20)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Use a new output directory to preserve previous trials')
    args.output.mkdir(parents=True)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(args.seed)
    protocol = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    protocol.update({'stage': 'P3_upright_velocity_ablation', 'gamma': 0.99, 'gae_lambda': 0.95,
                     'reward': 'dt*(existing torso linear+angular tracking -0.01 action-rate sum -10 soft-joint-limit sum)',
                     'termination': 'smooth height/tilt in imagination; original hard termination in simulation',
                     'actor_initialization': 'frozen PPO checkpoint', 'wm_updates': 0, 'amp': False,
                     'selection': 'highest development return among candidates with unchanged velocity survival',
                     'real_benefit_gate': '>0.5% mean return gain; >=2/3 seeds improve; unchanged survival; XY/yaw mean RMSE each <=105% baseline',
                     'real_failure_penalty': '-200*dt at hard termination; imagination uses smooth survival weighting',
                     'recovery': 'regression observation only, no recovery-task training'})
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    checkpoint = torch.load(args.p2_dir/'world_model.pt', map_location=args.device, weights_only=False)
    replay = SequenceReplay.from_state_dict(torch.load(args.p2_dir/'replay.pt', map_location=args.device, weights_only=False), args.device)
    wm = WorldModel(**checkpoint['architecture']).to(args.device); wm.load_state_dict(checkpoint['model']); wm.eval()
    for p in wm.parameters(): p.requires_grad_(False)
    frozen_wm = {k: v.clone() for k,v in wm.state_dict().items()}
    source = ROOT/checkpoint['metadata']['checkpoint_source']['path']
    assert sha256(source) == checkpoint['metadata']['checkpoint_source']['sha256']
    policy, norm = load_policy(source, args.device)
    teacher = copy.deepcopy(policy.actor).eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    for p in policy.critic.parameters(): p.requires_grad_(False)
    norm.eval()
    from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_env_cfg
    cfg = g1_amp_flat_env_cfg(); cfg.scene.num_envs = 4; cfg.events.pop('push_robot', None)
    cfg.observations['actor'].enable_corruption = False
    env = CaptureEnv(cfg=cfg, device=args.device)
    try:
        env.reset(seed=args.seed)
        heads = VelocityHeads(env)
        analytic = heads.observation(pack(read_physics(env)), env.action_manager.action, env.command_manager.get_command('twist'))
        reference = env.obs_buf['actor'].reshape(4, 4, -1)[:, -1]
        observation_error = (analytic-reference).abs().max().item()
        if observation_error > 1e-5: raise AssertionError(observation_error)
    finally: env.close()
    critic = ValueModel(wm.input_mean.numel()+3).to(args.device)
    opt_v = torch.optim.Adam(critic.parameters(), lr=2e-4)
    opt_actor = torch.optim.Adam(policy.actor.parameters(), lr=args.actor_lr, betas=(0.9, 0.95))
    eligible = eligible_starts(replay, checkpoint['metadata']['train_env_ids'], args.horizon, heads.dt)
    torch.manual_seed(args.seed)
    checks = {'observation_error': observation_error, 'eligible_starts': len(eligible[0])}
    checks['gradient'] = check_gradient(wm, policy, norm, critic, heads, sample_starts(replay, eligible, 8), min(args.horizon, 4))
    for update in range(args.critic_warmup):
        with torch.no_grad():
            rollout = imagine(wm, policy.actor, norm, critic, heads, sample_starts(replay, eligible, args.batch_size), args.horizon)
        loss = update_critic(critic, opt_v, rollout)
        if (update+1) % 20 == 0: print(f'critic warmup {update+1}: {loss:.6f}', flush=True)
    history = []
    snapshots = {0: copy.deepcopy(policy.state_dict())}
    for update in range(1, max(args.updates)+1):
        batch = sample_starts(replay, eligible, args.batch_size)
        with freeze_parameters(wm, critic):
            rollout = imagine(wm, policy.actor, norm, critic, heads, batch, args.horizon,
                              teacher=teacher, anchor_weight=args.anchor_weight, bootstrap_weight=args.bootstrap_weight)
            opt_actor.zero_grad(set_to_none=True)
            (-rollout['objective']).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), 1.0, error_if_nonfinite=True)
            torch.nn.utils.clip_grad_value_(policy.actor.parameters(), 0.1)
            opt_actor.step()
        critic_loss = update_critic(critic, opt_v, rollout)
        row = dict(update=update, imagined_task_return=rollout['task_return'].item(),
                   objective=rollout['objective'].item(), anchor_mse=rollout['anchor_mse'].item(),
                   critic_loss=critic_loss, actor_gradient_norm=grad_norm.item())
        history.append(row)
        if update in args.updates:
            snapshots[update] = copy.deepcopy(policy.state_dict())
            torch.save({'model_state_dict': snapshots[update], 'obs_norm_state_dict': norm.state_dict(),
                        'p3_critic': critic.state_dict(), 'actor_optimizer': opt_actor.state_dict(),
                        'critic_optimizer': opt_v.state_dict(), 'protocol': protocol, 'update': update}, args.output/f'actor_{update}.pt')
            print(f'actor update {update}: {row}', flush=True)
    checks['world_model_unchanged'] = all(torch.equal(v, wm.state_dict()[k]) for k,v in frozen_wm.items())
    if not checks['world_model_unchanged']: raise AssertionError('WM changed during actor training')
    (args.output/'training.json').write_text(json.dumps({'checks':checks, 'history':history}, indent=2))
    _, data, frames = fixed_recovery_frames()
    dev = {}
    for update in [0]+args.updates:
        policy.load_state_dict(snapshots[update]); policy.eval()
        trial = run_trial(args.development_seed, policy, norm, data, frames, args.device, args.development_seconds)
        dev[str(update)] = {'summary':summarize(trial), 'trial':trial}
        print(f'development {update}: {dev[str(update)]["summary"]}', flush=True)
        (args.output/'development.json').write_text(json.dumps(dev, indent=2))
    baseline = dev['0']
    for result in dev.values():
        for key in ('context_sha256', 'initial_state_sha256'):
            if result['trial'][key] != baseline['trial'][key]: raise AssertionError(f'Unpaired development {key}')
    eligible_updates = [u for u in args.updates if dev[str(u)]['summary']['velocity_survival_rate'] >= baseline['summary']['velocity_survival_rate']]
    if not eligible_updates: eligible_updates = args.updates  # evaluate the least-bad candidate and report failure honestly
    selected = max(eligible_updates, key=lambda u:dev[str(u)]['summary']['mean_task_return'])
    (args.output/'selection.json').write_text(json.dumps({'selected_update':selected,'before_test_evaluation':True}, indent=2))
    tests = []
    for seed in args.test_seeds:
        pair = {'seed':seed}
        for name, update in [('baseline',0),('bptt',selected)]:
            policy.load_state_dict(snapshots[update]); policy.eval()
            trial = run_trial(seed, policy, norm, data, frames, args.device, args.test_seconds)
            pair[name] = {'summary':summarize(trial), 'trial':trial}
        for key in ('context_sha256','initial_state_sha256'):
            if pair['baseline']['trial'][key] != pair['bptt']['trial'][key]: raise AssertionError(f'Unpaired test {key}')
        tests.append(pair)
        print(f'test seed {seed}: baseline={pair["baseline"]["summary"]}, bptt={pair["bptt"]["summary"]}', flush=True)
        (args.output/'test_progress.json').write_text(json.dumps(tests, indent=2))
    aggregate = {name:{key:sum(p[name]['summary'][key] for p in tests)/len(tests) for key in tests[0][name]['summary']} for name in ('baseline','bptt')}
    b, a = aggregate['baseline'], aggregate['bptt']
    gain = (a['mean_task_return'] - b['mean_task_return'])/abs(b['mean_task_return'])
    wins = sum(p['bptt']['summary']['mean_task_return'] > p['baseline']['summary']['mean_task_return'] for p in tests)
    passed = gain > 0.005 and wins >= (2*len(tests)+2)//3 and a['velocity_survival_rate'] >= b['velocity_survival_rate'] and a['xy_rmse'] <= 1.05*b['xy_rmse'] and a['yaw_rmse'] <= 1.05*b['yaw_rmse']
    result = {'status':'completed', 'protocol':protocol, 'checks':checks, 'selected_update':selected,
              'aggregate':aggregate, 'relative_return_gain':gain, 'winning_test_seeds':wins,
              'real_benefit_gate_passed':passed, 'development':dev, 'tests':tests,
              'limitations':['PPO-warm-started offline velocity ablation; no online WM updates, AMP or recovery objective',
                             'One training seed; three paired evaluation seeds are not independent training replications']}
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:result[k] for k in ('aggregate','relative_return_gain','real_benefit_gate_passed')},indent=2))


if __name__=='__main__': main()
