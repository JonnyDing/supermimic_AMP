"""P3 online closure: current actor collection, WM refresh, BPTT and critic updates.

Upright velocity ablation only. Final checkpoint is fixed before paired evaluation.
"""
import argparse
import copy
import json
import time
from dataclasses import asdict
from pathlib import Path
import torch
from supermimic.adapters.mjlab import CaptureEnv
from supermimic.core.collection import OnlineCollection
from supermimic.core.imagination import ValueModel
from supermimic.core.online import AlternatingUpdates, snapshot, same_state
from supermimic.core.replay import SequenceReplay
from supermimic.core.world_model import WorldModel
from supermimic.evaluation.acceptance import compare, ComparabilityMargins, json_safe_metrics
from supermimic.evaluation.baseline import sha256
from supermimic.evaluation.bptt import run_trial, summarize
from supermimic.core.policy import ROOT, load_policy
from supermimic.evaluation.scenarios import fixed_recovery_frames
from supermimic.evaluation.world_model import evaluate
from supermimic.mdp.velocity import VelocityHeads


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--updates', type=int, default=100)
    p.add_argument('--actor-lr', type=float, default=1e-7)
    p.add_argument('--seed', type=int, default=2028)
    p.add_argument('--test-seeds', type=int, nargs='+', default=[62,63,64])
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    if a.updates < 1: p.error('updates must be positive')
    a.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(a.seed); torch.set_float32_matmul_precision('highest')
    p2 = ROOT/'logs/supermimic/p2_diagnostic_20261026/world_model.pt'
    p3 = ROOT/'logs/supermimic/p3_bptt_20261026/actor_10.pt'
    ck = torch.load(p2, map_location=a.device, weights_only=False)
    source = ROOT/ck['metadata']['checkpoint_source']['path']
    assert sha256(source) == ck['metadata']['checkpoint_source']['sha256']
    protocol = {'stage':'P3_online_velocity_ablation', 'seed':a.seed, 'updates':a.updates,
                'actor_lr':a.actor_lr, 'horizon':8, 'batch_size':128, 'num_envs':64,
                'warmup_steps':64, 'collection_steps_per_update':2, 'wm_batches_per_update':2,
                'amp':False, 'recovery_objective':False, 'pushes':False,
                'normalization':'frozen PPO observation and P2 WM statistics',
                'anchor_weight':1.0, 'bootstrap_weight':1.0, 'gamma':0.99, 'gae_lambda':0.95,
                'action_noise_std':0.1, 'actor_optimizer_initialization':'P3 momentum, overridden learning rate',
                'wm_optimizer_initialization':'P2 optimizer state', 'critic_optimizer_initialization':'P3 optimizer state',
                'selection':'fixed final update; no test-set checkpoint selection',
                'test_seeds':a.test_seeds, 'test_seconds':20, 'margins':asdict(ComparabilityMargins()),
                'sources':{str(f.relative_to(ROOT)):sha256(f) for f in (source,p2,p3)}}
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    baseline, baseline_norm = load_policy(source,a.device)
    teacher = copy.deepcopy(baseline.actor).eval()
    policy,norm = load_policy(p3,a.device)
    c3 = torch.load(p3,map_location=a.device,weights_only=False)
    wm = WorldModel(**ck['architecture']).to(a.device); wm.load_state_dict(ck['model'])
    critic = ValueModel(wm.input_mean.numel()+3).to(a.device); critic.load_state_dict(c3['p3_critic'])
    norm_before = snapshot(norm); actor_before = snapshot(policy.actor); wm_before = snapshot(wm)
    from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_env_cfg
    cfg = g1_amp_flat_env_cfg(); cfg.seed=a.seed; cfg.scene.num_envs=64
    cfg.events.pop('push_robot',None); cfg.curriculum={}; cfg.episode_length_s=5.
    env = CaptureEnv(cfg=cfg,device=a.device)
    replay = SequenceReplay(256,64,a.device)
    partition = torch.randperm(64,generator=torch.Generator().manual_seed(a.seed)).tolist()
    train_ids,holdout_ids = partition[:48],partition[48:]
    history=[]; started=time.perf_counter()
    try:
        env.reset(seed=a.seed)
        collector = OnlineCollection(env,policy,norm,replay)
        updater = AlternatingUpdates(wm,policy,norm,critic,VelocityHeads(env),teacher,a.actor_lr)
        updater.opt_wm.load_state_dict(ck['optimizer'])
        updater.opt_actor.load_state_dict(c3['actor_optimizer'])
        for group in updater.opt_actor.param_groups: group['lr']=a.actor_lr
        updater.opt_critic.load_state_dict(c3['critic_optimizer'])
        collector.collect(64,0)
        fixed = SequenceReplay.from_state_dict(copy.deepcopy(replay.state_dict()),a.device)
        validation_before = evaluate(wm,fixed,holdout_ids)
        for step in range(1,a.updates+1):
            collector.collect(2,step-1)
            row = updater.update(replay,train_ids,audit=(step==1 or step%10==0 or step==a.updates))
            history.append({'update':step,**row})
            if step%10==0 or step==1:
                print(json.dumps(history[-1]),flush=True)
                (a.output/'training_progress.json').write_text(json.dumps(history,indent=2))
        validation_after = evaluate(wm,fixed,holdout_ids)
        validation_current = evaluate(wm,replay,holdout_ids)
        checks = {'normalizer_unchanged':same_state(norm_before,norm),
                  'wm_statistics_unchanged':all(torch.equal(wm_before[k],v) for k,v in wm.named_buffers()),
                  'actor_changed':not same_state(actor_before,policy.actor),
                  'wm_changed':not same_state(wm_before,wm),
                  'audited_updates':sum(r['isolation_audited'] for r in history)}
        assert all(checks.values()), checks
        rows=replay.rows(torch.arange(replay.size,device=replay.device))
        versions=replay.storage['task/policy_version'][rows].unique().tolist()
        torch.save({'schema_version':1,'model_state_dict':policy.state_dict(),'obs_norm_state_dict':norm.state_dict(),
                    'p3_critic':critic.state_dict(),'world_model':wm.state_dict(),'architecture':ck['architecture'],
                    'actor_optimizer':updater.opt_actor.state_dict(),'critic_optimizer':updater.opt_critic.state_dict(),
                    'wm_optimizer':updater.opt_wm.state_dict(),'protocol':protocol,'update':a.updates,
                    'train_ids':train_ids,'holdout_ids':holdout_ids,'torch_rng':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state_all()},a.output/'actor_final.pt')
        torch.save(replay.state_dict(),a.output/'replay.pt')
        training={'checks':checks,'history':history,'guard_events':collector.guard_events,
                  'collection_steps':collector.total_steps,'transitions':collector.total_steps*64,
                  'retained_policy_versions':versions,'wm_updates':updater.wm_updates,
                  'training_seconds':time.perf_counter()-started,
                  'fixed_holdout_before':validation_before,'fixed_holdout_after':validation_after,
                  'current_holdout':validation_current}
        (a.output/'training.json').write_text(json.dumps(training,indent=2,allow_nan=False))
    finally: env.close()
    # Evaluate the serialized artifact, also checking native deployment loading.
    loaded,loaded_norm=load_policy(a.output/'actor_final.pt',a.device)
    assert same_state(snapshot(policy),loaded) and same_state(snapshot(norm),loaded_norm)
    _,data,frames=fixed_recovery_frames(); tests=[]
    for seed in a.test_seeds:
        pair={'seed':seed}
        for name,actor,normalizer in [('baseline',baseline,baseline_norm),('online',loaded,loaded_norm)]:
            trial=run_trial(seed,actor,normalizer,data,frames,a.device,20)
            pair[name]={'summary':summarize(trial),'trial':trial}
        for key in ('context_sha256','initial_state_sha256'):
            assert pair['baseline']['trial'][key]==pair['online']['trial'][key],key
        pair['comparability']=compare(pair['baseline']['summary'],pair['online']['summary'])
        tests.append(pair)
        print(json.dumps({n:pair[n]['summary'] for n in ('baseline','online')}),flush=True)
        (a.output/'test_progress.json').write_text(json.dumps(tests,indent=2))
    aggregate={n:{k:sum(t[n]['summary'][k] for t in tests)/len(tests) for k in tests[0][n]['summary']} for n in ('baseline','online')}
    result={'protocol':protocol,'aggregate':aggregate,'comparability':compare(aggregate['baseline'],aggregate['online']),
            'tests':tests,'training_checks':checks,'limitations':['One warm-started training seed; engineering comparability, not statistical equivalence',
            'Velocity objective only; recovery evaluated as regression, no AMP or push evaluation',
            'Checkpoint saves optimizer and replay; simulator-exact resume CLI is not implemented']}
    result, nonfinite = json_safe_metrics(result)
    result['nonfinite_metrics'] = nonfinite
    if nonfinite:
        result['comparability']['passed'] = False
        result['comparability']['checks']['all_episode_metrics_finite'] = False
    (a.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({'aggregate':aggregate,'comparability':result['comparability']},indent=2),flush=True)


if __name__=='__main__': main()
