"""P4 online AMP + recovery diagnostic, with fixed original-discriminator evaluation."""
import argparse,copy,json,time,pickle,hashlib,math
from pathlib import Path
from dataclasses import asdict
import torch
from supermimic.adapters.mjlab import CaptureEnv
from supermimic.core.collection import OnlineCollection
from supermimic.core.replay import SequenceReplay
from supermimic.core.world_model import WorldModel,rollout_loss
from supermimic.core.imagination import ValueModel,freeze_parameters,update_critic
from supermimic.core.online import snapshot,same_state
from supermimic.core.p4 import sample_starts,imagine
from supermimic.mdp.amp import AMPFeatures,DifferentiableAMP,DiscriminatorUpdates
from supermimic.mdp.recovery import RecoveryHeads
from supermimic.evaluation.world_model import DYNAMICS_KEYS,evaluate
from supermimic.evaluation.acceptance import compare,ComparabilityMargins,json_safe_metrics
from supermimic.core.policy import ROOT, load_policy
from supermimic.evaluation.scenarios import fixed_recovery_frames
from supermimic.evaluation.bptt import run_trial,summarize
from supermimic.evaluation.baseline import sha256
from supermimic.evaluation.metrics import full_summary, acceptance


def stats_digest(norm): return hashlib.sha256(pickle.dumps(norm.__dict__)).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=100)
    parser.add_argument('--actor-lr',type=float,default=1e-7)
    parser.add_argument('--seed',type=int,default=2030)
    parser.add_argument('--test-seeds',type=int,nargs='+',default=[82,83,84])
    parser.add_argument('--seconds',type=int,default=20)
    parser.add_argument('--device',default='cuda:0')
    a=parser.parse_args()
    if a.updates<1:parser.error('positive updates required')
    a.output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(a.seed);torch.set_float32_matmul_precision('highest')
    source=ROOT/'logs/rsl_rl/g1_amp_locomotion/2026-10-24_13-24-33/model_66600.pt'
    initial=ROOT/'logs/supermimic/p3_online_conservative_20261026/actor_final.pt'
    protocol={'stage':'P4_AMP_recovery_online','seed':a.seed,'updates':a.updates,'actor_lr':a.actor_lr,
              'horizon':8,'num_envs':64,'warmup_steps':64,'collection_steps_per_update':2,'batch_size':128,
              'wm_batches_per_update':2,'discriminator_batches_per_update':1,'critic_warmup':40,
              'discriminator_lr':1e-5,'discriminator_gradient_penalty':'native raw-expert, lambda=10',
              'amp_normalizer_update':'native normalized current policy/expert features; frozen during imagination',
              'expert_sampling':'native round-robin clips, adjacent 50Hz frames; exclude duplicate terminal pairs',
              'anchor_weight':1.,'amp_reward_coef':.1,'task_reward_lerp':.75,
              'test_seeds':a.test_seeds,'seconds':a.seconds,'selection':'fixed final update, no test selection',
              'margins':asdict(ComparabilityMargins()),'additional_gate':'full velocity AND recovery return drop <=2%',
              'full_reward_judge':'original frozen PPO discriminator and AMP normalizer for both policies',
              'push_protocol':{'seed':a.test_seeds[0],'time_s':5,'delta_velocity_xy':[.6,.3],'training_pushes':False},
              'approximations':['control-step finite-difference joint acceleration','contact probability for foot slip',
                                '4*binary self-contact probability approximates 4-substep count','smooth physical termination',
                                'future observation noise omitted; true initial observation history retained'],
              'sources':{str(p.relative_to(ROOT)):sha256(p) for p in (source,initial)}}
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    ck=torch.load(initial,map_location=a.device,weights_only=False)
    original=torch.load(source,map_location=a.device,weights_only=False)
    policy,norm=load_policy(initial,a.device);baseline,baseline_norm=load_policy(source,a.device)
    teacher=copy.deepcopy(baseline.actor).eval()
    for p in teacher.parameters():p.requires_grad_(False)
    wm=WorldModel(**ck['architecture']).to(a.device);wm.load_state_dict(ck['world_model'])
    amp=DifferentiableAMP(original,a.device)
    from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_env_cfg
    cfg=g1_amp_flat_env_cfg();cfg.seed=a.seed;cfg.scene.num_envs=64;cfg.events.pop('push_robot',None);cfg.curriculum={};cfg.episode_length_s=5.
    env=CaptureEnv(cfg=cfg,device=a.device)
    history=[];started=time.perf_counter()
    before_actor=snapshot(policy.actor);before_wm=snapshot(wm);before_discriminator=snapshot(amp.discriminator)
    before_norm=snapshot(norm);before_amp_norm=stats_digest(amp.normalizer)
    try:
        env.reset(seed=a.seed)
        features=AMPFeatures(env);heads=RecoveryHeads(env)
        experts=features.experts(ROOT/'src/assets/motions/g1/amp',a.device,heads.dt)
        discriminator=DiscriminatorUpdates(amp,features,experts)
        replay=SequenceReplay(256,64,a.device);collector=OnlineCollection(env,policy,norm,replay)
        partition=torch.randperm(64,generator=torch.Generator().manual_seed(a.seed)).tolist();train_ids,holdout=partition[:48],partition[48:]
        critic=ValueModel(wm.input_mean.numel()+3+8).to(a.device)
        opt_wm=torch.optim.Adam(wm.parameters(),lr=2e-4);opt_wm.load_state_dict(ck['wm_optimizer'])
        opt_actor=torch.optim.Adam(policy.actor.parameters(),lr=a.actor_lr,betas=(.9,.95));opt_actor.load_state_dict(ck['actor_optimizer'])
        for group in opt_actor.param_groups:group['lr']=a.actor_lr
        opt_v=torch.optim.Adam(critic.parameters(),lr=2e-4)
        collector.collect(64,0)
        fixed=SequenceReplay.from_state_dict(copy.deepcopy(replay.state_dict()),a.device)
        validation_before=evaluate(wm,fixed,holdout)
        for _ in range(40):
            with torch.no_grad():rollout=imagine(wm,policy.actor,norm,critic,heads,amp,features,sample_starts(replay,train_ids,heads))
            update_critic(critic,opt_v,rollout)
        for step in range(1,a.updates+1):
            collector.collect(2,step-1);audit=step==1 or step%10==0 or step==a.updates
            actor_check=snapshot(policy.actor) if audit else None
            for p in wm.parameters():p.requires_grad_(True)
            wm_losses=[]
            for _ in range(2):
                loss,_,_=rollout_loss(wm,replay.sample(128,8,train_ids,DYNAMICS_KEYS))
                opt_wm.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(wm.parameters(),1.,error_if_nonfinite=True);opt_wm.step();wm_losses.append(float(loss.detach()))
            metrics=discriminator.update(replay,train_ids)
            if audit:assert same_state(actor_check,policy.actor),'D/WM update changed actor'
            opt_wm.zero_grad(set_to_none=True);opt_v.zero_grad(set_to_none=True)
            frozen=[snapshot(m) for m in (wm,critic,amp.discriminator)] if audit else None
            norm_digest=stats_digest(amp.normalizer)
            starts=sample_starts(replay,train_ids,heads)
            with freeze_parameters(wm,critic,amp.discriminator):
                rollout=imagine(wm,policy.actor,norm,critic,heads,amp,features,starts,teacher=teacher)
                opt_actor.zero_grad(set_to_none=True);(-rollout['objective']).backward()
                grad=torch.nn.utils.clip_grad_norm_(policy.actor.parameters(),1.,error_if_nonfinite=True)
                torch.nn.utils.clip_grad_value_(policy.actor.parameters(),.1);opt_actor.step()
            assert all(p.grad is None for m in (wm,critic,amp.discriminator) for p in m.parameters())
            assert stats_digest(amp.normalizer)==norm_digest
            if audit:assert all(same_state(s,m) for s,m in zip(frozen,(wm,critic,amp.discriminator)))
            actor_check=snapshot(policy.actor) if audit else None
            vloss=update_critic(critic,opt_v,rollout)
            if audit:assert same_state(actor_check,policy.actor)
            row={'update':step,**metrics,'wm_loss':sum(wm_losses)/2,'critic_loss':vloss,
                 'actor_gradient_norm':float(grad),'imagined_return':float(rollout['task_return'].detach()),
                 'recovery_mode_fraction':float(rollout['recovery_mode_fraction']),
                 'recovery_start_fraction':float(starts['task/recovery_enabled'].float().mean()),
                 'anchor_mse':float(rollout['anchor_mse'].detach()),'isolation_audited':audit}
            history.append(row)
            if step==1 or step%10==0:print(json.dumps(row),flush=True)
            (a.output/'training_progress.json').write_text(json.dumps(history,indent=2,allow_nan=False))
        checks={'actor_changed':not same_state(before_actor,policy.actor),'wm_changed':not same_state(before_wm,wm),
                'discriminator_changed':not same_state(before_discriminator,amp.discriminator),
                'amp_statistics_updated':before_amp_norm!=stats_digest(amp.normalizer),
                'actor_normalizer_frozen':same_state(before_norm,norm),
                'wm_statistics_frozen':all(torch.equal(before_wm[k],v) for k,v in wm.named_buffers()),
                'recovery_starts_used':all(r['recovery_start_fraction']>.39 for r in history),
                'recovery_reward_used':any(r['recovery_mode_fraction']>0 for r in history)}
        assert all(checks.values()),checks
        saved={'schema_version':2,'model_state_dict':policy.state_dict(),'obs_norm_state_dict':norm.state_dict(),
               'world_model':wm.state_dict(),'architecture':ck['architecture'],'p4_critic':critic.state_dict(),
               'discriminator_state_dict':amp.discriminator.state_dict(),'amp_normalizer':amp.normalizer,
               'wm_optimizer':opt_wm.state_dict(),'actor_optimizer':opt_actor.state_dict(),'critic_optimizer':opt_v.state_dict(),
               'discriminator_optimizer':discriminator.optimizer.state_dict(),'expert_cursor':experts.cursor,
               'train_ids':train_ids,'holdout_ids':holdout,'protocol':protocol,'update':a.updates,
               'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
        torch.save(saved,a.output/'actor_final.pt');torch.save(replay.state_dict(),a.output/'replay.pt')
        training={'checks':checks,'history':history,'transitions':collector.total_steps*64,'guard_events':collector.guard_events,
                  'training_seconds':time.perf_counter()-started,'fixed_holdout_before':validation_before,
                  'fixed_holdout_after':evaluate(wm,fixed,holdout),'expert_paths':experts.paths,
                  'train_ids':train_ids,'holdout_ids':holdout}
        (a.output/'training.json').write_text(json.dumps(training,indent=2,allow_nan=False))
    finally:env.close()
    loaded,loaded_norm=load_policy(a.output/'actor_final.pt',a.device)
    assert same_state(snapshot(policy),loaded) and same_state(snapshot(norm),loaded_norm)
    judge=DifferentiableAMP(original,a.device)
    _,data,frames=fixed_recovery_frames();tests=[]
    settings=[(seed,None) for seed in a.test_seeds]+[(a.test_seeds[0],250)]
    for seed,push in settings:
        pair={'seed':seed,'push_step':push}
        for name,actor,normalizer in [('baseline',baseline,baseline_norm),('p4',loaded,loaded_norm)]:
            trial=run_trial(seed,actor,normalizer,data,frames,a.device,a.seconds,full_amp=judge,push_step=push)
            pair[name]={'summary':full_summary(trial),'trial':trial}
        for key in ('context_sha256','initial_state_sha256'):assert pair['baseline']['trial'][key]==pair['p4']['trial'][key]
        pair['comparability']=acceptance(pair['baseline']['summary'],pair['p4']['summary']);tests.append(pair)
        print(json.dumps({k:pair[k]['summary'] for k in ('baseline','p4')}),flush=True)
        (a.output/'test_progress.json').write_text(json.dumps(tests,indent=2))
    groups={}
    for label,pairs in [('fixed',[p for p in tests if p['push_step'] is None]),('push',[p for p in tests if p['push_step'] is not None])]:
        aggregate={n:{k:sum(p[n]['summary'][k] for p in pairs)/len(pairs) for k in pairs[0][n]['summary']} for n in ('baseline','p4')}
        groups[label]={'aggregate':aggregate,'comparability':acceptance(aggregate['baseline'],aggregate['p4'])}
    result={'protocol':protocol,'groups':groups,'tests':tests,'training_checks':checks,
            'passed':all(g['comparability']['passed'] for g in groups.values()),
            'limitations':['single PPO-warm-started training seed, small online diagnostic',
                           'training pushes disabled; deterministic push regression only, not complete random-push curriculum',
                           'contact/acceleration/termination proxies; no simulator-exact resume CLI']}
    result,bad=json_safe_metrics(result);result['nonfinite_metrics']=bad
    if bad:result['passed']=False
    (a.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({'groups':result['groups'],'passed':result['passed']},indent=2),flush=True)


if __name__=='__main__':main()
