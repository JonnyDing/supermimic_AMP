"""Optimized P4 trainer: compiled shared FK, configurable throughput, atomic resume.

Resume restores learning state/replay/RNG, then starts fresh simulator episodes.
It does not claim bitwise continuation of simulator internal/contact state.
"""
import argparse,copy,json,time,random,os
from pathlib import Path
import numpy as np
import torch
from supermimic.adapters.mjlab import CaptureEnv
from supermimic.core.collection import OnlineCollection
from supermimic.core.replay import SequenceReplay
from supermimic.core.world_model import WorldModel
from supermimic.core.imagination import ValueModel,freeze_parameters,update_critic
from supermimic.core.online import snapshot,same_state
from supermimic.core.p4 import sample_starts
from supermimic.core.fast import FastP4
from supermimic.core.checkpoint import atomic_save,rng_state,restore_rng
from supermimic.mdp.amp import AMPFeatures,DifferentiableAMP,DiscriminatorUpdates
from supermimic.mdp.recovery import RecoveryHeads
from supermimic.evaluation.world_model import DYNAMICS_KEYS
from supermimic.core.policy import ROOT, load_policy
from supermimic.evaluation.baseline import sha256
from supermimic.core.logging import progress

DEFAULTS={'num_envs':64,'batch_size':128,'wm_batch_size':128,'buffer_steps':256,'warmup_steps':64,
          'horizon':8,'utd':.5,'wm_batches':2,'critic_epochs':2,'critic_minibatches':4,'critic_warmup':40,
          'gamma':.99,'gae_lambda':.95,'actor_lr':1e-7,'discriminator_lr':1e-5,'precision':'fp32','seed':2030,'compile':True}


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=1000)
    p.add_argument('--resume',type=Path);p.add_argument('--device',default='cuda:0')
    p.add_argument('--checkpoint-every',type=int,default=1000);p.add_argument('--log-every',type=int,default=50)
    p.add_argument('--audit-every',type=int,default=1000);p.add_argument('--resume-refill-steps',type=int,default=32)
    p.add_argument('--evaluate',action='store_true',help='Run full real paired evaluation after training')
    p.add_argument('--test-seeds',nargs='+',type=int,default=[82,83,84])
    p.add_argument('--seconds',type=int,default=20)
    for key,value in DEFAULTS.items():
        if key=='compile':p.add_argument('--compile',action=argparse.BooleanOptionalAction,default=None)
        else:p.add_argument('--'+key.replace('_','-'),type=type(value),default=None)
    p.add_argument('--logger',choices=['tensorboard'],default='tensorboard')
    p.add_argument('--checkpoint-prefix',choices=['actor','model'],default='actor')
    p.add_argument('--terminal-format',choices=['pretty','json'],default='pretty')
    a=p.parse_args(argv);checkpoint=None
    if a.resume:
        checkpoint=torch.load(a.resume,map_location='cpu',weights_only=False)
        if checkpoint.get('fast_schema')!=1 or 'replay' not in checkpoint:p.error('Resume needs a full train_fast last.pt checkpoint')
    for key,default in DEFAULTS.items():
        prior=checkpoint['config'].get(key,default) if checkpoint else default
        supplied=getattr(a,key)
        if checkpoint and supplied is not None and supplied!=prior:p.error(f'Resume config mismatch for {key}: {supplied} != {prior}')
        setattr(a,key,prior if supplied is None else supplied)
    if a.precision not in ('fp32','tf32','bf16'):p.error('precision: fp32, tf32 or bf16')
    if a.num_envs<16 or a.batch_size<8 or a.wm_batch_size<1:p.error('Need >=16 envs, >=8 actor batch, positive WM batch')
    if a.horizon<1 or a.buffer_steps<max(a.horizon,8) or a.warmup_steps<max(a.horizon,8):p.error('Buffer/warmup must fit rollout windows')
    if min(a.updates,a.wm_batches,a.critic_epochs,a.critic_minibatches,a.checkpoint_every,a.log_every)<1 or a.utd<=0:p.error('Counts and UTD must be positive')
    if checkpoint and a.updates<=checkpoint['update']:p.error('--updates is total target, must exceed saved update')
    return a,checkpoint


def main(argv=None,env_cfg=None,compatibility_config=None):
    a,resume=arguments(argv);a.output.mkdir(parents=True,exist_ok=False)
    os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','4')
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config['resume_semantics']='restore replay, optimizers and RNG; reset simulator, increment episode IDs and refill'
    config['task']='P4 AMP/recovery, startup DR on, random interval pushes off'
    config['imagined_action_guard']='exclude entire trajectory if any action is nonfinite or abs >=100; no reward/target clipping'
    (a.output/'config.json').write_text(json.dumps(config,indent=2))
    env_signature=None
    if compatibility_config is not None:
        from mjlab.utils.os import dump_yaml
        for name in ('env','agent'):dump_yaml(a.output/'params'/f'{name}.yaml',compatibility_config[name])
        env_signature=sha256(a.output/'params'/'env.yaml')
    if resume and resume.get('compatibility_env_signature') is not None:
        if env_signature!=resume['compatibility_env_signature']:
            raise ValueError('Resume environment config differs; use the same scripts/train.py env options')
    torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    torch.set_float32_matmul_precision('highest')
    source=ROOT/'logs/rsl_rl/g1_amp_locomotion/2026-10-24_13-24-33/model_66600.pt'
    initial=ROOT/'logs/supermimic/p3_online_conservative_20261026/actor_final.pt'
    if resume:
        for path,digest in resume['sources'].items():
            if sha256(ROOT/path)!=digest:raise ValueError(f'Resume source checkpoint changed: {path}')
    original=torch.load(source,map_location=a.device,weights_only=False)
    ck=resume or torch.load(initial,map_location=a.device,weights_only=False)
    policy,norm=load_policy(a.resume if resume else initial,a.device)
    baseline,_=load_policy(source,a.device);teacher=baseline.actor.eval()
    for p in teacher.parameters():p.requires_grad_(False)
    wm=WorldModel(**ck['architecture']).to(a.device);wm.load_state_dict(ck['world_model'])
    amp=DifferentiableAMP(ck if resume else original,a.device)
    from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_env_cfg
    if env_cfg is None:
        cfg=g1_amp_flat_env_cfg();cfg.events.pop('push_robot',None);cfg.curriculum={};cfg.episode_length_s=5.
    else:cfg=copy.deepcopy(env_cfg)
    cfg.seed=a.seed;cfg.scene.num_envs=a.num_envs
    env=CaptureEnv(cfg=cfg,device=a.device);start_time=time.perf_counter()
    start_update=ck['update'] if resume else 0
    checks=[];writer=None;collector=None
    try:
        if a.logger=='tensorboard':
            from torch.utils.tensorboard import SummaryWriter
            writer=SummaryWriter(log_dir=str(a.output))
        env.reset(seed=a.seed);heads=RecoveryHeads(env);features=AMPFeatures(env)
        experts=features.experts(ROOT/'src/assets/motions/g1/amp',a.device,heads.dt)
        critic=ValueModel(wm.input_mean.numel()+11).to(a.device)
        if resume:critic.load_state_dict(ck['p4_critic'])
        engine=FastP4(wm,policy.actor,norm,critic,heads,amp,features,teacher,compile=a.compile,precision=a.precision,horizon=a.horizon,gamma=a.gamma)
        discriminator=DiscriminatorUpdates(amp,features,experts,lr=a.discriminator_lr)
        opt_wm=torch.optim.Adam(wm.parameters(),lr=2e-4);opt_wm.load_state_dict(ck['wm_optimizer'])
        opt_actor=torch.optim.Adam(policy.actor.parameters(),lr=a.actor_lr,betas=(.9,.95));opt_actor.load_state_dict(ck['actor_optimizer'])
        for group in opt_actor.param_groups:group['lr']=a.actor_lr
        opt_v=torch.optim.Adam(critic.parameters(),lr=2e-4)
        if resume:
            opt_v.load_state_dict(ck['critic_optimizer']);discriminator.optimizer.load_state_dict(ck['discriminator_optimizer'])
            experts.cursor=ck['expert_cursor'];replay=SequenceReplay.from_state_dict(ck['replay'],a.device)
            train_ids,holdout=ck['train_ids'],ck['holdout_ids']
        else:
            replay=SequenceReplay(a.buffer_steps,a.num_envs,a.device)
            partition=torch.randperm(a.num_envs,generator=torch.Generator().manual_seed(a.seed)).tolist()
            train_ids,holdout=partition[:a.num_envs*3//4],partition[a.num_envs*3//4:]
        collector=OnlineCollection(env,policy,norm,replay)
        print(f'[SuperMimic] Collecting initial replay: {a.num_envs} envs; log directory: {a.output}',flush=True)
        if resume:
            restore_rng(ck['rng']);collector.total_steps=ck['collection_steps']
            collector.collect(max(a.resume_refill_steps,a.horizon),start_update)
        else:collector.collect(a.warmup_steps,0)
        stats_before={k:v.clone() for k,v in wm.named_buffers()};norm_before=snapshot(norm)
        # Critic warmup stays eager to avoid a separate compiled no-grad specialization.
        if not resume:
            print(f'[SuperMimic] Critic warmup: {a.critic_warmup} updates',flush=True)
            warm=FastP4(wm,policy.actor,norm,critic,heads,amp,features,teacher,precision=a.precision,horizon=a.horizon,gamma=a.gamma)
            for _ in range(a.critic_warmup):
                with torch.no_grad():r=warm(sample_starts(replay,train_ids,heads,a.batch_size,a.horizon))
                update_critic(critic,opt_v,r,a.critic_epochs,a.critic_minibatches,return_tensor=True,gamma=a.gamma,lam=a.gae_lambda)
        accumulator=ck.get('collection_accumulator',0.) if resume else 0.
        prior_wall=ck.get('training_wall_seconds',0.) if resume else 0.
        latest_metrics={};pending=[];last_log=time.perf_counter();last_log_update=start_update
        last_guard=0;last_collection_steps=collector.total_steps
        print(f'[SuperMimic] Training {start_update + 1}..{a.updates}; first update may compile kernels',flush=True)
        print(f'[Stats] Metrics are interval means (up to {a.log_every} updates); peak fields are interval maxima.',flush=True)
        def save_checkpoint(step):
            save_started=time.perf_counter()
            print(f'[Checkpoint] Saving update {step}: policy and full replay/optimizer state...',flush=True)
            payload={'fast_schema':1,'schema_version':2,'config':config,'update':step,
                     'model_state_dict':policy.state_dict(),'obs_norm_state_dict':norm.state_dict(),
                     'world_model':wm.state_dict(),'architecture':wm.architecture(),'p4_critic':critic.state_dict(),
                     'discriminator_state_dict':amp.discriminator.state_dict(),'amp_normalizer':amp.normalizer.native(),
                     'wm_optimizer':opt_wm.state_dict(),'actor_optimizer':opt_actor.state_dict(),
                     'critic_optimizer':opt_v.state_dict(),'discriminator_optimizer':discriminator.optimizer.state_dict(),
                     'expert_cursor':experts.cursor,'rng':rng_state(),'train_ids':train_ids,'holdout_ids':holdout,
                     'collection_steps':collector.total_steps,'collection_accumulator':accumulator,
                     'training_wall_seconds':prior_wall+time.perf_counter()-start_time,
                     'compatibility_env_signature':env_signature,
                     'sources':{str(p.relative_to(ROOT)):sha256(p) for p in (source,initial)}}
            atomic_save(payload,a.output/f'{a.checkpoint_prefix}_{step}.pt')
            atomic_save({**payload,'replay':replay.state_dict()},a.output/'last.pt')
            print(f'[Checkpoint] Saved {a.checkpoint_prefix}_{step}.pt and last.pt in {time.perf_counter()-save_started:.1f}s',flush=True)
            return payload
        for step in range(start_update+1,a.updates+1):
            accumulator+=1/a.utd;collect_steps=int(accumulator);accumulator-=collect_steps
            torch.set_float32_matmul_precision('highest')
            collector.collect(collect_steps,step-1)
            torch.set_float32_matmul_precision('high' if a.precision=='tf32' else 'highest')
            audit=step==start_update+1 or (a.audit_every>0 and step%a.audit_every==0) or step==a.updates
            actor_check=snapshot(policy.actor) if audit else None
            losses=[]
            for _ in range(a.wm_batches):
                window=replay.sample(a.wm_batch_size,8,train_ids,DYNAMICS_KEYS)
                loss=engine.wm_loss(window)[0]
                opt_wm.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(wm.parameters(),1.,error_if_nonfinite=True);opt_wm.step();losses.append(loss.detach())
            metrics=discriminator.update(replay,train_ids,a.batch_size,return_tensors=True)
            if audit:assert same_state(actor_check,policy.actor)
            opt_wm.zero_grad(set_to_none=True);opt_v.zero_grad(set_to_none=True)
            frozen=[snapshot(m) for m in (wm,critic,amp.discriminator,amp.normalizer)] if audit else None
            starts=sample_starts(replay,train_ids,heads,a.batch_size,a.horizon)
            with freeze_parameters(wm,critic,amp.discriminator):
                rollout=engine(starts)
                if not rollout['valid_trajectories'].any():
                    raise FloatingPointError('All imagined trajectories violate the action contract; actor update aborted')
                rejected=(~rollout['valid_trajectories']).nonzero().flatten()
                if len(rejected):
                    event={'update':step,'rejected_count':len(rejected),'batch_size':a.batch_size,
                           'env_ids':starts['_env_ids'][rejected].cpu().tolist(),
                           'start_ages':starts['_start_ages'][rejected].cpu().tolist(),
                           'reason':'nonfinite imagined action or max absolute action >=100'}
                    writer.add_text('Diagnostics/imagined_guard_events',json.dumps(event),step)
                opt_actor.zero_grad(set_to_none=True);(-rollout['objective']).backward()
                grad=torch.nn.utils.clip_grad_norm_(policy.actor.parameters(),1.,error_if_nonfinite=True)
                torch.nn.utils.clip_grad_value_(policy.actor.parameters(),.1);opt_actor.step()
            if audit:
                assert all(p.grad is None for m in (wm,critic,amp.discriminator) for p in m.parameters())
                assert all(same_state(s,m) for s,m in zip(frozen,(wm,critic,amp.discriminator,amp.normalizer)))
                actor_check=snapshot(policy.actor)
            vloss,critic_metrics=update_critic(critic,opt_v,rollout,a.critic_epochs,a.critic_minibatches,return_tensor=True,gamma=a.gamma,lam=a.gae_lambda,return_metrics=True)
            metrics.update(critic_metrics)
            # Rare outliers retain the actual trajectory instead of only a 50-update mean.
            if vloss.detach().item()>100:
                targets=rollout['critic_targets'];values=rollout['critic_values_before']
                residual=(values[:-1]-targets).square().masked_fill(~rollout['valid_trajectories'][None],-1)
                worst=int(residual.flatten().argmax())
                time_id,batch_id=divmod(worst,targets.shape[1])
                diagnostic={'update':step,'critic_loss':float(vloss),'worst_time':time_id,
                            'env_id':int(starts['_env_ids'][batch_id]),'start_age':int(starts['_start_ages'][batch_id]),
                            'rewards':rollout['rewards'][:,batch_id].detach().cpu().tolist(),
                            'continuations':rollout['continuations'][:,batch_id].detach().cpu().tolist(),
                            'values':values[:,batch_id].cpu().tolist(),'targets':targets[:,batch_id].cpu().tolist(),
                            'actions_abs_max':rollout['imagined_actions'][:,batch_id].abs().amax(-1).cpu().tolist(),
                            'inputs_abs_max':rollout['critic_inputs'][:,batch_id].detach().abs().amax(-1).cpu().tolist(),
                            'start_history_abs_max':float(starts['task/observation_history'][batch_id].abs().amax())}
                writer.add_text('Diagnostics/critic_outlier',json.dumps(diagnostic,allow_nan=False),step)
            if audit:
                assert same_state(actor_check,policy.actor)
                assert same_state(norm_before,norm)
                assert all(torch.equal(v,stats_before[k]) for k,v in wm.named_buffers())
                checks.append({'update':step,'parameter_and_statistics_isolation':True})
            metrics.update(wm_loss=torch.stack(losses).mean(),critic_loss=vloss,actor_gradient_norm=grad.detach(),
                           imagined_rejected_fraction=rollout['imagined_rejected_fraction'].detach(),
                           imagined_return=rollout['task_return'].detach(),recovery_mode_fraction=rollout['recovery_mode_fraction'].detach())
            pending.append(metrics)
            if step%a.log_every==0 or step==a.updates:
                keys=list(metrics);matrix=torch.stack([torch.stack([r[k] for k in keys]) for r in pending])
                values=matrix.mean(0)
                for i,key in enumerate(keys):
                    if key.endswith('_abs_max'):values[i]=matrix[:,i].amax()
                values=values.cpu().tolist()
                elapsed=time.perf_counter()-last_log
                new_guards=collector.guard_events[last_guard:]
                latest_metrics={'update':step,**dict(zip(keys,values)),'interval_seconds':elapsed,
                                'updates_per_second':(step-last_log_update)/elapsed,'collection_steps':collector.total_steps,
                                'transitions':collector.total_steps*a.num_envs,'first_interval_includes_compile':bool(a.compile and last_log_update==start_update),
                                'interval_updates':step-last_log_update,'elapsed_seconds':prior_wall+time.perf_counter()-start_time,
                                'eta_seconds':(a.updates-step)*elapsed/(step-last_log_update),
                                'transitions_per_second':(collector.total_steps-last_collection_steps)*a.num_envs/elapsed,
                                'critic_loss_peak':float(matrix[:,keys.index('critic_loss')].amax()),
                                'actor_gradient_norm_peak':float(matrix[:,keys.index('actor_gradient_norm')].amax()),
                                'guard_events_interval':len(new_guards),'guard_events_total':len(collector.guard_events),
                                'guard_envs_interval':sum(len(e['env_ids']) for e in new_guards)}
                progress(latest_metrics,a.updates,a.terminal_format)
                if new_guards:
                    writer.add_text('Diagnostics/collection_guard_events',json.dumps(new_guards,allow_nan=False),step)
                last_guard=len(collector.guard_events);last_collection_steps=collector.total_steps
                if writer is not None:
                    for key,value in latest_metrics.items():
                        if isinstance(value,(int,float)) and not isinstance(value,bool):writer.add_scalar('Train/'+key,value,step)
                    writer.flush()
                pending=[];last_log=time.perf_counter();last_log_update=step
            if step%a.checkpoint_every==0 or step==a.updates:
                saved=save_checkpoint(step)
        atomic_save(saved,a.output/'actor_final.pt')
        summary={'status':'completed','config':config,'audits':checks,
                 'elapsed_seconds':time.perf_counter()-start_time,'collection_steps':collector.total_steps,
                 'resumed_from_update':start_update,'resume_episode_boundary_checked':bool(resume),
                 'final_checkpoint':str(a.output/'actor_final.pt')}
        (a.output/'training.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
        print(f'[SuperMimic] Completed {a.updates} updates. Final policy: {a.output / "actor_final.pt"}',flush=True)
    except Exception as error:
        failure={'status':'failed','error':str(error),'config':config,
                 'elapsed_seconds':time.perf_counter()-start_time}
        if writer is not None and collector is not None:
            writer.add_text('Diagnostics/failure_guard_events',json.dumps(collector.guard_events),collector.total_steps)
        (a.output/'training.json').write_text(json.dumps(failure,indent=2,allow_nan=False))
        raise
    finally:
        env.close()
        if writer is not None:writer.close()
    if a.evaluate:
        from supermimic.evaluation.p4 import evaluate_checkpoint
        evaluate_checkpoint(a.output/'actor_final.pt',a.output/'evaluation.json',a.test_seeds,a.seconds,a.device)


if __name__=='__main__':main()
