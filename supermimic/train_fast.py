"""WMRL training: real collection, world-model fitting and imagined policy updates.

Resume restores learning state/RNG, then starts fresh episodes and refills replay.
It does not claim bitwise continuation of simulator internal/contact state.
"""
import argparse,copy,json,time,random,os
from pathlib import Path
import numpy as np
import torch
from supermimic.adapters.mjlab import CaptureEnv
from supermimic.core.collection import OnlineCollection
from supermimic.core.replay import SequenceReplay
from supermimic.core.world_model import WorldModel, DYNAMICS_KEYS
from supermimic.core.imagination import ValueModel,freeze_parameters,update_critic
from supermimic.core.p4 import sample_starts
from supermimic.core.fast import FastP4
from supermimic.core.checkpoint import atomic_save,rng_state,restore_rng,sha256
from supermimic.mdp.amp import AMPFeatures,DifferentiableAMP,DiscriminatorUpdates
from supermimic.mdp.recovery import RecoveryHeads
from supermimic.core.policy import ROOT, load_policy, create_policy
from supermimic.core.bootstrap import fit_initial_statistics, pretrain, exploration_std
from supermimic.core.logging import progress
from supermimic.core.batching import world_model_update, accumulate_imagination, learning_rate

DEFAULTS={'num_envs':4090,'batch_size':1024,'wm_batch_size':2048,'buffer_steps':256,'warmup_steps':64,
          'horizon':32,'utd':.4,'wm_batches':10,'critic_epochs':5,'critic_minibatches':4,
          'wm_pretrain_updates':500,'wm_width':2048,'exploration_initial':.1,'exploration_final':.1,'exploration_decay_updates':5000,
          'wm_microbatch':2048,'actor_microbatch':1024,'lr_decay_updates':250000,
          'gamma':.99,'gae_lambda':.95,'actor_lr':2e-4,'discriminator_lr':1e-5,'precision':'fp32','seed':42,'compile':True}


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    from supermimic.cli import SUPPORTED_TASKS
    p.add_argument('--task', choices=SUPPORTED_TASKS, default=None)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=1000)
    p.add_argument('--resume',type=Path);p.add_argument('--device',default='cuda:0')
    p.add_argument('--checkpoint-every',type=int,default=1000);p.add_argument('--log-every',type=int,default=50)
    p.add_argument('--resume-refill-steps',type=int,default=32)
    p.add_argument('--evaluate',action='store_true',help='Evaluate this policy in real simulation without a baseline checkpoint')
    p.add_argument('--bootstrap-only',action='store_true',help='Run model warmup, save last.pt, then stop before actor training')
    p.add_argument('--test-seeds',nargs='+',type=int,default=None)
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
        if checkpoint.get('fast_schema')!=3 or checkpoint.get('initialization')!='scratch':
            p.error('Resume needs a scratch-training checkpoint (fast_schema=3); legacy policies remain usable for playback/export')
    for key,default in DEFAULTS.items():
        prior=(checkpoint['config'].get(key,default) if checkpoint and key not in
               ('precision','wm_microbatch','actor_microbatch') else default)
        supplied=getattr(a,key)
        if checkpoint and supplied is not None and supplied!=prior and key not in ('wm_microbatch','actor_microbatch','precision','compile'):p.error(f'Resume config mismatch for {key}: {supplied} != {prior}')
        setattr(a,key,prior if supplied is None else supplied)
    previous_task=checkpoint['config'].get('task_id','Unitree-G1-AMP-Flat') if checkpoint else 'Unitree-G1-AMP-Flat'
    if checkpoint and a.task is not None and a.task!=previous_task:p.error('Resume task mismatch')
    a.task=a.task or previous_task
    if a.evaluate and a.task!='Unitree-G1-AMP-Flat':p.error('Fixed recovery evaluation supports AMP Flat only')
    if a.test_seeds is None:a.test_seeds=[a.seed]
    if a.precision not in ('fp32','tf32','bf16'):p.error('precision: fp32, tf32 or bf16')
    if a.num_envs<16 or a.batch_size<8 or a.wm_batch_size<1:p.error('Need >=16 envs, >=8 actor batch, positive WM batch')
    if a.horizon<1 or a.buffer_steps<max(a.horizon,8) or a.warmup_steps<max(a.horizon,8):p.error('Buffer/warmup must fit rollout windows')
    if min(a.updates,a.wm_batches,a.critic_epochs,a.critic_minibatches,a.checkpoint_every,a.log_every)<1 or a.utd<=0:p.error('Counts and UTD must be positive')
    if a.wm_pretrain_updates<1 or a.wm_width<1:p.error('Need positive model warmup and model width')
    if not (0<a.exploration_final<=a.exploration_initial<10) or a.exploration_decay_updates<1:p.error('Require 0 < final exploration <= initial exploration < 10 and positive decay duration')
    if min(a.wm_microbatch,a.actor_microbatch,a.lr_decay_updates)<1:p.error('Microbatches and learning-rate schedule must be positive')
    if checkpoint and a.updates<=checkpoint['update']:p.error('--updates is total target, must exceed saved update')
    return a,checkpoint


def main(argv=None,env_cfg=None,compatibility_config=None):
    a,resume=arguments(argv);a.output.mkdir(parents=True,exist_ok=False)
    os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','4')
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config['resume_semantics']='restore learning state and RNG; start fresh simulator episodes and refill replay'
    config['task_id']=a.task
    learned_task=a.task!='Unitree-G1-AMP-Flat'
    use_amp='-AMP-' in a.task
    config['task_model']='learned observation/reward/termination' if learned_task else 'analytic AMP/recovery'
    config['task']='WMRL scratch; startup DR on, random interval pushes off'
    config['initialization']='scratch; no pretrained policy, world model or teacher'
    config['normalization']='online train-partition actor/WM statistics; independent imagined-input critic statistics; frozen only during BPTT'
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
    ck=resume
    from supermimic.core.task_model import TaskWorldModel, TaskHeads, TaskImagination
    amp=DifferentiableAMP(ck,a.device) if use_amp else None
    if env_cfg is None:
        import src.tasks
        from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
        from supermimic.cli import configure_defaults
        cfg,_=configure_defaults(a.task,load_env_cfg(a.task),load_rl_cfg(a.task))
    else:cfg=copy.deepcopy(env_cfg)
    cfg.seed=a.seed;cfg.scene.num_envs=a.num_envs
    env=CaptureEnv(cfg=cfg,device=a.device);env.learned_task=learned_task
    start_time=time.perf_counter()
    start_update=ck['update'] if resume else 0
    actor_updates=ck.get('actor_updates',ck['update']) if resume else 0
    bootstrap=ck.get('bootstrap',{}) if resume else {}
    writer=None;collector=None
    try:
        if a.logger=='tensorboard':
            from torch.utils.tensorboard import SummaryWriter
            writer=SummaryWriter(log_dir=str(a.output))
        env.reset(seed=a.seed)
        policy,norm=load_policy(a.resume,a.device) if resume else create_policy(
            a.device,env.obs_buf['actor'].shape[-1],env.obs_buf['critic'].shape[-1],env.action_manager.action.shape[-1])
        heads=TaskHeads(env) if learned_task else RecoveryHeads(env)
        features=AMPFeatures(env) if use_amp else None
        experts=features.experts(ROOT/'src/assets/motions/g1/amp',a.device,heads.dt) if use_amp else None
        model_class=TaskWorldModel if learned_task else WorldModel
        if resume:
            wm=model_class(**ck['architecture']).to(a.device);wm.load_state_dict(ck['world_model'])
            replay=(SequenceReplay.from_state_dict(ck['replay'],a.device) if 'replay' in ck
                    else SequenceReplay(a.buffer_steps,a.num_envs,a.device))
            train_ids,holdout=ck['train_ids'],ck['holdout_ids']
        else:
            replay=SequenceReplay(a.buffer_steps,a.num_envs,a.device)
            partition=torch.randperm(a.num_envs,generator=torch.Generator().manual_seed(a.seed)).tolist()
            train_ids,holdout=partition[:a.num_envs*3//4],partition[a.num_envs*3//4:]
        collector=OnlineCollection(env,policy,norm,replay,action_noise=exploration_std(
            start_update,a.exploration_initial,a.exploration_final,a.exploration_decay_updates),statistics_ids=train_ids)
        print(f'[SuperMimic] Collecting initial replay: {a.num_envs} envs; log directory: {a.output}',flush=True)
        if resume:
            # Network construction consumes RNG; restore only once construction is complete below.
            collector.total_steps=ck['collection_steps']
        else:
            collector.collect(a.warmup_steps,0)
            extra=dict(frame_dim=env.obs_buf['actor'].shape[-1]//cfg.observations['actor'].history_length,
                       history_length=cfg.observations['actor'].history_length) if learned_task else {}
            wm=model_class(replay.storage['action'].shape[-1],replay.storage['context'].shape[-1],
                          hidden=(a.wm_width,)*3,statistics_mode='running',**extra).to(a.device)
            fit_initial_statistics(wm,norm,replay,train_ids)
        collector.world_model=wm
        critic=ValueModel(wm.input_mean.numel()+(0 if learned_task else 11),normalize=True).to(a.device)
        engine_class=TaskImagination if learned_task else FastP4
        engine=engine_class(wm,policy.actor,norm,critic,heads,amp,features,compile=a.compile,precision=a.precision,horizon=a.horizon,gamma=a.gamma)
        if learned_task:
            names=env.observation_manager._group_obs_term_names['actor']
            sizes=env.observation_manager._group_obs_term_dim['actor']
            phase_cfg=cfg.observations['actor'].terms.get('phase')
            engine.phase_period=phase_cfg.params['period'] if phase_cfg is not None else None
            engine.action_start=sum(int(np.prod(d)) for d in sizes[:names.index('actions')])//cfg.observations['actor'].history_length
        discriminator=DiscriminatorUpdates(amp,features,experts,lr=a.discriminator_lr) if use_amp else None
        opt_wm=torch.optim.Adam(wm.parameters(),lr=2e-4)
        opt_actor=torch.optim.Adam(policy.actor.parameters(),lr=a.actor_lr,betas=(.9,.95))
        opt_v=torch.optim.Adam(critic.parameters(),lr=2e-4)
        if resume:
            critic.load_state_dict(ck['p4_critic'])
            opt_wm.load_state_dict(ck['wm_optimizer']);opt_actor.load_state_dict(ck['actor_optimizer'])
            opt_v.load_state_dict(ck['critic_optimizer'])
            if use_amp:
                discriminator.optimizer.load_state_dict(ck['discriminator_optimizer'])
                experts.cursor=ck['expert_cursor'];discriminator.updates=ck['discriminator_updates']
            restore_rng(ck['rng'])
            collector.collect(max(a.resume_refill_steps,a.horizon,a.warmup_steps if 'replay' not in ck else 0),start_update)
        else:
            print('[SuperMimic] Scratch initialization: random networks, no teacher; pretraining world model'+(' and AMP' if use_amp else ''),flush=True)
            bootstrap=pretrain(wm,opt_wm,discriminator,collector,train_ids,holdout,
                               a.wm_pretrain_updates,a.wm_batch_size,a.log_every,writer,a.wm_batches,a.wm_microbatch,a.utd,
                               loss_fn=engine.wm_loss,precision=a.precision)
        if not resume:
            start_update=a.wm_pretrain_updates
            if not a.bootstrap_only and a.updates<=start_update:
                raise ValueError('Total iterations must exceed WM-only startup iterations')
        accumulator=ck.get('collection_accumulator',0.) if resume else 0.
        prior_wall=ck.get('training_wall_seconds',0.) if resume else 0.
        latest_metrics={};pending=[];last_log=time.perf_counter();last_log_update=start_update
        last_collection_steps=collector.total_steps
        print(f'[SuperMimic] Training {start_update + 1}..{a.updates}; first update may compile kernels',flush=True)
        print(f'[Stats] Metrics are interval means (up to {a.log_every} updates); peak fields are interval maxima.',flush=True)
        def save_checkpoint(step):
            save_started=time.perf_counter()
            print(f'[Checkpoint] Saving update {step}: networks, statistics and optimizer state...',flush=True)
            payload={'fast_schema':3,'schema_version':2,'initialization':'scratch','bootstrap':bootstrap,'config':config,'update':step,'actor_updates':actor_updates,
                     'model_state_dict':policy.state_dict(),'obs_norm_state_dict':norm.state_dict(),
                     'world_model':wm.state_dict(),'architecture':wm.architecture(),'p4_critic':critic.state_dict(),
                     'amp_normalization':amp.normalization_mode if use_amp else None,'discriminator_updates':discriminator.updates if use_amp else 0,
                     'discriminator_state_dict':amp.discriminator.state_dict() if use_amp else None,
                     'amp_normalizer':amp.normalizer.native() if use_amp else None,
                     'wm_optimizer':opt_wm.state_dict(),'actor_optimizer':opt_actor.state_dict(),
                     'critic_optimizer':opt_v.state_dict(),'discriminator_optimizer':discriminator.optimizer.state_dict() if use_amp else None,
                     'expert_cursor':experts.cursor if use_amp else 0,'rng':rng_state(),'train_ids':train_ids,'holdout_ids':holdout,
                     'collection_steps':collector.total_steps,'collection_accumulator':accumulator,
                     'training_wall_seconds':prior_wall+time.perf_counter()-start_time,
                     'compatibility_env_signature':env_signature,
                     'sources':{}}
            atomic_save(payload,a.output/f'{a.checkpoint_prefix}_{step}.pt')
            atomic_save(payload,a.output/'last.pt')
            print(f'[Checkpoint] Saved {a.checkpoint_prefix}_{step}.pt and last.pt in {time.perf_counter()-save_started:.1f}s',flush=True)
            return payload
        if a.bootstrap_only:
            save_checkpoint(start_update)
            summary={'status':'bootstrap_completed','bootstrap':bootstrap,'actor_updates':actor_updates,
                     'collection_steps':collector.total_steps,'elapsed_seconds':time.perf_counter()-start_time,
                     'resume_checkpoint':str(a.output/'last.pt')}
            (a.output/'training.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
            print(f'[SuperMimic] Bootstrap completed; resume checkpoint: {a.output / "last.pt"}',flush=True)
            return
        for step in range(start_update+1,a.updates+1):
            rate=learning_rate(a.actor_lr,step-1,a.lr_decay_updates)
            for group in opt_actor.param_groups:group['lr']=rate
            for opt in (opt_wm,opt_v):
                for group in opt.param_groups:group['lr']=learning_rate(2e-4,step-1,a.lr_decay_updates)
            accumulator+=1/a.utd;collect_steps=int(accumulator);accumulator-=collect_steps
            torch.set_float32_matmul_precision('highest')
            collector.action_noise=exploration_std(step-1,a.exploration_initial,a.exploration_final,a.exploration_decay_updates)
            collector.collect(collect_steps,step-1)
            torch.set_float32_matmul_precision('high' if a.precision in ('tf32','bf16') else 'highest')
            losses=[]
            for _ in range(a.wm_batches):
                window=replay.sample_prefix(a.wm_batch_size,8,train_ids,getattr(wm,'replay_keys',DYNAMICS_KEYS))
                loss=world_model_update(wm,opt_wm,window,engine.wm_loss,a.wm_microbatch)
                losses.append(loss)
            metrics=discriminator.update(replay,train_ids,a.batch_size,return_tensors=True) if use_amp else {}
            opt_wm.zero_grad(set_to_none=True);opt_v.zero_grad(set_to_none=True)
            starts=sample_starts(replay,train_ids,heads,a.batch_size,a.horizon)
            with freeze_parameters(wm,critic,*([amp.discriminator] if use_amp else [])):
                opt_actor.zero_grad(set_to_none=True)
                rollout=accumulate_imagination(engine,starts,policy.actor,a.actor_microbatch)
                grad=torch.nn.utils.clip_grad_norm_(policy.actor.parameters(),1.)
                torch.nn.utils.clip_grad_value_(policy.actor.parameters(),.1);opt_actor.step()
                actor_updates+=1
            vloss,critic_metrics=update_critic(critic,opt_v,rollout,a.critic_epochs,a.critic_minibatches,return_tensor=True,gamma=a.gamma,lam=a.gae_lambda,return_metrics=True)
            metrics.update(critic_metrics)
            metrics.update(critic_norm_count=critic.norm.count.detach().float(),actor_lr=torch.tensor(rate,device=a.device),observation_norm_count=norm.count.detach().float(),wm_norm_count=wm.input_norm.count.detach().float(),exploration_std=torch.tensor(collector.action_noise,device=a.device),wm_loss=torch.stack(losses).mean(),critic_loss=vloss,actor_gradient_norm=grad.detach(),
                           imagined_return=rollout['task_return'].detach(),recovery_mode_fraction=rollout['recovery_mode_fraction'].detach())
            pending.append(metrics)
            if step%a.log_every==0 or step==a.updates:
                keys=list(metrics);matrix=torch.stack([torch.stack([r[k] for k in keys]) for r in pending])
                values=matrix.mean(0)
                for i,key in enumerate(keys):
                    if key.endswith('_abs_max'):values[i]=matrix[:,i].amax()
                values=values.cpu().tolist()
                elapsed=time.perf_counter()-last_log
                latest_metrics={'update':step,**dict(zip(keys,values)),'interval_seconds':elapsed,
                                'updates_per_second':(step-last_log_update)/elapsed,'collection_steps':collector.total_steps,
                                'transitions':collector.total_steps*a.num_envs,'first_interval_includes_compile':bool(a.compile and last_log_update==start_update),
                                'interval_updates':step-last_log_update,'elapsed_seconds':prior_wall+time.perf_counter()-start_time,
                                'eta_seconds':(a.updates-step)*elapsed/(step-last_log_update),
                                'transitions_per_second':(collector.total_steps-last_collection_steps)*a.num_envs/elapsed,
                                'critic_loss_peak':float(matrix[:,keys.index('critic_loss')].amax()),
                                'actor_gradient_norm_peak':float(matrix[:,keys.index('actor_gradient_norm')].amax())}
                progress(latest_metrics,a.updates,a.terminal_format)
                last_collection_steps=collector.total_steps
                if writer is not None:
                    for key,value in latest_metrics.items():
                        if isinstance(value,(int,float)) and not isinstance(value,bool):writer.add_scalar('Train/'+key,value,step)
                    writer.flush()
                pending=[];last_log=time.perf_counter();last_log_update=step
            if step%a.checkpoint_every==0 or step==a.updates:
                saved=save_checkpoint(step)
        atomic_save(saved,a.output/'actor_final.pt')
        summary={'status':'completed','initialization':'scratch','bootstrap':bootstrap,'config':config,
                 'elapsed_seconds':time.perf_counter()-start_time,'collection_steps':collector.total_steps,
                 'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(a.device) if a.device.startswith('cuda') else 0,
                 'resumed_from_update':ck['update'] if resume else 0,'actor_updates':actor_updates,'resume_episode_boundary_checked':bool(resume),
                 'final_checkpoint':str(a.output/'actor_final.pt')}
        (a.output/'training.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
        print(f'[SuperMimic] Completed {a.updates} updates. Final policy: {a.output / "actor_final.pt"}',flush=True)
    except Exception as error:
        failure={'status':'failed','error':str(error),'config':config,
                 'elapsed_seconds':time.perf_counter()-start_time}
        (a.output/'training.json').write_text(json.dumps(failure,indent=2,allow_nan=False))
        raise
    finally:
        env.close()
        if writer is not None:writer.close()
    if a.evaluate:
        from supermimic.evaluation.scratch import evaluate_checkpoint
        evaluate_checkpoint(a.output/'actor_final.pt',a.output/'evaluation.json',a.test_seeds,a.seconds,a.device)


if __name__=='__main__':main()
