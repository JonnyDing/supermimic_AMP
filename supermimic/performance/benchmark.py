"""Stage-by-stage warmed P4 benchmark; records wall time, not unsynchronized launch time."""
import argparse,copy,json,time,statistics
from pathlib import Path
from types import SimpleNamespace
import torch
from supermimic.adapters.mjlab import CaptureEnv
from supermimic.core.collection import OnlineCollection
from supermimic.core.replay import SequenceReplay
from supermimic.core.world_model import WorldModel,rollout_loss
from supermimic.core.imagination import ValueModel,freeze_parameters,update_critic
from supermimic.core.p4 import sample_starts,imagine
from supermimic.core.policy import ROOT, load_policy
from supermimic.evaluation.world_model import DYNAMICS_KEYS
from supermimic.mdp.amp import AMPFeatures,DifferentiableAMP,DiscriminatorUpdates
from supermimic.mdp.recovery import RecoveryHeads


def fixture(batch=128,num_envs=64):
    device='cuda:0';torch.manual_seed(2040)
    source=ROOT/'logs/supermimic/p4_amp_recovery_20261026'
    ck=torch.load(source/'actor_final.pt',map_location=device,weights_only=False)
    policy,norm=load_policy(source/'actor_final.pt',device)
    teacher=copy.deepcopy(policy.actor).eval()
    for p in teacher.parameters():p.requires_grad_(False)
    wm=WorldModel(**ck['architecture']).to(device);wm.load_state_dict(ck['world_model'])
    critic=ValueModel(wm.input_mean.numel()+11).to(device);critic.load_state_dict(ck['p4_critic'])
    amp=DifferentiableAMP(ck,device)
    from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_env_cfg
    cfg=g1_amp_flat_env_cfg();cfg.scene.num_envs=num_envs;cfg.seed=2040;cfg.events.pop('push_robot',None);cfg.curriculum={};cfg.episode_length_s=5.
    env=CaptureEnv(cfg=cfg,device=device);env.reset(seed=2040)
    heads=RecoveryHeads(env);features=AMPFeatures(env)
    experts=features.experts(ROOT/'src/assets/motions/g1/amp',device)
    discriminator=DiscriminatorUpdates(amp,features,experts)
    replay=SequenceReplay.from_state_dict(torch.load(source/'replay.pt',map_location=device,weights_only=False),device)
    live=SequenceReplay(64,num_envs,device);collector=OnlineCollection(env,policy,norm,live)
    return SimpleNamespace(**locals(),opt_wm=torch.optim.Adam(wm.parameters(),lr=2e-4),
                           opt_actor=torch.optim.Adam(policy.actor.parameters(),lr=1e-7),
                           opt_v=torch.optim.Adam(critic.parameters(),lr=2e-4),train_ids=ck['train_ids'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--engine',choices=['reference','fast'],default='reference')
    p.add_argument('--compile',action='store_true');p.add_argument('--precision',choices=['fp32','tf32','bf16'],default='fp32')
    p.add_argument('--batch',type=int,default=128);p.add_argument('--num-envs',type=int,default=64)
    p.add_argument('--repeats',type=int,default=5);p.add_argument('--warmup',type=int,default=2)
    a=p.parse_args();torch.set_float32_matmul_precision('highest')
    f=fixture(a.batch,a.num_envs);engine=None;rows=[];start=time.perf_counter()
    try:
        if a.engine=='fast':
            from supermimic.core.fast import FastP4
            engine=FastP4(f.wm,f.policy.actor,f.norm,f.critic,f.heads,f.amp,f.features,f.teacher,compile=a.compile,precision=a.precision)
        def timed(row,name,fn):
            torch.cuda.synchronize();t=time.perf_counter();result=fn();torch.cuda.synchronize();row[name]=time.perf_counter()-t;return result
        torch.cuda.reset_peak_memory_stats()
        for i in range(a.warmup+a.repeats):
            row={}
            torch.set_float32_matmul_precision('highest')
            timed(row,'collection',lambda:f.collector.collect(2,i))
            torch.set_float32_matmul_precision('high' if a.precision=='tf32' else 'highest')
            def wm_update():
                for _ in range(2):
                    w=f.replay.sample(a.batch,8,f.train_ids,DYNAMICS_KEYS)
                    loss=(engine.wm_loss(w) if engine else rollout_loss(f.wm,w))[0]
                    f.opt_wm.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(f.wm.parameters(),1.,error_if_nonfinite=True);f.opt_wm.step()
                f.opt_wm.zero_grad(set_to_none=True)
            timed(row,'world_model',wm_update)
            timed(row,'discriminator',lambda:f.discriminator.update(f.replay,f.train_ids,a.batch))
            f.opt_v.zero_grad(set_to_none=True)
            starts=sample_starts(f.replay,f.train_ids,f.heads,a.batch)
            with freeze_parameters(f.wm,f.critic,f.amp.discriminator):
                rollout=timed(row,'actor_forward',lambda:engine(starts) if engine else imagine(f.wm,f.policy.actor,f.norm,f.critic,f.heads,f.amp,f.features,starts,teacher=f.teacher))
                def backward():
                    f.opt_actor.zero_grad(set_to_none=True);(-rollout['objective']).backward()
                    torch.nn.utils.clip_grad_norm_(f.policy.actor.parameters(),1.,error_if_nonfinite=True);f.opt_actor.step()
                timed(row,'actor_backward',backward)
            timed(row,'critic',lambda:update_critic(f.critic,f.opt_v,rollout))
            row['total']=sum(row.values())
            print(json.dumps({'iteration':i,'warmup':i<a.warmup,**row}),flush=True)
            if i>=a.warmup:rows.append(row)
        result={'config':vars(a)|{'output':str(a.output)},'median_seconds':{k:statistics.median(r[k] for r in rows) for k in rows[0]},
                'iterations':rows,'benchmark_seconds_including_compile_and_warmup':time.perf_counter()-start,
                'peak_torch_memory_gib':torch.cuda.max_memory_allocated()/2**30,'gpu':torch.cuda.get_device_name(),
                'note':'Same saved replay, P4 task and optimizer counts; explicit stage synchronization; no claim of policy quality'}
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(result['median_seconds'])
    finally:f.env.close()


if __name__=='__main__':main()
