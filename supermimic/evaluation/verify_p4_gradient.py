"""Finite-difference end-to-end P4 actor gradient on saved real replay starts."""
import argparse,json
from pathlib import Path
import torch
from supermimic.adapters.mjlab import CaptureEnv
from supermimic.core.world_model import WorldModel
from supermimic.core.replay import SequenceReplay
from supermimic.core.imagination import ValueModel,freeze_parameters
from supermimic.core.p4 import sample_starts,imagine
from supermimic.core.online import snapshot,same_state
from supermimic.core.policy import load_policy
from supermimic.mdp.amp import AMPFeatures,DifferentiableAMP
from supermimic.mdp.recovery import RecoveryHeads


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.manual_seed(90210);torch.set_float32_matmul_precision('highest');device='cuda:0'
    ck=torch.load(a.run/'actor_final.pt',map_location=device,weights_only=False)
    policy,norm=load_policy(a.run/'actor_final.pt',device)
    wm=WorldModel(**ck['architecture']).to(device);wm.load_state_dict(ck['world_model'])
    amp=DifferentiableAMP(ck,device)
    critic=ValueModel(wm.input_mean.numel()+11).to(device);critic.load_state_dict(ck['p4_critic'])
    replay=SequenceReplay.from_state_dict(torch.load(a.run/'replay.pt',map_location=device,weights_only=False),device)
    from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_env_cfg
    cfg=g1_amp_flat_env_cfg();cfg.scene.num_envs=4;cfg.events.pop('push_robot',None);cfg.episode_length_s=5.
    env=CaptureEnv(cfg=cfg,device=device)
    try:
        env.reset(seed=90210);heads=RecoveryHeads(env);features=AMPFeatures(env)
        starts=sample_starts(replay,ck['train_ids'],heads,batch=16,horizon=2)
        parameter=policy.actor[-1].bias;original=parameter.detach().clone()
        direction=torch.randn_like(parameter);direction/=direction.norm()
        before=[snapshot(m) for m in (wm,critic,amp.discriminator)]
        with freeze_parameters(wm,critic,amp.discriminator):
            objective=imagine(wm,policy.actor,norm,critic,heads,amp,features,starts,horizon=2)['objective']
            gradient=torch.autograd.grad(objective,parameter)[0]
            derivative=float((gradient*direction).sum());eps=.002
            try:
                with torch.no_grad():
                    parameter.copy_(original+eps*direction)
                    plus=float(imagine(wm,policy.actor,norm,critic,heads,amp,features,starts,horizon=2)['objective'])
                    parameter.copy_(original-eps*direction)
                    minus=float(imagine(wm,policy.actor,norm,critic,heads,amp,features,starts,horizon=2)['objective'])
            finally:
                with torch.no_grad():parameter.copy_(original)
        finite=(plus-minus)/(2*eps)
        assert abs(derivative-finite)<1e-4+.05*abs(derivative),(derivative,finite)
        assert gradient.norm()>0
        assert all(same_state(b,m) for b,m in zip(before,(wm,critic,amp.discriminator)))
        result={'status':'passed','autograd':derivative,'finite_difference':finite,'absolute_error':abs(derivative-finite),
                'actor_gradient_norm':float(gradient.norm()),'horizon':2,'states':16,
                'recovery_starts':int(starts['task/recovery_enabled'].sum()),
                'note':'Teacher stop-gradient anchor excluded; AMP, recovery task rewards and learned critic bootstrap included'}
        a.output.write_text(json.dumps(result,indent=2));print(result)
    finally:env.close()


if __name__=='__main__':main()
