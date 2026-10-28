"""GPU parity probe: target AMP features/reward, full task terms and gradients."""
import argparse,json
from pathlib import Path
import torch
from supermimic.adapters.mjlab import CaptureEnv,read_physics,read_task_metadata,read_contact_labels
from supermimic.core.state import pack
from supermimic.core.imagination import freeze_parameters
from supermimic.core.policy import ROOT
from supermimic.mdp.amp import AMPFeatures,DifferentiableAMP
from supermimic.mdp.recovery import RecoveryHeads


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_float32_matmul_precision('highest');torch.manual_seed(2030)
    from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_env_cfg
    cfg=g1_amp_flat_env_cfg();cfg.scene.num_envs=8;cfg.events.pop('push_robot',None)
    cfg.observations['actor'].enable_corruption=False
    env=CaptureEnv(cfg=cfg,device='cuda:0')
    ck=torch.load(ROOT/'logs/rsl_rl/g1_amp_locomotion/2026-10-24_13-24-33/model_66600.pt',map_location='cuda:0',weights_only=False)
    amp=DifferentiableAMP(ck,'cuda:0')
    try:
        env.reset(seed=2030);features=AMPFeatures(env);heads=RecoveryHeads(env)
        errors={'features':0.,'task_reward':0.,'amp_reward':0.,'heading_command':0.};per_term={}
        for t in range(6):
            if t:env.step(torch.randn(8,29,device=env.device)*.05)
            env.sim.forward();env.sim.sense()
            state=pack(read_physics(env)); robot=env.scene['robot']
            native=torch.cat([term.func(env,**term.params) for term in env.observation_manager._group_obs_term_cfgs['amp']],-1)
            errors['features']=max(errors['features'],float((features(state)-native).abs().max()))
            # Exercise both masked and unmasked reward modes at matched physical state.
            tm=env.termination_manager;tm._delay_env_mask[:4]=True;tm._delay_counters[:4]=t%2
            task=read_task_metadata(env);command=env.command_manager.get_term('twist')
            command._update_command();task['command']=command.command.clone()
            errors['heading_command']=max(errors['heading_command'],float((heads.command(state,task)-command.command).abs().max()))
            labels=read_contact_labels(env)
            actual=heads.reward_terms(state,state,env.action_manager.action,env.action_manager.prev_action,command.command,
                                     tm._delay_env_mask&(tm._delay_counters>0),tm.terminated.float(),torch.zeros(8,3,device=env.device),
                                     joint_acceleration=robot.data.joint_acc,feet_contact=labels['feet_in_contact'],self_collision=labels['self_collision_raw'])
            expected={}
            mapping={'tracking':['track_anchor_linear_velocity','track_anchor_angular_velocity'],
                     'root_height':['track_root_height'],'body_ang_vel_xy':['body_ang_vel_xy_l2'],
                     'termination':['is_terminated'],'joint_acceleration':['joint_acc_l2'],
                     'joint_limits':['joint_pos_limits'],'action_rate':['action_rate_l2'],
                     'foot_slip':['foot_slip'],'self_collision':['self_collisions']}
            for key,names in mapping.items():
                expected[key]=sum(env.reward_manager.get_term_cfg(n).weight*env.reward_manager.get_term_cfg(n).func(env,**env.reward_manager.get_term_cfg(n).params) for n in names)
                per_term[key]=max(per_term.get(key,0.),float(((actual[key]-expected[key])*env.step_dt).abs().max()))
            task_reward=env.step_dt*sum(actual.values());reference=env.step_dt*sum(expected.values())
            errors['task_reward']=max(errors['task_reward'],float((task_reward-reference).abs().max()))
            reward,_=amp.reward(features(state),features(state),task_reward)
            native_reward,_=amp.discriminator.predict_amp_reward(features(state),features(state),task_reward,amp.normalizer)
            errors['amp_reward']=max(errors['amp_reward'],float((reward-native_reward).abs().max()))
        # Isolate analytic input derivative and enforce frozen discriminator parameters.
        x=features(state).detach().requires_grad_();direction=torch.randn_like(x);direction/=direction.norm()
        with freeze_parameters(amp.discriminator):
            reward,_=amp.reward(x,x,torch.zeros(8,device=env.device))
            grad=torch.autograd.grad(reward.sum(),x)[0];ad=float((grad*direction).sum())
            eps=.01
            with torch.no_grad():
                plus=amp.reward(x+eps*direction,x+eps*direction,torch.zeros(8,device=env.device))[0].sum()
                minus=amp.reward(x-eps*direction,x-eps*direction,torch.zeros(8,device=env.device))[0].sum()
            fd=float((plus-minus)/(2*eps))
        assert errors['features']<1e-4,errors
        assert max(errors[k] for k in ('task_reward','amp_reward','heading_command'))<1e-5,errors
        assert grad.norm()>0 and abs(ad-fd)<2e-5+abs(ad)*.05,(ad,fd)
        assert all(p.grad is None for p in amp.discriminator.parameters())
        result={'status':'passed','max_errors':errors,'per_term_reward_errors':per_term,
                'amp_input_gradient_norm':float(grad.norm()),'directional_autograd':ad,'directional_finite_difference':fd,
                'comparison':'matched post-forward state; true contacts and native joint_acc override proxies',
                'states':48}
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(result)
    finally:env.close()


if __name__=='__main__':main()
