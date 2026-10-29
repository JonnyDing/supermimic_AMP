"""Differentiable AMP reward and independently trained motion discriminator."""
import copy
from pathlib import Path
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_conjugate, quat_mul, matrix_from_quat
from rsl_rl.modules.discriminator import Discriminator
from rsl_rl.utils.utils import Normalizer
from supermimic.core.state import unpack
from supermimic.mdp.kinematics import RobotKinematics


def body_features(positions, rotations, velocities, angular, bodies, anchor):
    q = rotations[:, bodies]
    anchor_q = rotations[:, anchor:anchor+1].expand_as(q)
    relative = quat_apply_inverse(anchor_q, positions[:, bodies]-positions[:, anchor:anchor+1])
    orientation = matrix_from_quat(quat_mul(quat_conjugate(anchor_q), q))[..., :2]
    return torch.cat([relative.flatten(1), orientation.flatten(1),
                      quat_apply_inverse(q, velocities[:, bodies]).flatten(1),
                      quat_apply_inverse(q, angular[:, bodies]).flatten(1)], -1)


class AMPFeatures:
    def __init__(self, env):
        from src.tasks.amp_loco.config.g1.rl_cfg import g1_wmrl_runner_cfg
        cfg = g1_wmrl_runner_cfg()
        self.kin = RobotKinematics(env)
        self.names = list(env.scene['robot'].body_names)
        self.bodies = [self.names.index(n) for n in cfg.amp_body_names]
        self.anchor = self.names.index(cfg.amp_anchor_name)

    def __call__(self, state):
        return body_features(*self.kin(unpack(state)), self.bodies, self.anchor)

    def experts(self, directory, device, dt=.02):
        clips=[]; paths=[]
        for path in sorted(Path(directory).rglob('*.npz')):
            with np.load(path) as data:
                if abs(float(data['fps'].item())*dt-1)>1e-6:
                    raise ValueError(f'Expert interval differs from control interval: {path}')
                tensors=[torch.as_tensor(data[k],device=device,dtype=torch.float32) for k in
                         ('body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w')]
                if tensors[0].shape[1] != len(self.names): raise ValueError('Expert body ordering/shape mismatch')
                features=body_features(*tensors,self.bodies,self.anchor)
                if len(features)<2: raise ValueError(f'Short expert clip: {path}')
                clips.append(features); paths.append(str(path))
        if not clips: raise ValueError('No expert clips')
        return ExpertPairs(clips,paths)


class ExpertPairs:
    def __init__(self, clips, paths):
        self.clips,self.paths,self.cursor=clips,paths,0

    def sample(self, batch):
        # Match native round-robin motion weighting; never make a cross-clip pair.
        clip=self.clips[self.cursor%len(self.clips)]; self.cursor+=1
        ids=torch.randint(len(clip)-1,(batch,),device=clip.device)
        return clip[ids],clip[ids+1]


class DifferentiableAMP:
    def __init__(self, checkpoint, device):
        self.discriminator=Discriminator(390,.1,[1024,512,256],device,task_reward_lerp=.75)
        self.normalizer = Normalizer(195)
        self.normalization_mode = 'raw' if checkpoint is None else checkpoint.get('amp_normalization', 'legacy_normalized')
        if checkpoint is not None:
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.normalizer=copy.deepcopy(checkpoint['amp_normalizer'])
        self.device=device

    def reward(self, state, next_state, task_reward):
        s=self.normalizer.normalize_torch(state,self.device)
        ns=self.normalizer.normalize_torch(next_state,self.device)
        d=self.discriminator(torch.cat((s,ns),-1)).squeeze(-1).float()
        style=self.discriminator.amp_reward_coef*(1-.25*(d-1).square()).clamp_min(0)
        lerp=self.discriminator.task_reward_lerp
        reward=(1-lerp)*style+lerp*task_reward if lerp>0 else style
        return reward,d


class DiscriminatorUpdates:
    def __init__(self, amp, features, experts, lr=1e-5):
        self.amp,self.features,self.experts=amp,features,experts
        d=amp.discriminator
        self.optimizer=torch.optim.Adam([{'params':d.trunk.parameters(),'weight_decay':1e-3},
                                         {'params':d.amp_linear.parameters(),'weight_decay':.1}],lr=lr)
        self.updates=0

    def update(self, replay, train_ids, batch=128, return_tensors=False):
        w=replay.sample(batch,1,train_ids,['state','next_state'])
        with torch.no_grad():
            real=self.features(w['state'][:,0]); next_real=self.features(w['next_state'][:,0])
            expert,next_expert=self.experts.sample(batch)
            norm=self.amp.normalizer
            s,ns,e,ne=[norm.normalize_torch(x,self.amp.device) for x in (real,next_real,expert,next_expert)]
        d=self.amp.discriminator; d.train()
        for p in d.parameters(): p.requires_grad_(True)
        policy_score=d(torch.cat((s,ns),-1)); expert_score=d(torch.cat((e,ne),-1))
        classification=.5*((expert_score-1).square().mean()+(policy_score+1).square().mean())
        # Native AMP PPO applies its gradient penalty to RAW expert features.
        penalty=d.compute_grad_pen(expert.detach(),next_expert.detach(),lambda_=10)
        self.optimizer.zero_grad(set_to_none=True)
        (classification+penalty).backward()
        torch.nn.utils.clip_grad_norm_(d.parameters(),1.)
        self.optimizer.step(); self.optimizer.zero_grad(set_to_none=True)
        # Scratch runs estimate raw feature statistics; old checkpoints retain their convention.
        samples = (real, next_real, expert, next_expert) if self.amp.normalization_mode == 'raw' else (s, e)
        if hasattr(norm,'update_tensor'):
            for x in samples: norm.update_tensor(x)
        else:
            for x in samples: norm.update(x.cpu().numpy())
        self.updates+=1
        metrics={'discriminator_loss':classification.detach(),'gradient_penalty':penalty.detach(),
                 'policy_score':policy_score.detach().mean(),'expert_score':expert_score.detach().mean()}
        return metrics if return_tensors else {k:float(v) for k,v in metrics.items()}
