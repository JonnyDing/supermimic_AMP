"""Compiled world-model rollout and differentiable AMP/recovery rewards."""
from contextlib import nullcontext
import torch
from .p4 import TASK_KEYS,value_input
from .state import unpack
from .world_model import rollout_loss
from .normalization import TensorAMPNormalizer
from supermimic.mdp.amp import body_features


class FastP4:
    def __init__(self,wm,actor,normalizer,critic,heads,amp,features,
                 compile=False,precision='bf16',horizon=32,gamma=.99):
        self.wm,self.actor,self.normalizer,self.critic=wm,actor,normalizer,critic
        self.heads,self.amp,self.features=heads,amp,features
        self.precision,self.horizon,self.gamma=precision,horizon,gamma
        self.wm.mlp_dtype=torch.bfloat16 if precision=='bf16' else None
        self.critic.mlp_dtype=self.wm.mlp_dtype
        if not isinstance(amp.normalizer,TensorAMPNormalizer):
            amp.normalizer=TensorAMPNormalizer(amp.normalizer,amp.device)
        # Collection/evaluation keep highest precision; training math is set by runner.
        self.step=self._step;self.reward=self._reward;self.wm_loss=self._wm_loss
        if compile:
            # Static body tree and tensor-only inputs avoid per-body Python launches.
            self.step=torch.compile(self._step,fullgraph=True)
            self.reward=torch.compile(self._reward,fullgraph=True)
            self.wm_loss=torch.compile(self._wm_loss,fullgraph=True)

    def autocast(self):
        return torch.autocast('cuda',dtype=torch.bfloat16) if self.precision=='bf16' else nullcontext()

    def _step(self,state,history,previous,context,task):
        command=self.heads.command(state,task)
        obs=self.normalizer(history.flatten(1))
        with self.autocast():
            action=self.actor(obs).float()
        ns,contact=self.wm(state,action,context)
        next_task,delayed,terminated,timeout,cont=self.heads.advance(ns,task)
        next_command=self.heads.command(ns,next_task)
        history=torch.cat((history[:,1:],self.heads.noisy_observation(ns,action,next_command)[:,None]),1)
        return ns,history,action,next_task,command,delayed,timeout,cont,contact

    def _reward(self,states,actions,previous,commands,delayed,timeout,cont,contacts):
        # A single FK on H+1 states supplies both AMP endpoints and all task terms.
        h,b=actions.shape[:2]
        fk=self.heads.kin(unpack(states.flatten(0,1)))
        feature=body_features(*fk,self.features.bodies,self.features.anchor).reshape(h+1,b,-1)
        next_fk=tuple(x.reshape(h+1,b,*x.shape[1:])[1:].flatten(0,1) for x in fk)
        death=torch.where(timeout,torch.zeros_like(cont),1-cont)
        flat=lambda x:x.flatten(0,1)
        terms=self.heads.reward_terms(flat(states[:-1]),flat(states[1:]),flat(actions),flat(previous),flat(commands),
                                     flat(delayed),flat(death),flat(contacts),kinematics=next_fk)
        # Geometry remains FP32 even when MLP matmuls use BF16.
        with self.autocast():
            reward,score=self.amp.reward(flat(feature[:-1]),flat(feature[1:]),self.heads.dt*sum(terms.values()))
        return reward.float().reshape(h,b),score.float().reshape(h,b)

    def _wm_loss(self,window):
        return rollout_loss(self.wm,window)

    def __call__(self,starts):
        state=starts['state'][:,0];history=starts['task/observation_history'][:,0]
        previous=starts['task/previous_action'][:,0];context=starts['context'][:,0]
        task={k:starts['task/'+k][:,0] for k in TASK_KEYS}
        states=[state];actions=[];prevs=[];commands=[];modes=[];timeouts=[];continuations=[];contacts=[];tasks=[task]
        for t in range(self.horizon):
            ns,history,action,next_task,command,delayed,timeout,cont,contact=self.step(state,history,previous,context,task)
            states.append(ns);actions.append(action);prevs.append(previous);commands.append(command)
            modes.append(delayed);timeouts.append(timeout);continuations.append(cont);contacts.append(contact);tasks.append(next_task)
            state,previous,task=ns,action,next_task
        stack=torch.stack
        states,actions,prevs,commands,modes,timeouts,continuations,contacts=map(stack,(states,actions,prevs,commands,modes,timeouts,continuations,contacts))
        rewards,scores=self.reward(states,actions,prevs,commands,modes,timeouts,continuations,contacts)
        alive=torch.cumprod(continuations,0)
        weights=torch.cat((torch.ones_like(alive[:1]),alive[:-1]),0)
        discounts=self.gamma**torch.arange(self.horizon,device=state.device,dtype=state.dtype)[:,None]
        total=(discounts*weights*rewards).sum(0)
        commands_all=torch.cat((commands,self.heads.command(state,task)[None]),0)
        prev_all=torch.cat((prevs,previous[None]),0)
        all_tasks={k:stack([d[k] for d in tasks]).flatten(0,1) for k in TASK_KEYS}
        all_context=context[None].expand(self.horizon+1,-1,-1).flatten(0,1)
        x=value_input(self.wm,states.flatten(0,1),prev_all.flatten(0,1),all_context,commands_all.flatten(0,1),all_tasks,self.heads).reshape(self.horizon+1,len(state),-1)
        bootstrap=self.gamma**self.horizon*alive[-1]*self.critic(x[-1])
        return {'objective':(total+bootstrap).mean(),'task_return':total.mean(),
                'critic_inputs':x,'rewards':rewards,'continuations':continuations,
                'imagined_actions':actions.detach(),
                'recovery_mode_fraction':modes.float().mean(),'style_score':scores.mean()}
