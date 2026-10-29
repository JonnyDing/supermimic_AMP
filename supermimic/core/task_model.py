"""SuperMimic task models for velocity and terrain-aware locomotion.

The original environment remains the reward/termination authority. Supervised
heads learn those signals and the next actor frame from real transitions; actor
updates differentiate imagined returns through the frozen learned model. Terrain
scans and gait phase remain in the actor history and dynamics conditioning.
"""
from contextlib import nullcontext

import torch
from torch import nn

from .world_model import WorldModel, DYNAMICS_KEYS, contact_targets
from .normalization import RunningNorm
from .state import delta, integrate, unpack
from .p4 import TASK_KEYS
from supermimic.mdp.recovery import RecoveryHeads


TASK_MODEL_KEYS = DYNAMICS_KEYS + ['next_task/observation_history', 'reward',
                                  'terminated', 'truncated', 'next_task/recovery_counter']


def task_features(task, max_episode_steps):
    return torch.stack((task['recovery_enabled'].float(), task['recovery_counter']/250.,
                        task['episode_step']/max_episode_steps, task['command_time_left']/8.,
                        task['heading_target'].sin(), task['heading_target'].cos(),
                        task['is_heading_env'].float(), task['is_standing_env'].float()), -1)


def task_context(physics_context, history, task, max_episode_steps):
    return torch.cat((physics_context, history.flatten(-2), task_features(task, max_episode_steps)), -1)


class TaskWorldModel(WorldModel):
    replay_keys = TASK_MODEL_KEYS

    def __init__(self, *args, frame_dim, history_length=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_dim, self.history_length = frame_dim, history_length
        self.task_head = nn.Linear(self.hidden[-1], frame_dim + 3)
        nn.init.zeros_(self.task_head.weight)
        nn.init.zeros_(self.task_head.bias)
        self.frame_norm = RunningNorm(frame_dim)
        self.reward_norm = RunningNorm(1)

    @torch.no_grad()
    def update_task_stats(self, next_history, reward, valid):
        self.frame_norm.update(next_history[..., -1, :][valid])
        self.reward_norm.update(reward.unsqueeze(-1))

    def predict(self, state, action, context):
        x = (self.inputs(state, action, context)-self.input_mean)/self.input_std
        dtype = getattr(self, 'mlp_dtype', None)
        with torch.autocast('cuda', dtype=dtype) if dtype is not None else nullcontext():
            hidden = self.trunk(x)
            increments, contacts, task = self.dynamics(hidden), self.contact(hidden), self.task_head(hidden)
        next_state = integrate(state, increments.float()*self.delta_std+self.delta_mean)
        task = task.float()
        frame = task[..., :self.frame_dim]*self.frame_norm.std()+self.frame_norm.mean
        reward = task[..., self.frame_dim]*self.reward_norm.std()[0]+self.reward_norm.mean[0]
        return next_state, contacts.float(), frame, reward, task[..., self.frame_dim+1], task[..., self.frame_dim+2]*250.

    def forward(self, state, action, context):
        return self.predict(state, action, context)[:2]

    def architecture(self):
        return dict(super().architecture(), frame_dim=self.frame_dim, history_length=self.history_length)


def task_rollout_loss(model, window):
    """Eight-step physics loss plus supervised observation/reward/survival heads.

    Observation targets at episode boundaries are excluded. Rewards and terminal
    flags remain valid on the terminal transition; truncation is not a death.
    """
    state = window['state'][:, 0]
    targets = contact_targets(window)
    mask = window.get('_mask', torch.ones_like(window['terminated']))
    physics, contact, auxiliary = [], [], []
    for t in range(window['action'].shape[1]):
        state = torch.where(mask[:, t, None], state, window['state'][:, 0])
        state, logits, frame, reward, death, counter = model.predict(state, window['action'][:, t], window['context'][:, t])
        weight = mask[:, t].float()
        error = delta(window['next_state'][:, t], state)/model.error_scale()
        physics.append((error.square().mean(-1)*weight).mean())
        contact.append((nn.functional.binary_cross_entropy_with_logits(
            logits, targets[:, t], reduction='none').mean(-1)*weight).mean())
        obs_error = ((frame-window['next_task/observation_history'][:, t, -1])/model.frame_norm.std()).square().mean(-1)
        observation_valid = (~(window['terminated'][:, t] | window['truncated'][:, t])).float()*weight
        obs_loss = (obs_error*observation_valid).sum()/observation_valid.sum().clamp_min(1)
        reward_error = ((reward-window['reward'][:, t])/model.reward_norm.std()[0]).square()
        death_loss = nn.functional.binary_cross_entropy_with_logits(death, window['terminated'][:, t].float(), reduction='none')
        counter_error = ((counter-window['next_task/recovery_counter'][:, t])/250.).square()
        auxiliary.append(obs_loss+((reward_error+death_loss+counter_error)*weight).mean())
    p, c, a = map(lambda xs: torch.stack(xs).mean(), (physics, contact, auxiliary))
    return p+.1*c+a, p.detach(), a.detach()


class TaskHeads:
    """Command evolution is analytic; task-specific survival is learned."""
    command = RecoveryHeads.command

    def __init__(self, env):
        self.dt = env.step_dt
        self.maximum = getattr(env.termination_manager, '_max_delay_steps', 0)
        self.max_episode_steps = env.max_episode_length
        command = env.command_manager.get_term('twist').cfg
        self.heading_stiffness = command.heading_control_stiffness
        self.yaw_range = command.ranges.ang_vel_z


class TaskImagination:
    def __init__(self, wm, actor, normalizer, critic, heads, amp=None, features=None,
                 compile=False, precision='bf16', horizon=32, gamma=.99):
        self.wm, self.actor, self.normalizer, self.critic = wm, actor, normalizer, critic
        self.heads, self.amp, self.features = heads, amp, features
        self.precision, self.horizon, self.gamma = precision, horizon, gamma
        self.phase_period = None
        wm.mlp_dtype = torch.bfloat16 if precision=='bf16' else None
        critic.mlp_dtype = wm.mlp_dtype
        if amp is not None:
            from .normalization import TensorAMPNormalizer
            amp.normalizer = TensorAMPNormalizer(amp.normalizer, amp.device)
        self.step = torch.compile(self._step, fullgraph=True) if compile else self._step
        self.wm_loss = torch.compile(self._wm_loss, fullgraph=True) if compile else self._wm_loss
        self.style_reward = torch.compile(self._style_reward, fullgraph=True) if compile and amp is not None else self._style_reward

    def _wm_loss(self, window):
        return task_rollout_loss(self.wm, window)

    def value_input(self, state, previous, context):
        return (self.wm.inputs(state, previous, context)-self.wm.input_mean)/self.wm.input_std

    def _step(self, state, history, previous, physics_context, task):
        context = task_context(physics_context, history, task, self.heads.max_episode_steps)
        with torch.autocast('cuda', dtype=torch.bfloat16) if self.precision=='bf16' else nullcontext():
            action = self.actor(self.normalizer(history.flatten(1))).float()
        ns, _, frame, reward, death, counter = self.wm.predict(state, action, context)
        next_task = {**task, 'episode_step':task['episode_step']+1,
                     'command_time_left':task['command_time_left']-self.heads.dt,
                     'recovery_counter':counter}
        command = self.heads.command(ns, next_task)
        # Command and executed-action channels have exact semantics, even when
        # predicted terrain/proprioceptive observations are still inaccurate.
        frame = torch.cat((frame[:, :6], command, frame[:, 9:]), -1)
        # Height scans follow actions; the action offset is supplied by schema.
        action_start = self.action_start
        if self.phase_period is not None:
            angle=next_task['episode_step']*self.heads.dt/self.phase_period*(2*torch.pi)
            phase=torch.stack((angle.sin(),angle.cos()),-1)
            phase=torch.where(command.norm(dim=-1,keepdim=True)<.1,torch.zeros_like(phase),phase)
            frame=torch.cat((frame[:,:9],phase,frame[:,11:]),-1)
        frame = torch.cat((frame[:, :action_start], action, frame[:, action_start+29:]), -1)
        history = torch.cat((history[:, 1:], frame[:, None]), 1)
        timeout = next_task['episode_step'] >= self.heads.max_episode_steps
        cont = torch.where(timeout, torch.zeros_like(death), torch.sigmoid(-death))
        return ns, history, action, next_task, reward.float(), cont.float()

    def _style_reward(self, states, rewards):
        from supermimic.mdp.amp import body_features
        h,b=rewards.shape
        # One FK over H+1 states supplies both endpoints of every AMP pair.
        features=body_features(*self.features.kin(unpack(states.flatten(0,1))),
                               self.features.bodies,self.features.anchor).reshape(h+1,b,-1)
        with torch.autocast('cuda', dtype=torch.bfloat16) if self.precision=='bf16' else nullcontext():
            reward,score=self.amp.reward(features[:-1].flatten(0,1),features[1:].flatten(0,1),rewards.flatten())
        return reward.float().reshape(h,b),score.float().reshape(h,b)

    def __call__(self, starts):
        state, history = starts['state'][:, 0], starts['task/observation_history'][:, 0]
        previous = starts['task/previous_action'][:, 0]
        task = {k:starts['task/'+k][:, 0] for k in TASK_KEYS}
        dynamic_dim = history.shape[1]*history.shape[2]+8
        physics_context = starts['context'][:, 0, :-dynamic_dim]
        values, rewards, continuations, actions = [], [], [], []
        states = [state]
        for _ in range(self.horizon):
            context = task_context(physics_context, history, task, self.heads.max_episode_steps)
            values.append(self.value_input(state, previous, context))
            state, history, previous, task, reward, cont = self.step(state, history, previous, physics_context, task)
            rewards.append(reward); continuations.append(cont); actions.append(previous); states.append(state)
        context = task_context(physics_context, history, task, self.heads.max_episode_steps)
        values.append(self.value_input(state, previous, context))
        x, rewards, cont = map(torch.stack, (values, rewards, continuations))
        if self.amp is not None:
            rewards,scores=self.style_reward(torch.stack(states),rewards)
        else:
            scores=torch.zeros_like(rewards)
        alive = cont.cumprod(0)
        weights = torch.cat((torch.ones_like(alive[:1]), alive[:-1]), 0)
        discounts = self.gamma**torch.arange(self.horizon, device=state.device)[:, None]
        total = (discounts*weights*rewards).sum(0)
        objective = total+self.gamma**self.horizon*alive[-1]*self.critic(x[-1])
        return dict(objective=objective.mean(), task_return=total.mean(), critic_inputs=x,
                    rewards=rewards, continuations=cont, imagined_actions=torch.stack(actions).detach(),
                    recovery_mode_fraction=task['recovery_enabled'].float().mean(), style_score=scores.mean())
