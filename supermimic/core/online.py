"""Alternating online dynamics, BPTT actor and GAE critic updates."""
import torch
from .imagination import freeze_parameters, imagine, update_critic
from .world_model import rollout_loss
from supermimic.evaluation.world_model import DYNAMICS_KEYS
from .sampling import eligible_starts, sample_starts


def snapshot(module): return {k: v.detach().clone() for k, v in module.state_dict().items()}
def same_state(before, module): return all(torch.equal(v, module.state_dict()[k]) for k, v in before.items())


class AlternatingUpdates:
    def __init__(self, wm, policy, normalizer, critic, heads, teacher, actor_lr=1e-6):
        self.wm, self.policy, self.normalizer, self.critic = wm, policy, normalizer, critic
        self.heads, self.teacher = heads, teacher
        self.opt_wm = torch.optim.Adam(wm.parameters(), lr=2e-4)
        self.opt_actor = torch.optim.Adam(policy.actor.parameters(), lr=actor_lr, betas=(0.9, 0.95))
        self.opt_critic = torch.optim.Adam(critic.parameters(), lr=2e-4)
        self.wm_updates = self.actor_updates = self.critic_updates = 0
        self.normalizer.eval()
        for p in self.teacher.parameters(): p.requires_grad_(False)
        for p in self.policy.critic.parameters(): p.requires_grad_(False)

    def update(self, replay, train_ids, batch_size=128, horizon=8, wm_batches=2, audit=False):
        actor_before = snapshot(self.policy.actor) if audit else None
        for p in self.wm.parameters(): p.requires_grad_(True)
        self.wm.train()
        wm_losses = []
        for _ in range(wm_batches):
            window = replay.sample(batch_size, 8, train_ids, DYNAMICS_KEYS)
            loss, physics, contact = rollout_loss(self.wm, window)
            self.opt_wm.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.wm.parameters(), 1., error_if_nonfinite=True)
            self.opt_wm.step()
            self.wm_updates += 1
            wm_losses.append(float(physics))
        if audit and not same_state(actor_before, self.policy.actor): raise AssertionError('WM update changed actor')
        self.opt_wm.zero_grad(set_to_none=True)
        self.opt_critic.zero_grad(set_to_none=True)
        self.wm.eval()
        wm_before = snapshot(self.wm) if audit else None
        critic_before = snapshot(self.critic) if audit else None
        eligible = eligible_starts(replay, train_ids, horizon, self.heads.dt)
        starts = sample_starts(replay, eligible, batch_size)
        with freeze_parameters(self.wm, self.critic):
            rollout = imagine(self.wm, self.policy.actor, self.normalizer, self.critic, self.heads, starts,
                              horizon=horizon, teacher=self.teacher, anchor_weight=1.0)
            self.opt_actor.zero_grad(set_to_none=True)
            (-rollout['objective']).backward()
            gradient = torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), 1., error_if_nonfinite=True)
            torch.nn.utils.clip_grad_value_(self.policy.actor.parameters(), 0.1)
            self.opt_actor.step()
        if any(p.grad is not None for p in self.wm.parameters()): raise AssertionError('Actor backward leaked into WM')
        if any(p.grad is not None for p in self.critic.parameters()): raise AssertionError('Actor backward leaked into critic')
        if audit and (not same_state(wm_before, self.wm) or not same_state(critic_before, self.critic)):
            raise AssertionError('Actor optimizer changed WM/critic parameters or stats')
        self.actor_updates += 1
        actor_after = snapshot(self.policy.actor) if audit else None
        critic_loss = update_critic(self.critic, self.opt_critic, rollout)
        self.critic_updates += 1
        if audit and not same_state(actor_after, self.policy.actor): raise AssertionError('Critic update changed actor')
        return {'wm_loss':sum(wm_losses)/len(wm_losses), 'critic_loss':critic_loss,
                'imagined_task_return':float(rollout['task_return']), 'actor_gradient_norm':float(gradient),
                'anchor_mse':float(rollout['anchor_mse']), 'eligible_starts':len(eligible[0]), 'isolation_audited':audit}
