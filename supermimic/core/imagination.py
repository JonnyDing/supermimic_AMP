"""Short-horizon analytic policy gradients and detached GAE critic targets."""
from contextlib import contextmanager
import torch
from torch import nn
from .normalization import RunningNorm


@contextmanager
def freeze_parameters(*modules):
    original = [(p, p.requires_grad) for m in modules for p in m.parameters()]
    try:
        for p, _ in original: p.requires_grad_(False)
        yield
    finally:
        for p, value in original: p.requires_grad_(value)


class ValueModel(nn.Module):
    def __init__(self, input_dim, normalize=False):
        super().__init__()
        self.norm=RunningNorm(input_dim) if normalize else nn.Identity()
        self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        x = self.norm(x)
        dtype = getattr(self, 'mlp_dtype', None)
        if dtype is None:
            return self.net(x).squeeze(-1)
        with torch.autocast('cuda', dtype=dtype):
            return self.net(x).float().squeeze(-1)


def critic_input(wm, state, previous, context, command):
    base = wm.inputs(state, previous, context)
    if getattr(wm,'statistics_mode','fixed') != 'running':
        base = (base-wm.input_mean)/wm.input_std
    return torch.cat((base, command), -1)


@torch.no_grad()
def gae_targets(rewards, continuations, values, gamma=0.99, lam=0.95):
    if values.shape[0] != rewards.shape[0] + 1:
        raise ValueError('Need H+1 value predictions for H transitions')
    result = values[-1]
    targets = []
    for t in reversed(range(rewards.shape[0])):
        result = rewards[t] + gamma * continuations[t] * ((1-lam) * values[t+1] + lam * result)
        targets.append(result)
    return torch.stack(targets[::-1])


def update_critic(critic, optimizer, rollout, epochs=2, minibatches=4, return_tensor=False, gamma=0.99, lam=0.95, return_metrics=False):
    x = rollout['critic_inputs'].detach()
    if isinstance(getattr(critic,'norm',None),RunningNorm):
        critic.norm.update(x)
    with torch.no_grad():
        values = critic(x)
        targets = gae_targets(rollout['rewards'].detach(), rollout['continuations'].detach(), values, gamma=gamma, lam=lam)
        if return_metrics:
            residual = (values[:-1] - targets).square()
            alive = torch.cat((torch.ones_like(rollout['continuations'][:1]),
                               rollout['continuations'].detach().cumprod(0)[:-1]), 0)
            diagnostics = {
                'critic_target_abs_max': targets.abs().amax(),
                'critic_value_abs_max': values.abs().amax(),
                'critic_target_std': targets.std(unbiased=False),
                'critic_input_abs_max': x.abs().amax(),
                'imagined_reward_abs_max': rollout['rewards'].detach().abs().amax(),
                'critic_mse_before': residual.mean(),
                'critic_alive_weighted_mse': (residual * alive).sum() / alive.sum().clamp_min(1),
            }
    x, targets = x[:-1].flatten(0, 1), targets.flatten()
    losses = []
    for _ in range(epochs):
        for indices in torch.randperm(len(x), device=x.device).tensor_split(minibatches):
            if not len(indices): continue
            loss = (critic(x[indices]) - targets[indices]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.detach())
    result = torch.stack(losses).mean()
    if return_metrics:
        return result, diagnostics
    return result if return_tensor else float(result)
