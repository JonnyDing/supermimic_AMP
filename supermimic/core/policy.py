"""WMRL policy initialization and checkpoint loading."""
from pathlib import Path
import torch
from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.modules.normalizer import EmpiricalNormalization

ROOT = Path(__file__).resolve().parents[2]


def create_policy(device, actor_obs=384, critic_obs=864, actions=29):
    """Random policy with a small action head; keep the exported checkpoint schema."""
    policy = ActorCritic(actor_obs, critic_obs, actions, actor_hidden_dims=[512, 256, 128],
                         critic_hidden_dims=[512, 256, 128], activation='elu').to(device)
    torch.nn.init.orthogonal_(policy.actor[-1].weight, gain=0.01)
    torch.nn.init.zeros_(policy.actor[-1].bias)
    norm = EmpiricalNormalization([actor_obs], until=1e8).to(device)
    policy.eval(); norm.eval()
    return policy, norm


def load_policy(checkpoint, device):
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    weights = saved['model_state_dict']
    policy, norm = create_policy(device, weights['actor.0.weight'].shape[1],
                                 weights['critic.0.weight'].shape[1], weights['std'].numel())
    policy.load_state_dict(saved['model_state_dict'], strict=True)
    norm.load_state_dict(saved['obs_norm_state_dict'], strict=True)
    return policy, norm
