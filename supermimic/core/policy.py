"""Shared checkpoint loader for WMRL and its frozen initialization policy."""
from pathlib import Path
import torch
from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.modules.normalizer import EmpiricalNormalization

ROOT = Path(__file__).resolve().parents[2]


def load_policy(checkpoint, device):
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    policy = ActorCritic(384, 864, 29, actor_hidden_dims=[512, 256, 128],
                         critic_hidden_dims=[512, 256, 128], activation='elu').to(device)
    policy.load_state_dict(saved['model_state_dict'], strict=True)
    norm = EmpiricalNormalization([384]).to(device)
    norm.load_state_dict(saved['obs_norm_state_dict'], strict=True)
    policy.eval(); norm.eval()
    return policy, norm
