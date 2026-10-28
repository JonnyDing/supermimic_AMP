"""Sample upright starts for velocity imagination updates."""
import torch

START_KEYS = ['state', 'context', 'task/observation_history', 'task/previous_action', 'task/command']


def eligible_starts(replay, ids, horizon, dt):
    ages, envs = replay.valid_starts(horizon, ids)
    rows = replay.rows(ages)
    values = replay.storage
    good = (~values['task/recovery_enabled'][rows, envs] & ~values['task/is_heading_env'][rows, envs] &
            (values['task/command_time_left'][rows, envs] > horizon * dt) &
            (values['state'][rows, envs, 2] > 0.65) &
            (values['task/previous_action'][rows, envs].abs().amax(-1) < 10))
    ages, envs = ages[good], envs[good]
    if not len(ages): raise ValueError('No eligible upright non-heading start states')
    return ages, envs


def sample_starts(replay, eligible, batch):
    ages, envs = eligible
    choice = torch.randint(len(ages), (batch,), device=replay.device)
    return replay.gather(ages[choice], envs[choice], 1, START_KEYS)
