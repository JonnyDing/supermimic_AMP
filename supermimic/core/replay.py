"""Time-major GPU ring replay. Windows never cross reset or invalid dynamics.

Terminal transitions may end a window; truncated transitions also break sequences.
All samples are owned snapshots, not live simulator views. Environment IDs permit
strict held-out splits including normalizer fitting.
"""
import torch
from .state import pack


class SequenceReplay:
    def __init__(self, capacity, num_envs, device='cpu'):
        if capacity <= 0 or num_envs <= 0:
            raise ValueError('Replay dimensions must be positive')
        self.capacity, self.num_envs = capacity, num_envs
        self.device = torch.device(device)
        self.storage = {}
        self.ptr = self.size = self.total_steps = 0
        self._cache = {}

    def add(self, transition, dynamics_valid=None):
        values = dict(state=pack(transition.state), next_state=pack(transition.next_state),
                      action=transition.action, context=transition.context,
                      terminated=transition.terminated, truncated=transition.truncated,
                      episode_id=transition.episode_id, reward=transition.reward,
                      dynamics_valid=torch.ones_like(transition.terminated) if dynamics_valid is None else dynamics_valid)
        values.update({f'task/{key}': value for key, value in transition.task.items()})
        values.update({f'next_task/{key}': value for key, value in transition.next_task.items()})
        values.update({f'labels/{key}': value for key, value in transition.labels.items()})
        self.add_values(values)

    @torch.no_grad()
    def add_values(self, values):
        required = {'state', 'next_state', 'action', 'context', 'terminated', 'truncated', 'episode_id', 'dynamics_valid'}
        if not required <= values.keys():
            raise ValueError(f'Missing fields: {required - values.keys()}')
        if self.storage and set(values) != set(self.storage):
            raise ValueError('Replay field schema changed')
        for key, value in values.items():
            if value.shape[0] != self.num_envs:
                raise ValueError(f'Wrong batch dimension: {key}')
            if key in self.storage:
                target = self.storage[key]
                if target.shape[1:] != value.shape or target.dtype != value.dtype:
                    raise ValueError(f'Replay field shape/dtype changed: {key}')
        if not self.storage:
            self.storage = {k: torch.empty((self.capacity, *v.shape), dtype=v.dtype, device=self.device)
                            for k, v in values.items()}
        for key, value in values.items():
            self.storage[key][self.ptr].copy_(value.detach())
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)
        self.total_steps += 1
        self._cache.clear()

    def rows(self, ages):
        return (self.ptr - self.size + ages) % self.capacity

    def valid_starts(self, horizon, env_ids):
        ids = tuple(int(i) for i in env_ids)
        if len(set(ids)) != len(ids) or not ids or min(ids) < 0 or max(ids) >= self.num_envs:
            raise ValueError('Invalid environment partition')
        key = (horizon, ids)
        if key in self._cache:
            return self._cache[key]
        if horizon < 1 or horizon > self.size:
            raise ValueError('Not enough replay steps for horizon')
        ages = torch.arange(self.size, device=self.device)
        envs = torch.tensor(ids, device=self.device)
        rows = self.rows(ages)
        done = (self.storage['terminated'] | self.storage['truncated'])[rows][:, envs]
        episode = self.storage['episode_id'][rows][:, envs].unfold(0, horizon, 1)
        valid = self.storage['dynamics_valid'][rows][:, envs].unfold(0, horizon, 1).all(-1)
        valid &= (episode == episode[..., :1]).all(-1)
        if horizon > 1:
            valid &= ~done.unfold(0, horizon, 1)[..., :-1].any(-1)
        start_age, env_column = valid.nonzero(as_tuple=True)
        if not len(start_age):
            raise ValueError('No valid contiguous windows; collect more data or shorten horizon')
        result = (start_age, envs[env_column])
        self._cache[key] = result
        return result

    def sample(self, batch_size, horizon, env_ids, keys=None, generator=None):
        ages, envs = self.valid_starts(horizon, env_ids)
        choices = torch.randint(len(ages), (batch_size,), device=self.device, generator=generator)
        return self.gather(ages[choices], envs[choices], horizon, keys)

    def gather(self, ages, envs, horizon, keys=None):
        rows = self.rows(ages[:, None] + torch.arange(horizon, device=self.device))
        out = {k: self.storage[k][rows, envs[:, None]] for k in (self.storage if keys is None else keys)}
        out['_env_ids'], out['_start_ages'] = envs.clone(), ages.clone()
        return out

    def sample_prefix(self, batch_size, horizon, env_ids, keys=None, generator=None):
        """Keep terminal prefixes, masking everything after a reset or invalid step.

        Unlike strict contiguous samples this retains states just before failure,
        including short episodes that cannot supply a full surviving window.
        """
        if horizon < 1 or horizon > self.size:
            raise ValueError('Not enough replay steps for horizon')
        ages, envs = self.valid_starts(1, env_ids)
        available = ages <= self.size - horizon
        ages, envs = ages[available], envs[available]
        if not len(ages):
            raise ValueError('No valid prefix starts')
        picks = torch.randint(len(ages), (batch_size,), device=self.device, generator=generator)
        ages, envs = ages[picks], envs[picks]
        out = self.gather(ages, envs, horizon, keys)
        rows = self.rows(ages[:, None] + torch.arange(horizon, device=self.device))
        valid = self.storage['dynamics_valid'][rows, envs[:, None]]
        episodes = self.storage['episode_id'][rows, envs[:, None]]
        done = (self.storage['terminated'][rows, envs[:, None]] |
                self.storage['truncated'][rows, envs[:, None]])
        preceding = torch.cat((torch.ones_like(done[:, :1]), ~done[:, :-1]), 1)
        mask = (valid & preceding & (episodes == episodes[:, :1])).long().cumprod(1).bool()
        out['_mask'] = mask
        # Masked fields can contain invalid physics values. Sanitize before model
        # evaluation: multiplying a NaN loss by zero would still yield a NaN.
        for key in tuple(out):
            if key.startswith('_'): continue
            shape = (*mask.shape, *((1,) * (out[key].ndim - 2)))
            out[key] = torch.where(mask.reshape(shape), out[key], out[key][:, :1])
        return out

    def state_dict(self):
        return dict(capacity=self.capacity, num_envs=self.num_envs, ptr=self.ptr, size=self.size,
                    total_steps=self.total_steps, storage=self.storage)

    @classmethod
    def from_state_dict(cls, state, device='cpu'):
        replay = cls(state['capacity'], state['num_envs'], device)
        for name in ('ptr', 'size', 'total_steps'):
            setattr(replay, name, state[name])
        replay.storage = {k: v.to(device).clone() for k, v in state['storage'].items()}
        return replay
