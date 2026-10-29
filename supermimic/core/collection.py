"""Collect current-policy transitions with exploration noise."""
import torch
from supermimic.adapters.mjlab import PhysicsCollector
from .state import pack


class OnlineCollection:
    def __init__(self, env, policy, normalizer, replay, action_noise=0.1, statistics_ids=None):
        self.env, self.policy, self.normalizer, self.replay = env, policy, normalizer, replay
        self.collector = PhysicsCollector(env)
        self.action_noise = action_noise
        self.total_steps = 0
        self.statistics_ids = statistics_ids
        self.world_model = None
        # Old checkpoints may contain replay from a different simulator episode.
        if replay.size:
            rows = replay.rows(torch.arange(replay.size, device=replay.device))
            self.collector.episode_ids.fill_(int(replay.storage['episode_id'][rows].max()) + 1)

    @torch.no_grad()
    def collect(self, count, policy_version, random_actions=False):
        self.policy.eval(); self.normalizer.eval()
        for _ in range(count):
            if self.statistics_ids is not None:
                self.normalizer.update(self.env.obs_buf['actor'][self.statistics_ids])
            action = (torch.zeros_like(self.env.action_manager.action) if random_actions else
                      self.policy.act_inference(self.normalizer(self.env.obs_buf['actor'])))
            action = action + self.action_noise * torch.randn_like(action)
            transition = self.collector.step(action)
            if self.world_model is not None:
                ids=self.statistics_ids
                self.world_model.update_stats(pack(transition.state)[ids],pack(transition.next_state)[ids],
                                              transition.action[ids],transition.context[ids])
            if self.world_model is not None and hasattr(self.world_model, 'update_task_stats'):
                self.world_model.update_task_stats(transition.next_task['observation_history'][self.statistics_ids],
                                                   transition.reward[self.statistics_ids],
                                                   ~(transition.terminated | transition.truncated)[self.statistics_ids])
            version = torch.full_like(transition.episode_id, policy_version)
            transition.task['policy_version'] = version
            transition.next_task['policy_version'] = version.clone()
            self.replay.add(transition)
            self.total_steps += 1
