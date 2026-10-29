"""mjlab 1.2 physics collection without changing the upstream step ordering.

This is a physics-data adapter, not yet the full SimulatorAdapter task contract.
Terminal task snapshots are before command update; terminal observations are NOT
represented by the post-reset observation returned by env.step().
"""
from dataclasses import dataclass, fields

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply

from supermimic.contracts import PhysicsState
from supermimic.mdp.observations import terminal_actor_history


def name_indices(available, requested):
    if len(set(available)) != len(available):
        raise ValueError("Duplicate names in robot schema")
    if len(set(requested)) != len(requested):
        raise ValueError("Duplicate requested names")
    missing = set(requested) - set(available)
    if missing:
        raise ValueError(f"Missing robot names: {sorted(missing)}")
    return [available.index(name) for name in requested]


def read_physics(env):
    robot = env.scene["robot"]
    idx = robot.indexing
    q, v = env.sim.data.qpos, env.sim.data.qvel
    fq, fv = idx.free_joint_q_adr.long(), idx.free_joint_v_adr.long()
    if fq.numel() != 7 or fv.numel() != 6:
        raise ValueError("Expected exactly one floating root")
    quat = q[:, fq[3:7]].clone()
    return PhysicsState(
        q[:, fq[:3]].clone() - env.scene.env_origins,
        quat,
        v[:, fv[:3]].clone(),
        quat_apply(quat, v[:, fv[3:6]]).clone(),
        q[:, idx.joint_q_adr.long()].clone(),
        v[:, idx.joint_v_adr.long()].clone(),
    )


def read_task_metadata(env):
    command = env.command_manager.get_term("twist")
    tm = env.termination_manager
    return {
        "observation_history": env.obs_buf["actor"].reshape(env.num_envs, env.cfg.observations["actor"].history_length, -1).clone(),
        "command": command.command.clone(),
        "heading_target": command.heading_target.clone(),
        "command_time_left": command.time_left.clone(),
        "is_heading_env": command.is_heading_env.clone(),
        "is_standing_env": command.is_standing_env.clone(),
        "recovery_enabled": getattr(tm, "_delay_env_mask", torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)).clone(),
        "recovery_counter": getattr(tm, "_delay_counters", torch.zeros(env.num_envs, device=env.device, dtype=torch.long)).clone(),
        "episode_step": env.episode_length_buf.clone(),
        "previous_action": env.action_manager.action.clone(),
    }


class CaptureEnv(ManagerBasedRlEnv):
    """Capture snapshots at the existing reset boundary; no extra forward call."""
    _collecting = False
    terminal_snapshot = None

    def _reset_idx(self, env_ids=None):
        if self._collecting:
            ids = env_ids.clone()
            state, task = read_physics(self), read_task_metadata(self)
            if not getattr(self, 'learned_task', False):
                task['observation_history'] = terminal_actor_history(self, state)
            # Learned-task observation loss masks terminal rows: reset observations
            # and stale terminal ray casts must never become prediction targets.
            self.terminal_snapshot = (ids, state, task)
        super()._reset_idx(env_ids)

    def step(self, action):
        self.contact_labels = None
        original_compute = self.reward_manager.compute
        def compute_and_capture(*args, **kwargs):
            reward = original_compute(*args, **kwargs)
            self.contact_labels = read_contact_labels(self)
            return reward
        self.reward_manager.compute = compute_and_capture
        self.terminal_snapshot = None
        self._collecting = True
        try:
            return super().step(action)
        finally:
            self._collecting = False
            self.reward_manager.compute = original_compute


@dataclass
class PhysicsTransition:
    state: PhysicsState
    next_state: PhysicsState
    action: torch.Tensor
    context: torch.Tensor
    task: dict
    next_task: dict
    terminated: torch.Tensor
    truncated: torch.Tensor
    reward: torch.Tensor
    episode_id: torch.Tensor
    labels: dict[str, torch.Tensor]


class PhysicsCollector:
    def __init__(self, env: CaptureEnv):
        if not isinstance(env, CaptureEnv):
            raise TypeError("PhysicsCollector requires CaptureEnv")
        # Events can change state after the capture boundary. Fail closed until
        # their semantics are explicitly represented in the dynamics contract.
        unsupported = [k for k, c in env.cfg.events.items() if c.mode in ("interval", "step")]
        if unsupported:
            raise ValueError(f"Disable unmodelled step/interval events: {unsupported}")
        self.env = env
        self.episode_ids = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        robot = env.scene["robot"]
        torso = name_indices(list(robot.body_names), ["torso_link"])[0]
        self.torso_id = int(robot.indexing.body_ids[torso])
        self.context_schema = {
            "geom_friction": env.sim.model.geom_friction.shape[1] * 3,
            "torso_com_local": 3,
            "encoder_bias": len(robot.joint_names),
        }

    def context(self):
        env = self.env
        def expand(x):
            return x.expand(env.num_envs, *x.shape[1:]).reshape(env.num_envs, -1)
        context = torch.cat((expand(env.sim.model.geom_friction),
                          expand(env.sim.model.body_ipos[:, self.torso_id]),
                          expand(env.scene["robot"].data.encoder_bias)), -1).clone()
        if getattr(env, "learned_task", False):
            from supermimic.core.task_model import task_context
            task = read_task_metadata(env)
            context = task_context(context, task["observation_history"], task, env.max_episode_length)
        return context

    @torch.no_grad()
    def step(self, action):
        state = read_physics(self.env)
        task = read_task_metadata(self.env)
        context = self.context()
        ids = self.episode_ids.clone()
        _, reward, terminated, truncated, _ = self.env.step(action)
        next_state = read_physics(self.env)
        next_task = read_task_metadata(self.env)
        snapshot = self.env.terminal_snapshot
        if snapshot is not None:
            reset_ids, terminal_state, terminal_task = snapshot
            for f in fields(PhysicsState):
                getattr(next_state, f.name)[reset_ids] = getattr(terminal_state, f.name)[reset_ids]
            for key in next_task:
                next_task[key][reset_ids] = terminal_task[key][reset_ids]
        terminated, truncated = terminated.clone(), truncated.clone()
        self.episode_ids += (terminated | truncated).long()
        return PhysicsTransition(state, next_state, action.to(self.env.device).clone(), context,
                                 task, next_task, terminated, truncated, reward.clone(), ids, self.env.contact_labels)


def read_contact_labels(env):
    """Reward-time labels, captured BEFORE reset/forward/sense.

    Force history covers physics substeps, not control-step history.
    Persist raw values as well as the exact task's contact-derived targets.
    """
    labels = {}
    for name in ('feet_ground_contact', 'self_collision'):
        sensor = env.scene[name]
        for field in ('found', 'force', 'force_history', 'current_air_time', 'current_contact_time'):
            value = getattr(sensor.data, field, None)
            if value is not None:
                labels[f'{name}/{field}'] = value.clone()
    feet_cfg = env.reward_manager.get_term_cfg('foot_slip')
    site_ids = feet_cfg.params['asset_cfg'].site_ids
    labels['feet_velocity_xy'] = env.scene['robot'].data.site_lin_vel_w[:, site_ids, :2].clone()
    found = labels['feet_ground_contact/found']
    labels['feet_in_contact'] = (found > 0).clone()
    labels['foot_slip_raw'] = feet_cfg.func(env, **feet_cfg.params).clone()
    collision_cfg = env.reward_manager.get_term_cfg('self_collisions')
    labels['self_collision_raw'] = collision_cfg.func(env, **collision_cfg.params).clone()
    return labels
