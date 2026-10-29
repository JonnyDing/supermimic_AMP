"""Fixed velocity commands and fallen poses for policy evaluation."""
import math
import numpy as np
import torch
from supermimic.core.policy import ROOT

COMMANDS = [[0., 0., 0.], [0.5, 0., 0.], [1.5, 0., 0.], [0., 0.5, 0.], [0., 0., 0.75]]


def fixed_recovery_frames():
    path = ROOT / 'src/assets/motions/g1/amp/Recovery/fallAndGetUp1_subject1.npz'
    data = dict(np.load(path))
    quat = data['body_quat_w'][:, 0]
    gravity_z_abs_upright = 1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    fallen = (data['body_pos_w'][:, 0, 2] < 0.5) | (gravity_z_abs_upright < math.cos(math.radians(70)))
    candidates = np.flatnonzero(fallen)
    frames = candidates[np.linspace(0, len(candidates)-1, 8, dtype=int)].tolist()
    return path, data, frames


def initialize_scenarios(env, data, frames):
    robot = env.scene['robot']
    device = env.device
    nv = len(COMMANDS)
    ids = torch.arange(env.num_envs, device=device)
    pose = robot.data.default_root_state[:, :7].clone()
    pose[:, :3] += env.scene.env_origins
    velocity = torch.zeros(env.num_envs, 6, device=device)
    q, qd = robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel)
    to_tensor = lambda a: torch.as_tensor(a, device=device, dtype=torch.float32)
    pose[nv:, 2] = to_tensor(data['body_pos_w'][frames, 0, 2]) + env.scene.env_origins[nv:, 2]
    pose[nv:, 3:] = to_tensor(data['body_quat_w'][frames, 0])
    velocity[nv:, :3] = to_tensor(data['body_lin_vel_w'][frames, 0])
    velocity[nv:, 3:] = to_tensor(data['body_ang_vel_w'][frames, 0])
    limits = robot.data.soft_joint_pos_limits
    q[nv:] = to_tensor(data['joint_pos'][frames]).clamp(limits[nv:, :, 0], limits[nv:, :, 1])
    qd[nv:] = to_tensor(data['joint_vel'][frames])
    robot.write_root_link_pose_to_sim(pose, env_ids=ids)
    robot.write_root_link_velocity_to_sim(velocity, env_ids=ids)
    robot.write_joint_state_to_sim(q, qd, env_ids=ids)
    env.episode_length_buf.zero_()
    env.action_manager.reset(ids)
    env.termination_manager._delay_counters.zero_()
    env.termination_manager._delay_env_mask[:] = torch.arange(env.num_envs, device=device) >= nv
    command = env.command_manager.get_term('twist')
    command.command.zero_()
    command.command[:nv] = to_tensor(COMMANDS)
    command.time_left.fill_(1e9)
    command.is_heading_env.zero_(); command.is_standing_env.zero_()
    env.scene.write_data_to_sim()
    env.sim.forward(); env.sim.sense()
    env.observation_manager.reset(ids)
    env.obs_buf = env.observation_manager.compute(update_history=True)
