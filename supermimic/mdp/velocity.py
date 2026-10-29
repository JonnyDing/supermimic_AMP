"""Velocity task observations, reward terms and differentiable survival."""
import math
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat
from supermimic.core.state import unpack
from supermimic.mdp.kinematics import RobotKinematics


class VelocityHeads:
    def __init__(self, env):
        self.kin = RobotKinematics(env)
        robot = env.scene['robot']
        self.torso = list(robot.body_names).index('torso_link')
        self.q0 = robot.data.default_joint_pos[0].clone()
        self.qd0 = robot.data.default_joint_vel[0].clone()
        self.limits = robot.data.soft_joint_pos_limits[0].clone()
        self.dt = env.step_dt
        terms = env.observation_manager._group_obs_term_cfgs['actor']
        # Channel-wise observation noise from the configured task.
        sizes=(3,3,3,29,29,29)
        lower=[];upper=[]
        enabled=env.cfg.observations['actor'].enable_corruption
        for term,size in zip(terms,sizes,strict=True):
            noise=term.noise if enabled else None
            lower.extend([0. if noise is None else noise.n_min]*size)
            upper.extend([0. if noise is None else noise.n_max]*size)
        self.noise_lower=torch.tensor(lower,device=env.device)
        self.noise_upper=torch.tensor(upper,device=env.device)

    def noisy_observation(self, packed, action, command):
        obs=self.observation(packed,action,command)
        return obs+self.noise_lower+(self.noise_upper-self.noise_lower)*torch.rand_like(obs)

    def observation(self, packed, action, command):
        state = unpack(packed)
        gravity = torch.zeros_like(state.root_pos); gravity[:, 2] = -1
        return torch.cat((quat_apply_inverse(state.root_quat_wxyz, state.root_ang_vel_w),
                          quat_apply_inverse(state.root_quat_wxyz, gravity), command,
                          state.joint_pos - self.q0, state.joint_vel - self.qd0, action), -1)

    def reward(self, packed, action, previous, command):
        state = unpack(packed)
        _, rotation, velocity, angular = self.kin(state)
        local_velocity = quat_apply_inverse(yaw_quat(rotation[:, self.torso]), velocity[:, self.torso])
        target = torch.cat((command[:, :2], torch.zeros_like(command[:, :1])), -1)
        linear = torch.exp(-(local_velocity - target).square().sum(-1))
        body_angular = quat_apply_inverse(rotation[:, self.torso], angular[:, self.torso])
        angular_error = (angular[:, self.torso, 2] - command[:, 2]).square() + body_angular[:, :2].square().sum(-1)
        yaw = torch.exp(-angular_error / 3.14**2)
        smooth = (action - previous).square().sum(-1)
        limits = ((self.limits[:, 0] - state.joint_pos).clamp_min(0) +
                  (state.joint_pos - self.limits[:, 1]).clamp_min(0)).sum(-1)
        return self.dt * (linear + yaw - 0.01 * smooth - 10 * limits)

    def continuation(self, packed):
        state = unpack(packed)
        gravity = torch.zeros_like(state.root_pos); gravity[:, 2] = -1
        up = -quat_apply_inverse(state.root_quat_wxyz, gravity)[:, 2]
        return torch.sigmoid((state.root_pos[:, 2] - 0.5) / 0.02) * torch.sigmoid((up - math.cos(math.radians(70))) / 0.03)
