"""Batched physical state: wxyz quaternions and world-frame link velocities."""
from dataclasses import dataclass
from torch import Tensor


@dataclass
class PhysicsState:
    root_pos: Tensor
    root_quat_wxyz: Tensor
    root_lin_vel_w: Tensor
    root_ang_vel_w: Tensor
    joint_pos: Tensor
    joint_vel: Tensor
