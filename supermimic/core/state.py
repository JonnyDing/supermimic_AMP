"""Packed state schema: position, wxyz, world link velocities, named joints."""
import torch
from supermimic.contracts import PhysicsState
from .rotations import quat_to_mat, mat_to_6d, yaw_mat, so3_log, so3_exp, mat_to_quat


def pack(state: PhysicsState):
    return torch.cat((state.root_pos, state.root_quat_wxyz, state.root_lin_vel_w,
                      state.root_ang_vel_w, state.joint_pos, state.joint_vel), -1)


def unpack(x):
    joints = (x.shape[-1] - 13) // 2
    if x.shape[-1] != 13 + 2 * joints:
        raise ValueError('Invalid packed state dimension')
    return PhysicsState(x[..., :3], x[..., 3:7], x[..., 7:10], x[..., 10:13],
                        x[..., 13:13+joints], x[..., 13+joints:])


def mv(matrix, vector):
    return (matrix * vector.unsqueeze(-2)).sum(-1)


def features(x):
    s = unpack(x)
    rotation = quat_to_mat(s.root_quat_wxyz)
    rt = rotation.transpose(-1, -2)
    heading = yaw_mat(rotation)
    return torch.cat((s.root_pos[..., 2:3], mat_to_6d(heading.transpose(-1, -2) @ rotation),
                      mv(rt, s.root_lin_vel_w), mv(rt, s.root_ang_vel_w), s.joint_pos, s.joint_vel), -1)


def delta(source, target):
    """Local increment (also useful as prediction error); 12 + 2*J dimensions."""
    s, t = unpack(source), unpack(target)
    r = quat_to_mat(s.root_quat_wxyz)
    rt = r.transpose(-1, -2)
    return torch.cat((mv(yaw_mat(r).transpose(-1, -2), t.root_pos - s.root_pos),
                      so3_log(rt @ quat_to_mat(t.root_quat_wxyz)),
                      mv(rt, t.root_lin_vel_w - s.root_lin_vel_w),
                      mv(rt, t.root_ang_vel_w - s.root_ang_vel_w),
                      t.joint_pos - s.joint_pos, t.joint_vel - s.joint_vel), -1)


def integrate(x, increments):
    s = unpack(x)
    joints = s.joint_pos.shape[-1]
    dp, dr, dv, dw, dq, dqd = increments.split((3, 3, 3, 3, joints, joints), -1)
    rotation = quat_to_mat(s.root_quat_wxyz)
    return pack(PhysicsState(s.root_pos + mv(yaw_mat(rotation), dp),
                             mat_to_quat(rotation @ so3_exp(dr)),
                             s.root_lin_vel_w + mv(rotation, dv),
                             s.root_ang_vel_w + mv(rotation, dw), s.joint_pos + dq, s.joint_vel + dqd))
