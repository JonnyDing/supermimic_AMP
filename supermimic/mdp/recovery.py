"""Native delayed termination state machine and differentiable full-task proxy."""
import math
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_apply, yaw_quat
from supermimic.core.state import unpack
from supermimic.mdp.velocity import VelocityHeads


def advance_recovery(bad, timeout, enabled, counter, maximum=250):
    """Exactly mirrors native compute, including timeout counting and ready clear."""
    done=bad|timeout
    if maximum<=0: return counter, bad, timeout
    incremented=counter+(enabled & done).long()
    ready=enabled & done & (incremented>=maximum)
    terminated=bad & ~(enabled & done & (incremented<maximum))
    updated=torch.where(ready | (enabled & ~done),torch.zeros_like(counter),incremented)
    return updated,terminated,timeout


class RecoveryHeads(VelocityHeads):
    def __init__(self,env):
        super().__init__(env)
        robot=env.scene['robot']; model=env.sim.mj_model
        self.height=robot.data.default_root_state[0,2].clone()
        self.maximum=env.termination_manager._max_delay_steps
        self.max_episode_steps=env.max_episode_length
        command=env.command_manager.get_term('twist').cfg
        self.heading_stiffness=command.heading_control_stiffness
        self.yaw_range=command.ranges.ang_vel_z
        local={int(b):i for i,b in enumerate(robot.indexing.body_ids)}
        self.feet=[]
        for name in ('left_foot','right_foot'):
            sid=int(robot.indexing.site_ids[list(robot.site_names).index(name)])
            self.feet.append((local[int(model.site_bodyid[sid])],torch.tensor(model.site_pos[sid].copy(),device=env.device,dtype=torch.float32)))
        self.collision_history=4

    def command(self,state,task):
        q=unpack(state).root_quat_wxyz
        heading=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2].square()+q[:,3].square()))
        error=task['heading_target']-heading
        error=torch.atan2(error.sin(),error.cos())
        yaw=(self.heading_stiffness*error).clamp(*self.yaw_range)
        cmd=torch.cat((task['command'][:,:2],torch.where(task['is_heading_env'],yaw,task['command'][:,2])[:,None]),-1)
        return torch.where(task['is_standing_env'][:,None],torch.zeros_like(cmd),cmd)

    def advance(self,state,task):
        s=unpack(state); gravity=torch.zeros_like(s.root_pos); gravity[:,2]=-1
        up=-quat_apply_inverse(s.root_quat_wxyz,gravity)[:,2]
        bad=(s.root_pos[:,2]<.5)|(up<math.cos(math.radians(70)))
        step=task['episode_step']+1
        timeout=step>=self.max_episode_steps
        counter,terminated,truncated=advance_recovery(bad,timeout,task['recovery_enabled'],task['recovery_counter'],self.maximum)
        next_task={**task,'episode_step':step,'recovery_counter':counter,
                   'command_time_left':task['command_time_left']-self.dt}
        delayed=task['recovery_enabled'] & (counter>0)
        soft_alive=self.continuation(state)
        # Delay permits fallen recovery; at the deadline use smooth physical survival.
        permitted=task['recovery_enabled'] & (task['recovery_counter']+1<self.maximum)
        continuation=torch.where(permitted,torch.ones_like(soft_alive),soft_alive)
        continuation=torch.where(timeout,torch.zeros_like(continuation),continuation)
        return next_task, delayed, terminated, truncated, continuation

    def reward_terms(self,state,next_state,action,previous,command,delayed,termination,contact_logits,
                     kinematics):
        s=unpack(next_state); old=unpack(state)
        _,rotations,velocities,angular=kinematics
        smooth=(action-previous).square().sum(-1)
        limits=((self.limits[:,0]-s.joint_pos).clamp_min(0)+(s.joint_pos-self.limits[:,1]).clamp_min(0)).sum(-1)
        local=quat_apply_inverse(yaw_quat(rotations[:,self.torso]),velocities[:,self.torso])
        target=torch.cat((command[:,:2],torch.zeros_like(command[:,:1])),-1)
        linear=torch.exp(-(local-target).square().sum(-1))
        body_w=quat_apply_inverse(rotations[:,self.torso],angular[:,self.torso])
        angular_error=(angular[:,self.torso,2]-command[:,2]).square()+body_w[:,:2].square().sum(-1)
        tracking=linear+torch.exp(-angular_error/3.14**2)
        root_w=quat_apply_inverse(s.root_quat_wxyz,s.root_ang_vel_w)
        root_rotation=torch.exp(-root_w[:,:2].square().sum(-1)/3.14**2)
        height=torch.exp(-(s.root_pos[:,2]-self.height).square()/.3**2)
        acc=(s.joint_vel-old.joint_vel)/self.dt
        foot_vel=[]
        for body,offset in self.feet:
            radius=quat_apply(rotations[:,body],offset.expand(len(action),-1))
            foot_vel.append((velocities[:,body]+torch.cross(angular[:,body],radius,dim=-1))[:,:2])
        foot_vel=torch.stack(foot_vel,1)
        contact=contact_logits[:,:2].sigmoid()
        collision=self.collision_history*contact_logits[:,2].sigmoid()
        active=command[:,:2].norm(dim=-1)+command[:,2].abs()>.1
        return {'tracking':torch.where(delayed,torch.zeros_like(tracking),tracking),
                'root_height':3.5*height*delayed,
                'body_ang_vel_xy':.5*torch.where(delayed,torch.zeros_like(root_rotation),root_rotation),
                'termination':-200*termination,
                'joint_acceleration':-2.5e-7*acc.square().sum(-1),
                'joint_limits':-10*limits,'action_rate':-.01*smooth,
                'foot_slip':-.25*(foot_vel.square().sum(-1)*contact).sum(-1)*active,
                'self_collision':-.1*collision}
