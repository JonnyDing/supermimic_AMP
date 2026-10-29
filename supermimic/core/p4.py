"""AMP/recovery rollout inputs and real-state sampling."""
import torch
from .imagination import critic_input

TASK_KEYS=['command','heading_target','command_time_left','is_heading_env','is_standing_env',
           'recovery_enabled','recovery_counter','episode_step']
START_KEYS=['state','context','task/observation_history','task/previous_action']+['task/'+k for k in TASK_KEYS]


def value_input(wm,state,previous,context,command,task,heads):
    extra=torch.stack((task['recovery_enabled'].float(),task['recovery_counter']/max(heads.maximum,1),
                       task['episode_step']/heads.max_episode_steps,task['command_time_left']/8,
                       task['heading_target'].sin(),task['heading_target'].cos(),
                       task['is_heading_env'].float(),task['is_standing_env'].float()),-1)
    return torch.cat((critic_input(wm,state,previous,context,command),extra),-1)


def sample_starts(replay,ids,heads,batch=128,horizon=8):
    # Imagination needs one real initial state, not H future surviving transitions.
    # Requiring a surviving real window silently excludes pre-fall states.
    ages,envs=replay.valid_starts(1,ids); rows=replay.rows(ages)
    good=replay.storage['task/command_time_left'][rows,envs]>horizon*heads.dt
    ages,envs=ages[good],envs[good]; rows=replay.rows(ages)
    # Guarantee recovery data actually participates, without sampling held-out envs.
    recovery=replay.storage['task/recovery_enabled'][rows,envs]
    picks=[]
    for mask,count in ((recovery,int(.4*batch)),(~recovery,batch-int(.4*batch))):
        candidates=mask.nonzero().flatten()
        if len(candidates) and count:
            picks.append(candidates[torch.randint(len(candidates),(count,),device=replay.device)])
    if not picks:
        raise ValueError('No imagination starts before command resampling; collect more data')
    chosen=torch.cat(picks)
    if len(chosen)<batch:
        chosen=chosen[torch.randint(len(chosen),(batch,),device=replay.device)]
    return replay.gather(ages[chosen],envs[chosen],1,START_KEYS)
