"""Shared AMP task summaries and fixed-baseline acceptance metrics."""
import math
from .bptt import summarize
from .acceptance import compare


def full_summary(trial):
    summary=summarize(trial)
    for kind in ('velocity','recovery'):
        episodes=[e for e in trial['episodes'] if e['kind']==kind]
        summary[kind+'_full_return']=sum(e['full_amp_task_return'] for e in episodes)/len(episodes)
        summary[kind+'_style_reward']=sum(e['style_reward_mean'] for e in episodes)/len(episodes)
    return summary


def acceptance(b,c):
    result=compare(b,c)
    # The original fixed discriminator is the common judge; never compare moving rewards.
    result['checks']['full_metrics_finite']=all(math.isfinite(m[k]) for m in (b,c) for k in ('velocity_full_return','recovery_full_return','velocity_style_reward','recovery_style_reward'))
    result['checks']['full_velocity_return']=c['velocity_full_return']>=b['velocity_full_return']-.02*abs(b['velocity_full_return'])
    result['checks']['full_recovery_return']=c['recovery_full_return']>=b['recovery_full_return']-.02*abs(b['recovery_full_return'])
    result['passed']=all(result['checks'].values())
    return result
