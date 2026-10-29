"""Measured velocity, survival and recovery metrics."""


def summarize(trial):
    velocity = [r for r in trial['episodes'] if r['kind'] == 'velocity']
    recovery = [r for r in trial['episodes'] if r['kind'] == 'recovery']
    mean = lambda rows, key: sum(r[key] for r in rows) / len(rows)
    return {'mean_task_return': mean(velocity, 'p3_task_return'),
            'velocity_survival_rate': mean(velocity, 'survived_full_horizon'),
            'xy_rmse': mean(velocity, 'velocity_xy_rmse_m_s'),
            'yaw_rmse': mean(velocity, 'yaw_rate_rmse_rad_s'),
            'recovery_success_rate': mean(recovery, 'recovery_success')}



def full_summary(trial):
    summary=summarize(trial)
    for kind in ('velocity','recovery'):
        episodes=[e for e in trial['episodes'] if e['kind']==kind]
        summary[kind+'_full_return']=sum(e['full_amp_task_return'] for e in episodes)/len(episodes)
        summary[kind+'_style_reward']=sum(e['style_reward_mean'] for e in episodes)/len(episodes)
    return summary
