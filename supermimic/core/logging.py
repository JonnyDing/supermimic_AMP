"""Readable terminal progress; numerical diagnostics are TensorBoard-only."""
import json


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def progress(metrics, total, terminal_format='pretty'):
    hidden = {'critic_target_abs_max', 'critic_value_abs_max', 'critic_target_std',
              'critic_input_abs_max', 'imagined_reward_abs_max',
              'critic_mse_before', 'critic_alive_weighted_mse'}
    terminal_metrics = {k:v for k,v in metrics.items() if k not in hidden}
    if terminal_format == 'json':
        print(json.dumps(terminal_metrics, allow_nan=False), flush=True)
        return
    m = metrics
    eta = 'warming up' if m['first_interval_includes_compile'] else duration(m['eta_seconds'])
    lines = [
        f"\n{'=' * 72}",
        f"SuperMimic  {m['update']:,}/{total:,}  ({100*m['update']/total:.1f}%)"
        f"  elapsed {duration(m['elapsed_seconds'])}  ETA {eta}",
        f"Speed     {m['updates_per_second']:.2f} updates/s | {m['transitions_per_second']:,.0f} transitions/s"
        f" | total {m['transitions']:,}",
        f"Loss      critic {m['critic_loss']:.5g} (interval peak {m['critic_loss_peak']:.5g})"
        f" | WM {m['wm_loss']:.5g}"
        + (f" | discriminator {m['discriminator_loss']:.5g}" if 'discriminator_loss' in m else ''),
        f"Actor     imagined return {m['imagined_return']:.5g}"
        f" | grad norm {m['actor_gradient_norm']:.5g} (peak {m['actor_gradient_norm_peak']:.5g})"
        f" | recovery {100*m['recovery_mode_fraction']:.1f}%",
    ]
    print('\n'.join(lines), flush=True)
