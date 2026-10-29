"""Data-only normalization and model pretraining before any actor gradient."""
import torch
from .world_model import rollout_loss, DYNAMICS_KEYS




@torch.no_grad()
def fit_initial_statistics(wm, normalizer, replay, train_ids, batch=8192):
    """Fit only valid training transitions; holdout environments never enter stats."""
    window = replay.sample(batch, 1, train_ids,
                           ['state', 'next_state', 'action', 'context', 'task/observation_history', 'next_task/observation_history', 'reward', 'terminated', 'truncated'])
    if hasattr(wm, 'update_task_stats'):
        wm.update_task_stats(window['next_task/observation_history'], window['reward'],
                             ~(window['terminated'] | window['truncated']))
    wm.fit_stats(window['state'], window['next_state'], window['action'], window['context'])
    if int(normalizer.count)==0: normalizer.update(window['task/observation_history'][:, 0].flatten(1))
    normalizer.eval()


def exploration_std(update, initial, final, decay_updates):
    fraction = min(max(update, 0) / max(decay_updates, 1), 1.)
    return initial + fraction * (final - initial)


def pretrain(wm, optimizer, discriminator, collector, train_ids, holdout_ids,
             updates, batch, log_every=50, writer=None, wm_batches=10, microbatch=2048, utd=.4,
             loss_fn=None, precision='fp32'):
    """Learn dynamics and AMP before bootstrapping a critic or updating the actor.

    Use the reference eight-step loss and full effective world-model batches.
    Collect with the current policy and exploration noise, as in the reference.
    """
    replay = collector.replay
    # Reuse the trainer's compiled loss; eager eight-step BPTT otherwise launches
    # all of its small operations from Python for every microbatch.
    if loss_fn is None:
        loss_fn = lambda window: rollout_loss(wm, window)
    keys = getattr(wm, 'replay_keys', DYNAMICS_KEYS)
    validation = replay.sample_prefix(batch, 8, holdout_ids, keys)
    with torch.no_grad():
        initial_loss = float(loss_fn(validation)[0])
    from .batching import world_model_update
    accumulator=0.
    for step in range(1, updates + 1):
        accumulator+=1/utd; count=int(accumulator); accumulator-=count
        torch.set_float32_matmul_precision('highest')
        collector.collect(count, 0)
        torch.set_float32_matmul_precision('high' if precision in ('tf32','bf16') else 'highest')
        for _ in range(wm_batches):
            window = replay.sample_prefix(batch, 8, train_ids, keys)
            loss=world_model_update(wm,optimizer,window,loss_fn,microbatch)
        metrics = discriminator.update(replay, train_ids, batch, return_tensors=True) if discriminator is not None else {}
        if step == 1 or step % log_every == 0 or step == updates:
            with torch.no_grad():
                heldout = loss_fn(validation)[0]
            print(f'[Bootstrap] {step}/{updates} WM={float(loss):.5f} heldout8={float(heldout):.5f}', flush=True)
            if writer is not None:
                for key, value in dict(wm_loss=loss.detach(), heldout_loss=heldout, **metrics).items():
                    writer.add_scalar('Bootstrap/' + key, float(value), step)
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        final_loss = float(loss_fn(validation)[0])
    return {'updates': updates, 'initial_heldout_loss': initial_loss,
            'final_heldout_loss': final_loss, 'actor_updated': False}
