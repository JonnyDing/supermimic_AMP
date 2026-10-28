"""P2 diagnostic: collect PPO+noise, fit dynamics only, evaluate held-out worlds.

Not SuperMimic policy training: PPO is only a data source for verifying the learned
simulator. No actor or discriminator updates. Default network is deliberately small.
"""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from supermimic.adapters.mjlab import CaptureEnv, PhysicsCollector
from supermimic.core.replay import SequenceReplay
from supermimic.core.state import pack
from supermimic.core.world_model import WorldModel, rollout_loss
from supermimic.evaluation.world_model import evaluate, DYNAMICS_KEYS
from supermimic.core.policy import load_policy, ROOT
from supermimic.evaluation.baseline import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--steps', type=int, default=512)
    parser.add_argument('--updates', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--save-replay', action='store_true')
    args = parser.parse_args()
    if args.num_envs < 8 or args.steps < 32 or args.updates < 1:
        parser.error('Need >=8 environments, >=32 collection steps and >=1 update')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision('highest')  # Match the frozen PPO baseline during collection
    from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_env_cfg
    manifest = json.loads((ROOT / 'supermimic/reports/baseline_manifest.json').read_text())
    checkpoint = ROOT / manifest['checkpoint']['path']
    if sha256(checkpoint) != manifest['checkpoint']['sha256']:
        raise ValueError('Frozen PPO checkpoint changed')
    policy, normalizer = load_policy(checkpoint, args.device)
    cfg = g1_amp_flat_env_cfg()
    cfg.seed = args.seed; cfg.scene.num_envs = args.num_envs
    cfg.events.pop('push_robot', None)
    cfg.curriculum = {}
    cfg.episode_length_s = 5.0
    env = CaptureEnv(cfg=cfg, device=args.device)
    replay = SequenceReplay(args.steps, args.num_envs, args.device)
    start = time.perf_counter()
    try:
        env.reset(seed=args.seed)
        collector = PhysicsCollector(env)
        partition_rng = torch.Generator().manual_seed(args.seed)
        permutation = torch.randperm(args.num_envs, generator=partition_rng).tolist()
        split = args.num_envs * 3 // 4
        train_ids, validation_ids = permutation[:split], permutation[split:]
        metadata = {'schema_version': 1, 'data_source': 'frozen PPO + independent Gaussian action noise std=0.1',
                    'actor_training': False, 'pushes': False, 'control_dt': env.step_dt,
                    'seed': args.seed, 'num_envs': args.num_envs, 'collection_steps': args.steps,
                    'train_env_ids': train_ids, 'validation_env_ids': validation_ids,
                    'recovery_env_ids': env.termination_manager._delay_env_mask.nonzero().flatten().tolist(),
                    'joint_names': list(env.scene['robot'].joint_names),
                    'context_schema': collector.context_schema, 'checkpoint_source': manifest['checkpoint'],
                    'contact_labels': 'feet contact flags and any self-collision >10N in final control-step history',
                    'config': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}}
        (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        rejected = []
        with torch.no_grad():
            for step in range(args.steps):
                for attempt in range(4):
                    observation = env.obs_buf['actor']
                    action = policy.act_inference(normalizer(observation))
                    good_action = torch.isfinite(action).all(-1) & (action.abs().amax(-1) < 100)
                    if good_action.all():
                        break
                    bad_ids = (~good_action).nonzero().flatten()
                    rejected.append({'step': step, 'kind': 'policy_action_guard', 'env_ids': bad_ids.tolist()})
                    print(f'reset invalid PPO action source: {rejected[-1]}', flush=True)
                    env.reset(env_ids=bad_ids)
                    env.termination_manager._delay_counters[bad_ids] = 0
                    collector.episode_ids[bad_ids] += 1
                else:
                    raise FloatingPointError(f'PPO remained invalid after 4 resets at step {step}')
                action = action + torch.randn_like(action) * 0.1
                transition = collector.step(action)
                source, target = pack(transition.state), pack(transition.next_state)
                valid = torch.isfinite(source).all(-1) & torch.isfinite(target).all(-1)
                # Generous numerical-stability limits, not task fall thresholds.
                for physics in (transition.state, transition.next_state):
                    valid &= physics.root_lin_vel_w.abs().amax(-1) < 1000
                    valid &= physics.root_ang_vel_w.abs().amax(-1) < 1000
                    valid &= physics.joint_vel.abs().amax(-1) < 1000
                for value in transition.labels.values():
                    valid &= torch.isfinite(value).reshape(args.num_envs, -1).all(-1)
                replay.add(transition, dynamics_valid=valid)
                bad_ids = (~valid).nonzero().flatten()
                if len(bad_ids):
                    rejected.append({'step': step, 'kind': 'physics_guard', 'env_ids': bad_ids.tolist()})
                    print(f'reject unstable dynamics: {rejected[-1]}', flush=True)
                    env.reset(env_ids=bad_ids)
                    env.termination_manager._delay_counters[bad_ids] = 0
                    collector.episode_ids[bad_ids] += 1
                if (step + 1) % 128 == 0:
                    print(f'collect {step+1}/{args.steps}', flush=True)
        torch.cuda.synchronize() if str(args.device).startswith('cuda') else None
        collection_seconds = time.perf_counter() - start
        metadata['rejected_transitions'] = rejected
        metadata['numerical_guard'] = 'Reset source on nonfinite PPO action or abs(action)>=100 before executing; reject nonfinite physics/contact labels or abs velocity>=1000 after executing. Never fit invalid transitions or cross these resets.'
        metadata['numerical_reset_history_note'] = 'mjlab reset recomputes observation history for the batch; extra observation updates occur only on recorded numerical resets'
        (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    finally:
        env.close()
    del env, policy, normalizer
    if args.save_replay:
        torch.save(replay.state_dict(), args.output / 'replay.pt')
    torch.set_float32_matmul_precision('high')  # WM matmuls only
    model = WorldModel(replay.storage['action'].shape[-1], replay.storage['context'].shape[-1],
                       hidden=(args.hidden,)*3).to(args.device)
    # Fit statistics on training environments ONLY, never validation rows.
    rows = replay.rows(torch.arange(replay.size, device=args.device))
    train = {k: replay.storage[k][rows][:, train_ids].flatten(0, 1) for k in ('state', 'next_state', 'action', 'context')}
    valid_train = replay.storage['dynamics_valid'][rows][:, train_ids].flatten()
    model.fit_stats(**{k: v[valid_train] for k, v in train.items()})
    del train
    initial = evaluate(model, replay, validation_ids)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    history = []
    start = time.perf_counter()
    for update in range(args.updates):
        window = replay.sample(args.batch_size, 8, train_ids, DYNAMICS_KEYS)
        loss, physics, contact = rollout_loss(model, window)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite loss at update {update}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        history.append({'update': update+1, 'loss': loss.item(), 'physics_loss': physics.item(),
                        'contact_loss': contact.item(), 'gradient_norm_before_clip': grad_norm.item()})
        if (update+1) % 50 == 0:
            print(f'update {update+1}/{args.updates} state_loss={physics.item():.4f} contact={contact.item():.4f}', flush=True)
    torch.cuda.synchronize() if str(args.device).startswith('cuda') else None
    training_seconds = time.perf_counter() - start
    final = evaluate(model, replay, validation_ids)
    checkpoint_path = args.output / 'world_model.pt'
    torch.save({'schema_version': 1, 'architecture': model.architecture(), 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'update': args.updates, 'metadata': metadata,
                'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                'replay_saved': args.save_replay}, checkpoint_path)
    # Loading must reproduce exactly the deployed state integration and normalization.
    saved = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    restored = WorldModel(**saved['architecture']).to(args.device)
    restored.load_state_dict(saved['model'])
    sample = replay.sample(16, 1, validation_ids, DYNAMICS_KEYS)
    with torch.no_grad():
        call = lambda m: m(sample['state'][:, 0], sample['action'][:, 0], sample['context'][:, 0])[0]
        reload_error = (call(model) - call(restored)).abs().max().item()
    if reload_error != 0:
        raise AssertionError(f'Checkpoint reload mismatch: {reload_error}')
    result = {'status': 'completed', 'stage': 'P2_dynamics_diagnostic', 'metadata': metadata,
              'architecture': model.architecture(), 'collection_seconds': collection_seconds,
              'training_seconds': training_seconds, 'reload_max_error': reload_error,
              'history': history, 'initial_validation': initial, 'final_validation': final,
              'world_model_checkpoint': str(checkpoint_path),
              'limitations': ['Offline diagnostic dataset from PPO, not online joint actor/world-model training',
                              '512-width default is smaller than paper 2048-width dynamics',
                              'No random pushes; startup physical randomization remains enabled',
                              'Held-out environments share motion library but not replay samples or normalization fitting',
                              '32-step accuracy is diagnostic, not a guarantee of safe BPTT policy optimization']}
    (args.output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({'collection_seconds': collection_seconds, 'training_seconds': training_seconds,
                      'validation_8_all': final['8']['all'], 'reload_error': reload_error}, indent=2))


if __name__ == '__main__':
    main()
