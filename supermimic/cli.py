"""Configure the WMRL training entry point."""
from dataclasses import dataclass,asdict
from pathlib import Path
from typing import Literal
from mjlab.utils.os import get_checkpoint_path


SUPPORTED_TASKS = ('Unitree-G1-AMP-Flat', 'Unitree-G1-AMP-Rough',
                   'Unitree-G1-Flat', 'Unitree-G1-Rough')


@dataclass
class SuperMimicConfig:
    batch_size: int = 1024
    wm_batch_size: int = 2048
    buffer_steps: int = 256
    warmup_steps: int = 64
    horizon: int = 32
    wm_batches: int = 10
    wm_pretrain_updates: int = 500
    wm_width: int = 2048
    exploration_initial: float = 0.1
    exploration_final: float = 0.1
    exploration_decay_updates: int = 5000
    wm_microbatch: int = 2048
    actor_microbatch: int = 1024
    lr_decay_updates: int = 250000
    utd: float = 0.4
    discriminator_lr: float = 1e-5
    precision: Literal['fp32','tf32','bf16'] = 'bf16'
    compile: bool = True
    log_every: int = 50
    terminal_format: Literal['pretty','json'] = 'pretty'
    resume_refill_steps: int = 32
    evaluate: bool = False
    bootstrap_only: bool = False
    seconds: int = 20


def configure_defaults(task_id,env,agent):
    if task_id not in SUPPORTED_TASKS:
        raise ValueError(f'SuperMimic supports {SUPPORTED_TASKS}, got {task_id}')
    env.scene.num_envs=512 if task_id.endswith('Rough') else 4090
    env.events.pop('push_robot',None)
    env.curriculum={k:v for k,v in env.curriculum.items() if k=='terrain_levels'}
    env.episode_length_s=5.
    agent.experiment_name={'Unitree-G1-AMP-Flat':'g1_supermimic_locomotion',
                           'Unitree-G1-AMP-Rough':'g1_supermimic_amp_rough',
                           'Unitree-G1-Flat':'g1_supermimic_velocity_flat',
                           'Unitree-G1-Rough':'g1_supermimic_velocity_rough'}[task_id]
    agent.save_interval=1000
    agent.load_checkpoint='last.pt'
    agent.num_steps_per_env=2
    agent.algorithm.learning_rate=2e-4
    agent.algorithm.num_learning_epochs=5
    agent.upload_model=False
    return env,agent


def build_arguments(task_id,cfg,log_dir,device):
    if task_id not in SUPPORTED_TASKS:raise ValueError(f'Unsupported SuperMimic task: {task_id}')
    if not device.startswith('cuda:'):raise ValueError('SuperMimic currently requires a CUDA GPU')
    if cfg.video:raise ValueError('SuperMimic training video is not supported; use scripts/play.py with model_<iteration>.pt')
    if cfg.motion_file:raise ValueError('--motion-file is not used by the SuperMimic backend; configure the task motion loader instead')
    if cfg.agent.logger!='tensorboard':raise ValueError('SuperMimic compatibility entry currently supports --agent.logger=tensorboard')
    if tuple(cfg.agent.actor.hidden_dims)!=(512,256,128) or tuple(cfg.agent.critic.hidden_dims)!=(512,256,128):
        raise ValueError('Policy export compatibility requires the existing 512/256/128 policy architecture')
    if cfg.agent.clip_actions is not None:
        raise ValueError('SuperMimic preserves the existing unclipped executed-action contract; clip_actions must be None')
    if cfg.agent.num_steps_per_env<1:raise ValueError('--agent.num-steps-per-env must be positive')
    # Reject agent fields that this backend does not consume.
    from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
    _,reference=configure_defaults(task_id,load_env_cfg(task_id),load_rl_cfg(task_id))
    actual,expected=asdict(cfg.agent),asdict(reference)
    allowed={'seed','max_iterations','save_interval','resume','load_run','load_checkpoint',
             'experiment_name','run_name','logger','num_steps_per_env'}
    algorithm_allowed={'learning_rate','num_learning_epochs','num_mini_batches','gamma','lam'}
    changed=[]
    for key,value in actual.items():
        if key=='algorithm':
            changed.extend('algorithm.'+k for k,v in value.items() if k not in algorithm_allowed and v!=expected[key][k])
        elif key not in allowed and value!=expected[key]:changed.append(key)
    if changed:raise ValueError('Unsupported SuperMimic agent overrides: '+', '.join(changed))
    agent,sm=cfg.agent,cfg.supermimic
    if sm.evaluate and task_id!='Unitree-G1-AMP-Flat':
        raise ValueError('Fixed recovery evaluation supports AMP Flat only; use scripts/play.py for other tasks')
    args=['--task',task_id,'--output',str(log_dir),'--updates',str(agent.max_iterations),'--device',device,
          '--checkpoint-every',str(agent.save_interval),'--logger','tensorboard','--checkpoint-prefix','model',
          '--log-every',str(sm.log_every),
          '--terminal-format',sm.terminal_format,
          '--resume-refill-steps',str(sm.resume_refill_steps),'--seconds',str(sm.seconds),
          '--test-seeds',str(agent.seed)]
    values={'num_envs':cfg.env.scene.num_envs,'seed':agent.seed,'actor_lr':agent.algorithm.learning_rate,
            'utd':sm.utd if agent.num_steps_per_env==reference.num_steps_per_env else 1./agent.num_steps_per_env,'critic_epochs':agent.algorithm.num_learning_epochs,
            'critic_minibatches':agent.algorithm.num_mini_batches,'gamma':agent.algorithm.gamma,
            'gae_lambda':agent.algorithm.lam}
    values.update({k:v for k,v in asdict(sm).items() if k in ('batch_size','wm_batch_size','buffer_steps','warmup_steps','horizon','wm_batches','wm_pretrain_updates','wm_width','exploration_initial','exploration_final','exploration_decay_updates','wm_microbatch','actor_microbatch','lr_decay_updates','discriminator_lr','precision')})
    for key,value in values.items():args.extend(['--'+key.replace('_','-'),str(value)])
    args.append('--compile' if sm.compile else '--no-compile')
    if sm.evaluate:args.append('--evaluate')
    if sm.bootstrap_only:args.append('--bootstrap-only')
    if agent.resume:
        path=get_checkpoint_path(Path(log_dir).parent,agent.load_run,agent.load_checkpoint)
        args.extend(['--resume',str(path)])
    return args


def run(task_id,cfg,log_dir,device):
    from supermimic.train_fast import main
    args=build_arguments(task_id,cfg,log_dir,device)
    print('[INFO] Backend: WMRL / SuperMimic (online world model + imagined BPTT)',flush=True)
    print(f'[INFO] Checkpoints: {log_dir}/model_<iteration>.pt; full resume: last.pt',flush=True)
    main(args,env_cfg=cfg.env,compatibility_config={'task':task_id,'env':asdict(cfg.env),'agent':asdict(cfg.agent)})
