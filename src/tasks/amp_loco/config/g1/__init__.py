from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  g1_amp_flat_env_cfg,
  g1_amp_rough_env_cfg,
)
from .rl_cfg import g1_wmrl_runner_cfg

register_mjlab_task(
  task_id="Unitree-G1-AMP-Rough",
  env_cfg=g1_amp_rough_env_cfg(),
  play_env_cfg=g1_amp_rough_env_cfg(play=True),
  rl_cfg=g1_wmrl_runner_cfg(),
)

register_mjlab_task(
  task_id="Unitree-G1-AMP-Flat",
  env_cfg=g1_amp_flat_env_cfg(),
  play_env_cfg=g1_amp_flat_env_cfg(play=True),
  rl_cfg=g1_wmrl_runner_cfg(),
)