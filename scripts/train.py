"""Train WMRL (SuperMimic) through the AMP-compatible command line."""

import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import tyro

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlBaseRunnerCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.gpu import select_gpus
from supermimic.cli import SuperMimicConfig


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlBaseRunnerCfg
  backend: Literal["wmrl", "supermimic"] = "wmrl"
  supermimic: SuperMimicConfig = field(default_factory=SuperMimicConfig)
  motion_file: str | None = None
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  torchrunx_log_dir: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str, backend: str = "wmrl") -> "TrainConfig":
    if backend not in ("wmrl", "supermimic"):
      raise ValueError("PPO training backend has been removed; use WMRL (default)")
    from supermimic.cli import configure_defaults
    env_cfg, agent_cfg = configure_defaults(task_id, load_env_cfg(task_id), load_rl_cfg(task_id))
    return TrainConfig(env=env_cfg, agent=agent_cfg, backend=backend)


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + local_rank

  if cfg.backend not in ("wmrl", "supermimic"):
    raise ValueError("PPO training backend has been removed; use WMRL (default)")
  from supermimic.cli import run
  cfg.agent.seed = seed
  cfg.env.seed = seed
  run(task_id, cfg, log_dir, device)


def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)

  # Create log directory once before launching workers.
  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  if num_gpus != 1:
    raise ValueError("WMRL currently supports exactly one CUDA GPU; use --gpu-ids 0")

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  run_train(task_id, args, log_dir)


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  backend_parser = argparse.ArgumentParser(add_help=False)
  backend_parser.add_argument("--backend", choices=("wmrl", "supermimic"), default="wmrl")
  backend_args, _ = backend_parser.parse_known_args(remaining_args)

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task, backend_args.backend),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
