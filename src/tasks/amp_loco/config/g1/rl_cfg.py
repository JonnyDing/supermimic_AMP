"""SuperMimic configuration for AMP locomotion."""

import os
from dataclasses import dataclass, field
from typing import List

from supermimic.agent_config import SuperMimicRunnerCfg


# AMP motion data directory (npz files)
_MOTION_DATA_DIR = os.path.join(
  os.path.dirname(os.path.abspath(__file__)),
  os.pardir, os.pardir, os.pardir, os.pardir, os.pardir,
  "src", "assets", "motions", "g1", "amp",
)


@dataclass
class SuperMimicAmpRunnerCfg(SuperMimicRunnerCfg):
  """Extended runner config with AMP-specific parameters."""
  amp_reward_coef: float = 0.1
  amp_motion_files: str = ""
  amp_num_preload_transitions: int = 200000
  amp_task_reward_lerp: float = 0.75
  amp_discr_hidden_dims: List[int] = field(default_factory=lambda: [1024, 512, 256])
  min_normalized_std: List[float] = field(default_factory=lambda: [0.05] * 29)
  amp_body_names: tuple = ()
  amp_anchor_name: str = ""


def g1_wmrl_runner_cfg() -> SuperMimicAmpRunnerCfg:
  """Create RL runner configuration for Unitree G1 AMP locomotion task."""
  return SuperMimicAmpRunnerCfg(
    experiment_name="g1_amp_locomotion",
    logger="tensorboard",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=100001,
    # AMP parameters
    amp_reward_coef=0.1,
    amp_motion_files=os.path.normpath(_MOTION_DATA_DIR),
    amp_num_preload_transitions=200000,
    amp_task_reward_lerp=0.75,
    amp_discr_hidden_dims=[1024, 512, 256],
    min_normalized_std=[0.05] * 29,
    amp_body_names=(
      "pelvis",
      "left_hip_roll_link",
      "left_knee_link",
      "left_ankle_roll_link",
      "right_hip_roll_link",
      "right_knee_link",
      "right_ankle_roll_link",
      "left_shoulder_roll_link",
      "left_elbow_link",
      "left_wrist_yaw_link",
      "right_shoulder_roll_link",
      "right_elbow_link",
      "right_wrist_yaw_link",
    ),
    amp_anchor_name="torso_link",
  )