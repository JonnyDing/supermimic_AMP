"""SuperMimic world-model RL for G1 velocity tracking (without AMP)."""
from supermimic.agent_config import SuperMimicRunnerCfg


def unitree_g1_wmrl_runner_cfg() -> SuperMimicRunnerCfg:
    return SuperMimicRunnerCfg(experiment_name="g1_supermimic_velocity")
