"""WMRL configuration; shared CLI fields retain their existing spelling."""
from dataclasses import dataclass, field
from mjlab.rl import RslRlBaseRunnerCfg, RslRlModelCfg


@dataclass
class WmrlAlgorithmCfg:
    learning_rate: float = 2e-4
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    gamma: float = .99
    lam: float = .95
    class_name: str = 'WMRL'


@dataclass
class SuperMimicRunnerCfg(RslRlBaseRunnerCfg):
    actor: RslRlModelCfg = field(default_factory=lambda: RslRlModelCfg(
        hidden_dims=(512,256,128), activation='elu', obs_normalization=True))
    critic: RslRlModelCfg = field(default_factory=lambda: RslRlModelCfg(
        hidden_dims=(512,256,128), activation='elu', obs_normalization=True))
    algorithm: WmrlAlgorithmCfg = field(default_factory=WmrlAlgorithmCfg)
    class_name: str = 'SuperMimic'
    logger: str = 'tensorboard'
    max_iterations: int = 100001
    save_interval: int = 1000
    num_steps_per_env: int = 2
    upload_model: bool = False
