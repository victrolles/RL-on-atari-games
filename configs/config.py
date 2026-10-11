from dataclasses import dataclass, field
from typing import Literal
from dataclasses import asdict
from pathlib import Path

import yaml

# =====================================
# Common configurations
# =====================================

@dataclass
class EnvConfig:
    name: str = "CartPole-v1"
    device: str = "cuda"
    normalize: bool = False


@dataclass
class NetworkConfig:
    hidden_size: int = 256

@dataclass
class TrainingConfig:
    nb_episodes: int = 1000

    learning_starts: int = 1000
    random_steps: int = 1000

    train_freq: int = 1
    gradient_steps: int = 1

@dataclass
class EvaluationConfig:
    nb_episodes: int = 3
    nb_interval: int = 100

@dataclass
class ReplayConfig:
    capacity: int = 100_000
    batch_size: int = 256


@dataclass
class LoggingConfig:
    run_dir: str = "./runs"
    wandb_project: str = "atari-rl"
    checkpoint_interval: int = 50


# =====================================
# Algorithm-specific configurations
# =====================================

@dataclass
class SACConfig:
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2

    log_std_min: float = -20.0
    log_std_max: float = 2.0


@dataclass
class DQNConfig:
    learning_rate: float = 1e-3
    gamma: float = 0.99

    epsilon: float = 1.0
    epsilon_decay: float = 0.999
    epsilon_min: float = 0.01


# =====================================
# Complete experiment configurations
# =====================================

@dataclass
class BaseConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


@dataclass
class SACRunConfig(BaseConfig):
    algorithm: Literal["sac"] = "sac"
    agent: SACConfig = field(default_factory=SACConfig)


@dataclass
class DQNRunConfig(BaseConfig):
    algorithm: Literal["dqn"] = "dqn"
    agent: DQNConfig = field(default_factory=DQNConfig)


RunConfig = SACRunConfig | DQNRunConfig



def save_config(config: RunConfig, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            asdict(config),
            f,
            sort_keys=False
        )


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Configuration must be a YAML mapping")

    config_classes = {
        "sac": SACRunConfig,
        "dqn": DQNRunConfig,
    }

    algorithm = data.get("algorithm")

    if algorithm not in config_classes:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Start with the algorithm-specific defaults
    config = config_classes[algorithm]()

    sections = (
        "env", "network", "training", "evaluation",
        "replay", "logging", "agent"
    )

    unknown = set(data) - set(sections) - {"algorithm"}
    if unknown:
        raise ValueError(f"Unknown config sections: {unknown}")

    for section in sections:
        overrides = data.get(section, {})

        if not isinstance(overrides, dict):
            raise ValueError(f"{section} must be a YAML mapping")

        current = getattr(config, section)

        # Preserve default values for omitted YAML fields
        values = {**asdict(current), **overrides}

        # Reconstruct the nested dataclass
        setattr(config, section, type(current)(**values))

    return config
