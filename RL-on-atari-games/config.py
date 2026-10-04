from dataclasses import dataclass

@dataclass
class Config:
    # Environment settings
    env_name: str = "CartPole-v1"

    # Model settings
    hidden_size: int = 128

    # Training settings
    nb_episodes: int = 1000
    nb_evaluation_episodes: int = 3
    evaluation_interval: int = 100

    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01

    learning_rate: float = 0.001
    gamma: float = 0.99