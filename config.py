from dataclasses import dataclass

@dataclass
class Config:
    # Environment settings
    env_name: str = "CartPole-v1"
    run_dir: str = ""

    # Model settings
    hidden_size: int = 64
    save_model: bool = True
    load_model: bool = True
    model_path: str = "./runs/CartPole-v1/run_4413/models/model_episode_1600_score_500.00.pth"

    # Training settings
    nb_episodes: int = 10000
    nb_evaluation_episodes: int = 3
    evaluation_interval: int = 100

    epsilon: float = 1.0
    epsilon_decay: float = 0.999
    epsilon_min: float = 0.01

    learning_rate: float = 1e-2
    gamma: float = 0.99
    training_steps: int = 10

    # Replay buffer settings
    capacity: int = 1000
    batch_size: int = 64