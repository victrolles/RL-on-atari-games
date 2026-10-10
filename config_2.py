from dataclasses import dataclass


@dataclass
class Config:
    # Environment settings
    env_name: str = "BipedalWalker-v3"
    run_dir: str = ""

    # Model settings
    hidden_size: int = 64
    save_model: bool = True
    load_model: bool = False
    actor_model_path: str = "./runs/CartPole-v1/run_4413/models/actor_model_episode_1600_score_500.00.pth"
    critic_1_model_path: str = "./runs/CartPole-v1/run_4413/models/critic_1_model_episode_1600_score_500.00.pth"
    critic_2_model_path: str = "./runs/CartPole-v1/run_4413/models/critic_2_model_episode_1600_score_500.00.pth"

    # Training settings
    nb_episodes: int = 10000
    nb_evaluation_episodes: int = 2
    evaluation_interval: int = 50

    gamma: float = 0.98
    tau: float = 0.02
    alpha: float = 0.2

    # Optimizers
    actor_lr: float = 3e-3
    critic_lr: float = 3e-3

    # Replay buffer settings
    capacity: int = 10_000
    batch_size: int = 64

    # Initial exploration
    random_steps: int = 10_000

    # Device
    device: str = "cuda"