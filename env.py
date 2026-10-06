from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from config import Config

class Environment:
    def __init__(self, config: Config):
        self.train_env = gym.make(config.env_name, render_mode="rgb_array")

        eval_env = gym.make(config.env_name, render_mode="rgb_array")
        # Create a directory for recording videos if it doesn't exist
        path = Path(config.run_dir)
        record_path = Path.joinpath(path, "records")
        if not record_path.exists():
            record_path.mkdir(parents=True, exist_ok=True)

        # Add video recording for every episode
        self.eval_env = RecordVideo(
            eval_env,
            video_folder=record_path,
            episode_trigger=lambda episode_id: True  # Record every episode
        )