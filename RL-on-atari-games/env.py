from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

class Environment:
    def __init__(self, env_name):
        self.train_env = gym.make(env_name, render_mode="human")

        eval_env = gym.make(env_name, render_mode="rgb_array")
        # Create a directory for recording videos if it doesn't exist
        record_path = Path(f"./records/'{env_name}'")
        if not record_path.exists():
            record_path.mkdir(parents=True, exist_ok=True)
        print(f"Videos will be saved to: {record_path.resolve()}")

        # Add video recording for every episode
        self.eval_env = RecordVideo(
            eval_env,
            video_folder=record_path,
            name_prefix=f"eval",
            episode_trigger=lambda episode_id: True  # Record every episode
        )