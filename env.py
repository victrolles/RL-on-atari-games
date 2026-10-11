
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RescaleAction

from configs import RunConfig

class Environment:
    def __init__(self, config: RunConfig):

        # Training environment (no rendering)
        self.train_env = gym.make(config.env.name, render_mode="rgb_array")

        # Evaluation environment (supports video)
        eval_env = gym.make(config.env.name, render_mode="rgb_array")

        # Normalize action space to [-1, 1]
        if config.env.normalize:
            self.train_env = RescaleAction(
                self.train_env,
                min_action=-1.0,
                max_action=1.0
            )

            eval_env = RescaleAction(
                eval_env,
                min_action=-1.0,
                max_action=1.0
            )

        # Video recordings
        record_path = Path(config.logging.run_dir) / "records"
        record_path.mkdir(parents=True, exist_ok=True)

        self.eval_env = RecordVideo(
            eval_env,
            video_folder=str(record_path),
            name_prefix="eval",
            episode_trigger=lambda episode_id: True
        )
