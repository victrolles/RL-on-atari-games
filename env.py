
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import (
    RecordVideo,
    RescaleAction,
    GrayscaleObservation,
    ResizeObservation,
    FrameStackObservation,
)

from configs import RunConfig


class Environment:
    def __init__(self, config: RunConfig):

        def create_env():
            env = gym.make(
                config.env.name,
                render_mode="rgb_array"
            )

            # Preprocessing for image observations
            if config.env.name == "CarRacing-v3":
                env = GrayscaleObservation(env, keep_dim=False)
                env = ResizeObservation(env, shape=(84, 84))
                env = FrameStackObservation(env, stack_size=4)

            # Normalize actions to [-1, 1]
            if config.env.normalize:
                env = RescaleAction(
                    env,
                    min_action=-1.0,
                    max_action=1.0
                )

            return env

        self.train_env = create_env()
        self.eval_env = create_env()
        record_env = create_env()
        

        # Video recording
        record_path = Path(config.logging.run_dir) / "records"
        record_path.mkdir(parents=True, exist_ok=True)

        self.record_env = RecordVideo(
            record_env,
            video_folder=str(record_path),
            name_prefix="record",
            episode_trigger=lambda episode_id: True
        )
