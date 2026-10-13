
import argparse
import random
from pathlib import Path

from agents import SACAgent, DQNAgent
from trainer import Trainer
from env import Environment
from configs import load_config, save_config, RunConfig
from replay_buffer import ReplayBuffer

def create_run_directory(config: RunConfig) -> Path:
    """Create a unique directory for the experiment."""

    runs_dir = Path(config.logging.run_dir)
    game_dir = runs_dir / config.env.name

    game_dir.mkdir(parents=True, exist_ok=True)

    # Generate a unique run directory
    while True:
        name = f"run_{random.randint(1000, 9999)}"
        run_dir = game_dir / name

        try:
            run_dir.mkdir(exist_ok=False)
            break
        except FileExistsError:
            continue

    config.logging.run_dir = str(run_dir)

    print(f"Run directory: {run_dir}")

    return run_dir



def create_agent(config: RunConfig, replay_buffer: ReplayBuffer, env: Environment):
    """Initialize the appropriate RL agent."""

    observation_shape = env.train_env.observation_space.shape

    if len(observation_shape) == 3:
        state_size = observation_shape   # (4, 84, 84)
    else:
        state_size = observation_shape[0]

    if config.algorithm == "sac":
        action_size = env.train_env.action_space.shape[0]

        return SACAgent(
            config, replay_buffer, state_size, action_size
        )

    elif config.algorithm == "dqn":
        action_size = env.train_env.action_space.n

        return DQNAgent(
            config, replay_buffer, state_size, action_size
        )

    else:
        raise ValueError(
            f"Unsupported algorithm: {config.algorithm}"
        )


def main():
    # ==================================
    # Load configuration
    # ==================================

    parser = argparse.ArgumentParser(
        description="Deep Reinforcement Learning Trainer"
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML configuration file"
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Resume training from a checkpoint"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Total number of episodes to train"
    )

    args = parser.parse_args()

    # ==================================
    # Setup experiment
    # ==================================

    if args.checkpoint is not None:

        checkpoint_path = args.checkpoint.resolve()

        if not checkpoint_path.is_file():
            parser.error(
                f"Checkpoint does not exist: {checkpoint_path}"
            )

        # Expected structure:
        # run_1234/checkpoints/latest.pt
        run_dir = checkpoint_path.parent.parent

        config_path = run_dir / "config.yaml"

        if not config_path.is_file():
            parser.error(
                f"Configuration not found: {config_path}"
            )

        if args.config is not None:
            parser.error(
                "Do not specify --config when resuming. "
                "The saved run configuration is loaded automatically."
            )

        config = load_config(config_path)
        config.logging.run_dir = str(run_dir)

        print(f"Resuming experiment from: {checkpoint_path}")

    else:

        config_path = (
            args.config
            if args.config is not None
            else Path("configs/sac.yaml")
        )

        config = load_config(config_path)

        run_dir = create_run_directory(config)

        save_config(
            config,
            run_dir / "config.yaml"
        )

        checkpoint_path = None

    if args.episodes is not None:
        if args.episodes <= 0:
            parser.error("--episodes must be positive")

        config.training.nb_episodes = args.episodes

    # ==================================
    # Initialize environment
    # ==================================

    env = Environment(config)

    try:
        # Initialize replay buffer
        replay_buffer = ReplayBuffer(
            config.replay.capacity,
            config.replay.batch_size
        )

        # Initialize agent
        agent = create_agent(config, replay_buffer, env)

        # Initialize trainer
        trainer = Trainer(agent, env, config, replay_buffer)

        # ==================================
        # Load checkpoint if provided
        # ==================================

        if checkpoint_path is not None:

            checkpoint_info = agent.load_checkpoint(
                checkpoint_path
            )

            trainer.start_episode = (
                checkpoint_info["episode"] + 1
            )

            trainer.global_step = (
                checkpoint_info["global_step"]
            )

            trainer.best_reward = (
                checkpoint_info["best_reward"]
            )

        # ==================================
        # Training
        # ==================================

        trainer.train()

    finally:
        # Close environments
        env.train_env.close()
        env.eval_env.close()


if __name__ == "__main__":
    main()
