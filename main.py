
import argparse
import random
from pathlib import Path

from agents import SACAgent, DQNAgent
from trainer import Trainer
from env import Environment
from configs import load_config, save_config, Config
from replay_buffer import ReplayBuffer

def create_run_directory(config: Config) -> Path:
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


def create_agent(config, replay_buffer, env):
    """Initialize the appropriate RL agent."""

    state_size = env.train_env.observation_space.shape[0]

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
        default=Path("configs/sac.yaml"),
        help="Path to the YAML configuration file"
    )

    args = parser.parse_args()
    config = load_config(args.config)

    # ==================================
    # Setup experiment
    # ==================================

    run_dir = create_run_directory(config)

    # Save the exact configuration
    save_config(config, run_dir / "config.yaml")

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
        trainer = Trainer(
            agent, env, config, replay_buffer
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
