from pathlib import Path
import random

from agents.sac.sac_agent import SACAgent
from agents.sac.trainer import Trainer
from env import Environment
from config import Config
from replay_buffer import ReplayBuffer

def main():
    config = Config()

    # Initialize save directory if it doesn't exist
    runs_dir = Path("./runs")
    if not runs_dir.exists():
        runs_dir.mkdir(parents=True, exist_ok=True)

    game_dir = Path.joinpath(runs_dir, config.env_name)
    if not game_dir.exists():
        game_dir.mkdir(parents=True, exist_ok=True)

    # Generate run dir
    name = f"run_{random.randint(1000, 9999)}"
    while Path.joinpath(game_dir, name).exists():
        name = f"run_{random.randint(1000, 9999)}"
    run_dir = Path.joinpath(game_dir, name)
    run_dir.mkdir(parents=True, exist_ok=True)
    config.run_dir = str(run_dir)
    print(f"Run directory: {config.run_dir}")

    # Initialize the environment
    env = Environment(config)

    # Initialize the replay buffer
    replay_buffer = ReplayBuffer(config.capacity, config.batch_size)

    # Initialize the agent
    state_size = env.train_env.observation_space.shape[0]
    action_size = env.train_env.action_space.shape[0]
    agent = SACAgent(config, replay_buffer, state_size, action_size)
    
    # Initialize the trainer
    trainer = Trainer(agent, env, config, replay_buffer)
    
    # Start training
    trainer.train()

    # Close the environment
    env.train_env.close()
    env.eval_env.close()

if __name__ == "__main__":
    main()