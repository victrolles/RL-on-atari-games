from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm
import wandb

from config import Config
from agent import Agent
from env import Environment
from replay_buffer import ReplayBuffer

class Trainer:
    def __init__(self,
                 agent: Agent,
                 env: Environment,
                 config: Config,
                 replay_buffer: ReplayBuffer):
        
        self.agent = agent
        self.env = env
        self.config = config
        self.replay_buffer = replay_buffer
        wandb.init(project="atari-rl", config=asdict(config), dir=config.run_dir)

    def train(self):
        for episode in tqdm(range(self.config.nb_episodes)):
            # Reset the environment
            obs, __import__ = self.env.train_env.reset()
            done = False
            cumulated_reward = 0.0

            while not done:
                # Select an action using the agent
                action = self.agent.select_action(obs)

                # Take a step in the environment
                next_obs, reward, terminated, truncated, _ = self.env.train_env.step(action)     
                cumulated_reward += reward
                done = terminated or truncated

                # Add the experience to the replay buffer
                self.replay_buffer.push(obs, action, reward, next_obs, done)

                # Move to the next observation
                obs = next_obs

            # Update the agent
            cumulated_loss = self.agent.update()

            # evaluate the agent
            if (episode + 1) % 100 == 0:
                mean_reward = self.eval(episode + 1)
                self.agent.save_model(episode + 1, mean_reward)
        
            wandb.log({
                "reward": cumulated_reward,
                "eval_reward": mean_reward if (episode + 1) % 100 == 0 else None,
                "loss": cumulated_loss,
                "epsilon": self.agent.epsilon,
                "buffer_size": len(self.replay_buffer)
            })

    def eval(self, episode: int) -> float:
        
        print(f"\nStarting evaluation for {self.config.nb_evaluation_episodes} episodes...")
        mean_reward = 0.0

        for eval_episode in range(self.config.nb_evaluation_episodes):
            obs, _ = self.env.eval_env.reset()
            done = False
            cumulated_reward = 0.0

            while not done:
                # Select an action using the agent
                action = self.agent.select_action(obs, deterministic=True)

                # Take a step in the environment
                next_obs, reward, terminated, truncated, _ = self.env.eval_env.step(action)
                cumulated_reward += reward
                done = terminated or truncated

                # Move to the next observation
                obs = next_obs

            mean_reward += cumulated_reward
            self.env.eval_env.name_prefix=f"record_episode_{episode}_score_{cumulated_reward}"
            path = Path(self.config.run_dir)
            path = Path.joinpath(path, "records")
            self.env.eval_env.video_folder = path
            print(f"Episode {episode}: Total Reward: {cumulated_reward}")

        mean_reward /= self.config.nb_evaluation_episodes
        return mean_reward