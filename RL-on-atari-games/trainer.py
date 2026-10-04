from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm
import wandb
from gymnasium.wrappers import RecordVideo

from config import Config
from agent import Agent
from env import Environment

class Trainer:
    def __init__(self,
                 agent: Agent,
                 env: Environment,
                 config: Config):
        
        self.agent = agent
        self.env = env
        self.config = config
        wandb.init(project="atari-rl", config=asdict(config))

    def train(self):
        for episode in tqdm(range(self.config.nb_episodes)):
            # Reset the environment
            obs, __import__ = self.env.train_env.reset()
            done = False
            cumulated_loss = 0.0
            cumulated_reward = 0.0

            while not done:
                # Select an action using the agent
                action = self.agent.select_action(obs)

                # Take a step in the environment
                next_obs, reward, terminated, truncated, _ = self.env.train_env.step(action)
                
                cumulated_reward += reward
                done = terminated or truncated

                # Update the agent with the experience
                loss = self.agent.update(obs, action, reward, next_obs, done)
                cumulated_loss += loss

                # Move to the next observation
                obs = next_obs

            # Decay the exploration rate
            self.agent.epsilon = max(0.01, self.agent.epsilon - 0.001)

            # evaluate the agent
            if (episode + 1) % 100 == 0:
                self.eval()
        
            wandb.log({
                "episode": episode,
                "reward": cumulated_reward,
                "loss": cumulated_loss
            })

    def eval(self):
        
        print(f"Starting evaluation for {self.config.nb_evaluation_episodes} episodes...")
        for episode in range(self.config.nb_evaluation_episodes):
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

            print(f"Episode {episode + 1}: Total Reward: {cumulated_reward}")