from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm
import wandb

from configs import RunConfig
from replay_buffer import ReplayBuffer
from agents import Agent
from env import Environment

class Trainer:
    def __init__(self,
                 agent: Agent,
                 env: Environment,
                 config: RunConfig,
                 replay_buffer: ReplayBuffer):
        
        self.agent = agent
        self.env = env
        self.config = config
        self.replay_buffer = replay_buffer

        self.start_episode = 0
        self.global_step = 0
        self.best_reward = float("-inf")

        wandb.init(project="atari-rl", config=asdict(config), dir=config.logging.run_dir)

    def train(self):
        for episode in tqdm(range(self.start_episode, self.config.training.nb_episodes)):
            # Reset the environment
            state, _ = self.env.train_env.reset()
            done = False
            cumulated_reward = 0.0
            steps_per_episode = 0
            metrics = {}

            # ==================================
            # Episode loop
            # ==================================

            while not done:

                # Select an action using the agent
                if (self.global_step < self.config.training.random_steps):
                    action = self.env.train_env.action_space.sample()
                else:
                    action = self.agent.select_action(state)

                # Take a step in the environment
                next_state, reward, terminated, truncated, _ = self.env.train_env.step(action)
                cumulated_reward += reward
                done = terminated or truncated

                # Store the experience to the replay buffer
                self.replay_buffer.push(state, action, reward, next_state, terminated)

                # Move to the next observation
                state = next_state
                steps_per_episode += 1
                self.global_step += 1

                # Update the agent
                if (
                    self.global_step >= self.config.training.learning_starts
                    and len(self.replay_buffer) >= self.config.replay.batch_size
                    and self.global_step % self.config.training.train_freq == 0
                ):

                    for _ in range(self.config.training.gradient_steps):
                        metrics = self.agent.update()
                        wandb.log(metrics)

            # evaluate the agent
            if (episode + 1) % self.config.evaluation.nb_interval == 0:
                mean_reward = self.eval(episode + 1)
                # checkpoint the best model
                if mean_reward > self.best_reward:
                    self.best_reward = mean_reward

                    self.agent.save_checkpoint(
                        episode=episode,
                        global_step=self.global_step,
                        best_reward=self.best_reward,
                        filename="best.pt"
                    )

            # checkpoint the agent
            if (episode + 1) % self.config.logging.checkpoint_interval == 0:
                self.agent.save_checkpoint(
                    episode=episode,
                    global_step=self.global_step,
                    best_reward=cumulated_reward,
                    filename="latest.pt"
                )
        
            wandb.log({
                "Steps total": self.global_step,
                "Steps per episode": steps_per_episode,
                "Reward training": cumulated_reward,
                "Reward evaluation": mean_reward if (episode + 1) % self.config.evaluation.nb_interval == 0 else None,
                "Buffer Size": len(self.replay_buffer)
            })

    def eval(self, episode: int) -> float:
            mean_reward = 0.0
    
            for _ in range(self.config.evaluation.nb_episodes):
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
                path = Path(self.config.logging.run_dir)
                path = Path.joinpath(path, "records")
                self.env.eval_env.video_folder = path
                print(f"Episode {episode}: Total Reward: {cumulated_reward}")
    
            mean_reward /= self.config.evaluation.nb_episodes
            return mean_reward