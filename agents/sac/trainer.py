from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm
import wandb

from config import Config
from replay_buffer import ReplayBuffer
from agents.sac.sac_agent import SACAgent
from env import Environment

class Trainer:
    def __init__(self,
                 agent: SACAgent,
                 env: Environment,
                 config: Config,
                 replay_buffer: ReplayBuffer):
        
        self.agent = agent
        self.env = env
        self.config = config
        self.replay_buffer = replay_buffer
        wandb.init(project="atari-rl", config=asdict(config), dir=config.run_dir)

    def train(self):
        total_steps = 0

        for episode in tqdm(range(self.config.nb_episodes)):
            # Reset the environment
            state, _ = self.env.train_env.reset()
            done = False
            cumulated_reward = 0.0
            steps_per_episode = 0
            last_losses = None

            # ==================================
            # Episode loop
            # ==================================

            while not done:

                # Select an action using the agent
                if (total_steps < self.config.random_steps):
                    action = self.env.train_env.action_space.sample()
                else:
                    action = self.agent.select_action(state)

                # Take a step in the environment
                next_state, reward, terminated, truncated, _ = self.env.train_env.step(action)
                cumulated_reward += reward
                done = terminated or truncated

                # Store the experience to the replay buffer
                self.replay_buffer.push(state, action, reward, next_state, done)

                # Move to the next observation
                state = next_state
                steps_per_episode += 1
                total_steps += 1

                # Update the agent
                if (
                    total_steps >= self.config.learning_starts
                    and len(self.replay_buffer) >= self.config.batch_size
                    and total_steps % self.config.train_freq == 0
                ):

                    for _ in range(self.config.gradient_steps):
                        last_losses = self.agent.train()

            # evaluate the agent
            if (episode + 1) % self.config.evaluation_interval == 0:
                mean_reward = self.eval(episode + 1)
                self.agent.save_model(episode + 1, mean_reward)
        
            wandb.log({
                "Steps total": total_steps,
                "Steps per episode": steps_per_episode,
                "Reward training": cumulated_reward,
                "Reward evaluation": mean_reward if (episode + 1) % self.config.evaluation_interval == 0 else None,
                "Loss Actor": last_losses['actor_loss'] if last_losses is not None else None,
                "Loss Critic 1": last_losses['critic_1_loss'] if last_losses is not None else None,
                "Loss Critic 2": last_losses['critic_2_loss'] if last_losses is not None else None,
                "Buffer Size": len(self.replay_buffer),
                "Mean Q": last_losses["mean_q"] if last_losses else None,
                "STD": last_losses["std"] if last_losses else None,
                "Log STD": last_losses["log_std"] if last_losses else None,
                "Log Prob": last_losses["log_prob"] if last_losses else None,
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