
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from config_2 import Config
from replay_buffer import ReplayBuffer
from agents import BaseAgent
    
class DQLAgent(BaseAgent):
    def __init__(self,
                 config: Config,
                 replay_buffer: ReplayBuffer,
                 state_size: int,
                 action_size: int):

        super().__init__(config, replay_buffer)

        self.model = DQLAgent(state_size, config.hidden_size, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=50)
        self.epsilon = config.epsilon

        if self.config.load_model:
            self.load_model(self.config.model_path)
            

    def select_action(self, state, deterministic=False):
        if torch.rand(1).item() < self.epsilon and not deterministic:
            # Explore: select a random action
            action = torch.randint(0, self.model.fully_connected_layers[-1].out_features, (1,)).item()
        else:
            # Exploit: select the action with the highest Q-value
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.model(state)
                action = torch.argmax(q_values).item()
        return action

    def update(self):

        if len(self.replay_buffer) < self.config.batch_size:
            return 0.0  # Not enough samples to update
        else :
            accumulated_loss = 0.0
            for _ in range(self.config.training_steps):
                # Sample a batch of experiences from the replay buffer
                states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)

                # Convert batch to tensors
                states = torch.from_numpy(np.asarray(states, dtype=np.float32)).to(self.device)

                actions = torch.tensor(
                    actions,
                    dtype=torch.long,
                    device=self.device
                ).unsqueeze(1)

                rewards = torch.tensor(
                    rewards,
                    dtype=torch.float32,
                    device=self.device
                ).unsqueeze(1)

                next_states = torch.from_numpy(
                    np.asarray(next_states, dtype=np.float32)
                ).to(self.device)

                dones = torch.tensor(
                    dones,
                    dtype=torch.float32,
                    device=self.device
                ).unsqueeze(1)

                # Compute the target Q-values
                with torch.no_grad():
                    target_q_values = rewards + (self.config.gamma * torch.max(self.model(next_states), dim=1, keepdim=True)[0] * (1 - dones))

                # Compute the current Q-values
                current_q_values = self.model(states).gather(1, actions)

                # Compute the loss
                loss = self.criterion(current_q_values, target_q_values)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                accumulated_loss += loss.item()

            # Decay epsilon
            self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)

            # Update the learning rate scheduler
            self.scheduler.step()

            return accumulated_loss / self.config.training_steps

    def save_model(self, episode: int, reward: float):
        path = Path(self.config.run_dir)
        path = Path.joinpath(path, "models")
        model_name = f"model_episode_{episode}_reward_{reward:.2f}.pth"
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        full_path = Path.joinpath(path, model_name)
        torch.save(self.model.state_dict(), full_path)

    def load_model(self, path: str):
        print(f"Loading model from {path}...")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        