
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from configs import DQNRunConfig
from replay_buffer import ReplayBuffer
from agents.base_agent import BaseAgent
from agents.dqn.dqn import DQN
    
class DQNAgent(BaseAgent):
    def __init__(self,
                 config: DQNRunConfig,
                 replay_buffer: ReplayBuffer,
                 state_size: int,
                 action_size: int):

        super().__init__(config, replay_buffer, state_size, action_size)

        self.model = DQN(state_size, config.network.hidden_size, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.agent.learning_rate)
        self.criterion = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=50)
        self.epsilon = config.agent.epsilon

    def select_action(self, state, deterministic: bool = False):
        if torch.rand(1).item() < self.epsilon and not deterministic:
            # Explore: select a random action
            action = torch.randint(0, self.model.fully_connected_layers[-1].out_features, (1,)).item()
        else:
            # Exploit: select the action with the highest Q-value
            with torch.no_grad():
                state = torch.as_tensor(
                    state,
                    dtype=torch.float32,
                    device=self.device
                ).unsqueeze(0)
                q_values = self.model(state)
                action = torch.argmax(q_values).item()
        return action

    def update(self) -> dict:

        # Sample a batch of experiences from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

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
            target_q_values = rewards + (self.config.agent.gamma * torch.max(self.model(next_states), dim=1, keepdim=True)[0] * (1 - dones))

        # Compute the current Q-values
        current_q_values = self.model(states).gather(1, actions)

        # Compute the loss
        loss = self.criterion(current_q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the learning rate scheduler
        self.scheduler.step()

        #Update epsilon for exploration-exploitation trade-off
        self.epsilon = max(self.config.agent.epsilon_min, self.epsilon * self.config.agent.epsilon_decay)

        return {
            "loss": loss.item()
        }

    def save_checkpoint(
        self,
        episode: int,
        global_step: int,
        best_reward: float,
        filename: str = "latest.pt",
    ):
        checkpoint_dir = (
            Path(self.config.logging.run_dir)
            / "checkpoints"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            # Training progress
            "episode": int(episode),
            "global_step": int(global_step),
            "best_reward": float(best_reward),

            # Q-network
            "model": self.model.state_dict(),

            # Optimizer
            "optimizer": self.optimizer.state_dict(),

            # Learning rate scheduler
            "scheduler": self.scheduler.state_dict(),

            # Exploration
            "epsilon": self.epsilon,
        }

        path = checkpoint_dir / filename

        # Save to a temporary file first
        temporary_path = path.with_suffix(".tmp")
        torch.save(checkpoint, temporary_path)

        # Atomically replace the previous checkpoint
        temporary_path.replace(path)

        print(f"Checkpoint saved: {path}")

    # Save the model weights only (without optimizer states or training progress)
    def save_model(self, episode: int, reward: float):
        
        # Create the model directory if it doesn't exist
        model_dir = Path(self.config.logging.run_dir) / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save the model weights
        model_path = model_dir / f"model_episode_{episode}_reward_{reward:.2f}.pt"
        torch.save({
            "env_name": self.config.env.name,
            "algorithm": self.config.algorithm,
            "state_size": self.state_size,
            "action_size": int(self.action_size),
            "hidden_size": self.config.network.hidden_size,
            "normalize": self.config.env.normalize,
            "reward": reward,
            "model_state_dict": self.model.state_dict(),
        }, model_path)

        print(f"Model saved: {model_path}")

    def load_checkpoint(self, checkpoint_path: str | Path) -> dict:
        checkpoint_path = Path(checkpoint_path)

        print(f"Loading checkpoint from {checkpoint_path}...")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True
        )

        # Restore Q-network
        self.model.load_state_dict(
            checkpoint["model"]
        )

        # Restore optimizer
        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        # Restore scheduler
        self.scheduler.load_state_dict(
            checkpoint["scheduler"]
        )

        # Restore exploration state
        self.epsilon = float(checkpoint["epsilon"])

        # Keep model in training mode
        self.model.train()

        print(
            f"Checkpoint loaded: episode={checkpoint['episode']}, "
            f"global_step={checkpoint['global_step']}, "
            f"epsilon={self.epsilon:.4f}"
        )

        return {
            "episode": checkpoint["episode"],
            "global_step": checkpoint["global_step"],
            "best_reward": checkpoint["best_reward"],
        }

    def load_model(self, model_path: str):
        model = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(model["model_state_dict"])
        print(f"Model loaded: {model_path}")