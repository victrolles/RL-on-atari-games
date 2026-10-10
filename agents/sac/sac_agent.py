import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from agents.sac.policy_network import PolicyNetwork
from agents.sac.q_network import QNetwork
from config import Config
from replay_buffer import ReplayBuffer
from agents import BaseAgent


class SACAgent(BaseAgent):
    def __init__(self,
                 config: Config,
                 replay_buffer: ReplayBuffer,
                 state_size: int,
                 action_size: int):

        super().__init__(config, replay_buffer)

        # Actor
        self.actor = PolicyNetwork(
            num_inputs=state_size,
            hidden_size=config.hidden_size,
            num_actions=action_size
        ).to(self.device)

        # Critics
        self.critic_1 = QNetwork(
            num_inputs=state_size,
            num_actions=action_size,
            hidden_size=config.hidden_size
        ).to(self.device)

        self.critic_2 = QNetwork(
            num_inputs=state_size,
            num_actions=action_size,
            hidden_size=config.hidden_size
        ).to(self.device)

        # ----------------------------------
        # Target critics
        # ----------------------------------

        self.target_critic_1 = copy.deepcopy(
            self.critic_1
        )

        self.target_critic_2 = copy.deepcopy(
            self.critic_2
        )

        # Target critics are not optimized directly
        for parameter in self.target_critic_1.parameters():
            parameter.requires_grad = False

        for parameter in self.target_critic_2.parameters():
            parameter.requires_grad = False

        # ----------------------------------
        # Optimizers
        # ----------------------------------

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=config.critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=config.critic_lr)

        if self.config.load_model:
            self.load_model(
                actor_path=self.config.actor_model_path,
                critic_1_path=self.config.critic_1_model_path,
                critic_2_path=self.config.critic_2_model_path
            )

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic_action(state_tensor)
            else:
                action, _ = self.actor.sample(state_tensor)

        return (action.cpu().numpy()[0])

    def train(self) -> dict:

        # ==================================
        # Sample replay buffer
        # ==================================

        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        # ==================================
        # Convert NumPy -> tensors
        # ==================================

        states = torch.as_tensor(
            states,
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.as_tensor(
            actions,
            dtype=torch.float32,
            device=self.device
        )

        rewards = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(1)

        next_states = torch.as_tensor(
            next_states,
            dtype=torch.float32,
            device=self.device
        )

        dones = torch.as_tensor(
            dones,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(1)

        # ==================================
        # Critic target
        # ==================================

        with torch.no_grad():

            next_actions, next_log_probs = self.actor.sample(next_states)

            target_q1 = self.target_critic_1(next_states, next_actions)
            target_q2 = self.target_critic_2(next_states, next_actions)
            minimum_target_q = torch.min(target_q1, target_q2)

            # SAC entropy term
            soft_target_q = minimum_target_q - self.config.alpha * next_log_probs
            target_q = rewards + self.config.gamma * (1 - dones) * soft_target_q

        # ==================================
        # Train critic 1
        # ==================================

        current_q1 = self.critic_1(states, actions)
        critic_1_loss = F.mse_loss(current_q1, target_q)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        # ==================================
        # Train critic 2
        # ==================================

        current_q2 = self.critic_2(states, actions)
        critic_2_loss = F.mse_loss(current_q2, target_q)

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # ==================================
        # Train actor
        # ==================================

        new_actions, log_probs = self.actor.sample(states)

        # Get the Q-values for the new actions from both critics
        q1_new_actions = self.critic_1(states, new_actions)
        q2_new_actions = self.critic_2(states, new_actions)

        # Get the minimum of the two Q-values for the new actions
        minimum_q = torch.min(q1_new_actions,q2_new_actions)

        actor_loss = (self.config.alpha * log_probs - minimum_q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ==================================
        # Update target critics
        # ==================================

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

        with torch.no_grad():
            _, log_std = self.actor(states)
            std = log_std.exp()

        return {
            "actor_loss": actor_loss.item(),
            "critic_1_loss": critic_1_loss.item(),
            "critic_2_loss": critic_2_loss.item(),
            # Q-value estimates
            "mean_q": minimum_q.mean().item(),
            # Policy exploration
            "std": std.mean().item(),
            "log_std": log_std.mean().item(),
            "log_prob": log_probs.mean().item()
        }

    def save_model(self, episode: int, reward: float):
        model_dir = Path(self.config.run_dir) / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save actor
        actor_path = model_dir / f"actor_episode_{episode}_reward_{reward:.2f}.pth"
        torch.save(self.actor.state_dict(), actor_path)

        # Save critic 1
        critic_1_path = model_dir / f"critic_1_episode_{episode}_reward_{reward:.2f}.pth"
        torch.save(self.critic_1.state_dict(), critic_1_path)

        # Save critic 2
        critic_2_path = model_dir / f"critic_2_episode_{episode}_reward_{reward:.2f}.pth"
        torch.save(self.critic_2.state_dict(), critic_2_path)

        print(f"Models saved at episode {episode} with mean reward {reward:.2f}")

    def load_model(self, actor_path: str, critic_1_path: str, critic_2_path: str):
        print(f"Loading actor model from {actor_path}...")
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))

        print(f"Loading critic 1 model from {critic_1_path}...")
        self.critic_1.load_state_dict(torch.load(critic_1_path, map_location=self.device))

        print(f"Loading critic 2 model from {critic_2_path}...")
        self.critic_2.load_state_dict(torch.load(critic_2_path, map_location=self.device))

    def soft_update(
        self,
        network,
        target_network
    ):
        tau = self.config.tau

        with torch.no_grad():
            for parameter, target_parameter in zip(
                network.parameters(),
                target_network.parameters()
            ):

                target_parameter.data.mul_(1 - tau)
                target_parameter.data.add_(tau * parameter.data)