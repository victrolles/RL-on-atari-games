import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from agents.sac.policy_network import PolicyNetwork
from agents.sac.cnn_networks import CNNPolicyNetwork, CNNQNetwork
from agents.sac.q_network import QNetwork
from configs import SACRunConfig
from replay_buffer import ReplayBuffer
from agents import Agent


class SACAgent(Agent):
    def __init__(
        self,
        config: SACRunConfig,
        replay_buffer: ReplayBuffer,
        state_size: int | tuple[int, int, int],
        action_size: int
    ):
        super().__init__(config, replay_buffer)

        # Detect image observations
        is_image = isinstance(state_size, tuple)

        if is_image:
            actor_class = CNNPolicyNetwork
            critic_class = CNNQNetwork
            input_kwargs = {"num_channels": state_size[0]}
        else:
            actor_class = PolicyNetwork
            critic_class = QNetwork
            input_kwargs = {"num_inputs": state_size}

        # Actor
        self.actor = actor_class(
            **input_kwargs,
            hidden_size=config.network.hidden_size,
            num_actions=action_size
        ).to(self.device)

        # Critics
        self.critic_1 = critic_class(
            **input_kwargs,
            hidden_size=config.network.hidden_size,
            num_actions=action_size
        ).to(self.device)

        self.critic_2 = critic_class(
            **input_kwargs,
            hidden_size=config.network.hidden_size,
            num_actions=action_size
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

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.agent.actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=config.agent.critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=config.agent.critic_lr)

        # Update alpha
        self.log_alpha = torch.tensor(
            np.log(config.agent.alpha),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True
        )

        self.target_entropy = -float(action_size)

        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=config.agent.alpha_lr
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

    def update(self) -> dict:

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

        # Get the current alpha value from log_alpha

        alpha = self.log_alpha.exp().detach()

        # ==================================
        # Critic target
        # ==================================

        with torch.no_grad():

            next_actions, next_log_probs = self.actor.sample(next_states)

            target_q1 = self.target_critic_1(next_states, next_actions)
            target_q2 = self.target_critic_2(next_states, next_actions)
            minimum_target_q = torch.min(target_q1, target_q2)

            # SAC entropy term
            soft_target_q = minimum_target_q - alpha * next_log_probs
            target_q = rewards + self.config.agent.gamma * (1 - dones) * soft_target_q

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

        # We don't want to backpropagate through the critics when updating the actor, so we freeze their parameters.
        # Freeze critic parameters
        for critic in (self.critic_1, self.critic_2):
            critic.requires_grad_(False)

        new_actions, log_probs = self.actor.sample(states)

        # Get the Q-values for the new actions from both critics
        q1_new_actions = self.critic_1(states, new_actions)
        q2_new_actions = self.critic_2(states, new_actions)

        # Get the minimum of the two Q-values for the new actions
        minimum_q = torch.min(q1_new_actions,q2_new_actions)

        actor_loss = (alpha * log_probs - minimum_q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critics
        for critic in (self.critic_1, self.critic_2):
            critic.requires_grad_(True)

        # ==================================
        # Update target critics
        # ==================================

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

        # ==================================
        # Update alpha
        # ==================================

        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ==================================
        # Log metrics
        # ==================================

        with torch.no_grad():
            _, log_std = self.actor(states)
            std = log_std.exp()

        return {
            "actor_loss": actor_loss.item(),
            "critic_1_loss": critic_1_loss.item(),
            "critic_2_loss": critic_2_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.log_alpha.exp().item(),
            # Q-value estimates
            "mean_q": minimum_q.mean().item(),
            # Policy exploration
            "std": std.mean().item(),
            "log_std": log_std.mean().item(),
            "log_prob": log_probs.mean().item()
        }

    # Checkpointing is used to save the entire training state :
    # - Training progress (episode, global step, best reward)
    # - Model weights
    # - Optimizer states
    # - Training progress
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

            # Actor
            "actor": self.actor.state_dict(),

            # Critics
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),

            # Target critics
            "target_critic_1": self.target_critic_1.state_dict(),
            "target_critic_2": self.target_critic_2.state_dict(),

            # Automatic entropy tuning
            "log_alpha": self.log_alpha.detach().clone(),

            # Optimizers
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }

        path = checkpoint_dir / filename

        # Write to a temporary file first
        temporary_path = path.with_suffix(".tmp")
        torch.save(checkpoint, temporary_path)

        # Replace the previous checkpoint atomically
        temporary_path.replace(path)

        print(f"Checkpoint saved: {path}")

    # Save the model weights only (without optimizer states or training progress)
    def save_model(self, reward: float):
        
        # Create the model directory if it doesn't exist
        model_dir = Path(self.config.logging.run_dir) / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save the model weights
        model_path = model_dir / f"model_{reward:.2f}.pt"
        torch.save(self.actor.state_dict(), model_path)

        print(f"Model saved: {model_path}")


    def load_checkpoint(self, checkpoint_path: str) -> dict:

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True
        )

        # Actor
        self.actor.load_state_dict(checkpoint["actor"])

        # Critics
        self.critic_1.load_state_dict(checkpoint["critic_1"])
        self.critic_2.load_state_dict(checkpoint["critic_2"])

        # Target critics
        self.target_critic_1.load_state_dict(
            checkpoint["target_critic_1"]
        )
        self.target_critic_2.load_state_dict(
            checkpoint["target_critic_2"]
        )

        # Restore alpha
        with torch.no_grad():
            self.log_alpha.copy_(checkpoint["log_alpha"])

        # Optimizers
        self.actor_optimizer.load_state_dict(
            checkpoint["actor_optimizer"]
        )
        self.critic_1_optimizer.load_state_dict(
            checkpoint["critic_1_optimizer"]
        )
        self.critic_2_optimizer.load_state_dict(
            checkpoint["critic_2_optimizer"]
        )
        self.alpha_optimizer.load_state_dict(
            checkpoint["alpha_optimizer"]
        )

        print(f"Checkpoint loaded: {checkpoint_path}")

        return {
            "episode": checkpoint["episode"],
            "global_step": checkpoint["global_step"],
            "best_reward": checkpoint["best_reward"],
        }

    def soft_update(
        self,
        network,
        target_network
    ):
        tau = self.config.agent.tau

        with torch.no_grad():
            for parameter, target_parameter in zip(
                network.parameters(),
                target_network.parameters()
            ):

                target_parameter.data.mul_(1 - tau)
                target_parameter.data.add_(tau * parameter.data)