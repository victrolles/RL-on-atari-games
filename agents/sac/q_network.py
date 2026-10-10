import torch
import torch.nn as nn


class QNetwork(nn.Module): # Also known as the critic
    def __init__(self, num_inputs: int, num_actions: int, hidden_size: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:

        state_action = torch.cat([state, action], dim=-1)

        return self.network(state_action)