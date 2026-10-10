import torch
import torch.nn as nn

from torch.distributions import Normal

# A stochastic policy (opposite of a deterministic policy) is a policy that selects actions according to a probability distribution, rather than always choosing the same action for a given state.

class PolicyNetwork(nn.Module): # Also known as the actor
    def __init__(self,
                 num_inputs: int,
                 hidden_size: int,
                 num_actions: int,
                 init_w: float = 3e-3,
                 log_std_min: float = -5,
                 log_std_max: float = 2):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Backbone (or trunk): The main part of the network that learns shared features.
        self.backbone = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Head: The final part of the network responsible for producing a specific output.
        # We use uniform initialization for the weights and biases of the mean and log_std layers to ensure that the initial outputs are not too extreme.
        self.mean_head = nn.Linear(hidden_size, num_actions)
        self.mean_head.weight.data.uniform_(-init_w, init_w)
        self.mean_head.bias.data.uniform_(-init_w, init_w)

        self.log_std_head = nn.Linear(hidden_size, num_actions)
        self.log_std_head.weight.data.uniform_(-init_w, init_w)
        self.log_std_head.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        # Extract features from the state using the backbone network
        features = self.backbone(state)

        # Compute the mean and log standard deviation
        # The standard deviation (écart type, \(\sigma\)) is a measure of the amount of variation or dispersion in a set of values. In the context of a Gaussian distribution, it determines the "spread" of the distribution.
        # The log standard deviation is used instead of the standard deviation directly for numerical stability and to ensure that the standard deviation remains positive when exponentiated.
        # The mean (\(\mu\)) represents the expected value of the distribution.
        # Small \(\sigma\): actions are concentrated near the mean.
        # Large \(\sigma\): actions are more spread out, allowing more exploration.
        # The mean (\(\mu\)) represents the expected value of the distribution.

        mean = self.mean_head(features)

        log_std = self.log_std_head(features)
        log_std = torch.clamp(
            log_std,
            min=self.log_std_min,
            max=self.log_std_max
        )

        return mean, log_std

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        mean, log_std = self(state)

        std = log_std.exp()

        distribution = Normal(
            mean,
            std
        )

        # Reparameterization trick
        x = distribution.rsample()

        # BipedalWalker actions must be between -1 and 1
        action = torch.tanh(x)

        # Log probability before tanh correction
        log_prob = distribution.log_prob(x)

        # Correct probability because of tanh transformation
        log_prob -= torch.log(
            1 - action.pow(2) + 1e-6
        )

        # One log probability per action vector
        log_prob = log_prob.sum(
            dim=-1,
            keepdim=True
        )

        return action, log_prob

    def deterministic_action(
        self,
        state: torch.Tensor
    ) -> torch.Tensor:
        """
        Used during evaluation.

        Instead of sampling from the Gaussian,
        directly use its mean.
        """

        mean, _ = self(state)

        return torch.tanh(mean)