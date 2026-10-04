
import torch
import torch.nn as nn

from config import Config

class DQN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(DQN, self).__init__()

        self.fully_connected_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.fully_connected_layers(x)

class Agent:
    def __init__(self, config: Config, input_size: int, output_size: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(input_size, config.hidden_size, output_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.MSELoss()
        self.epsilon = config.epsilon
        self.config = config

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

    def update(self, state, action, reward, next_state, done):
        # Compute the target Q-value
        with torch.no_grad():
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
            target_q_value = reward + (self.config.gamma * torch.max(self.model(next_state)) * (1 - done))

        # Compute the current Q-value
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action = torch.tensor(action).unsqueeze(0).to(self.device)
        current_q_value = self.model(state)[0, action].squeeze()

        # Compute the loss
        loss = self.criterion(current_q_value, target_q_value)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()