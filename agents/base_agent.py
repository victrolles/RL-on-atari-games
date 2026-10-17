from abc import ABC, abstractmethod

from configs import RunConfig
from replay_buffer import ReplayBuffer

import torch

class BaseAgent(ABC):
    def __init__(self,
                 config: RunConfig,
                 replay_buffer: ReplayBuffer,
                 state_size: int | tuple[int, int, int],
                 action_size: int):
        
        self.config = config
        self.replay_buffer = replay_buffer
        self.state_size = state_size
        self.action_size = action_size

        # Device
        if config.env.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")

    @abstractmethod
    def select_action(self, state, deterministic: bool = False):
        pass

    @abstractmethod
    def update(self) -> dict:
        pass

    @abstractmethod
    def save_checkpoint(self, episode: int, global_step: int, best_reward: float, filename: str):
        pass

    @abstractmethod
    def load_checkpoint(self, filename: str):
        pass

    @abstractmethod
    def save_model(self, episode: int, reward: float):
        pass

    @abstractmethod
    def load_model(self, filename: str):
        pass