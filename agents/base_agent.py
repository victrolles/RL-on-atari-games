from abc import ABC, abstractmethod

from config import Config
from replay_buffer import ReplayBuffer

import torch

class BaseAgent(ABC):
    def __init__(self,
                 config: Config,
                 replay_buffer: ReplayBuffer):
        
        self.config = config
        self.replay_buffer = replay_buffer

        # Device
        if config.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")

    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def save_model(self, episode: int, reward: float):
        pass