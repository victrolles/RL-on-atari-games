from collections import deque

import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int, batch_size: int):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.buffer)