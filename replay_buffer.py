from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity, batch_size):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            raise ValueError("Batch size must be less than or equal to the number of elements in the buffer.")
        batch = random.sample(self.buffer, batch_size)
        return list(zip(*batch))

    def __len__(self):
        return len(self.buffer)