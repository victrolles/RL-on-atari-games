import gymnasium as gym

class Environment:
    def __init__(self, env_name):
        self.env = gym.make(env_name, render_mode="human")

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        return self.env.close()