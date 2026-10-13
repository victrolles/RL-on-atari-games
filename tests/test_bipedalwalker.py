import gymnasium as gym

env = gym.make("BipedalWalker-v3")

print("Observation:", env.observation_space)
print("Action:", env.action_space)