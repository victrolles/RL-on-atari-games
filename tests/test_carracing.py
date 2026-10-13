import gymnasium as gym

env = gym.make("CarRacing-v3")

print("Observation:", env.observation_space)
print("Action:", env.action_space)

state, _ = env.reset()
print("Initial state shape:", state.shape)