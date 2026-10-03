from tqdm import tqdm

NB_EPISODES = 1000

class Trainer:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def train(self):
        for episode in tqdm(range(NB_EPISODES)):
            # Reset the environment
            obs, info = self.env.reset()
            done = False

            while not done:
                # Select an action using the agent
                action = self.agent.select_action(obs)

                # Take a step in the environment
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Update the agent with the experience
                self.agent.update(obs, action, reward, next_obs, done)

                # Move to the next observation
                obs = next_obs

            # Decay the exploration rate
            self.agent.epsilon = max(0.01, self.agent.epsilon - 0.01)