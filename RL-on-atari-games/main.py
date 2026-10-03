from agent import Agent
from trainer import Trainer
from env import Environment

def main():
    # Initialize the environment
    env = Environment("CartPole-v1")

    # Initialize the agent
    input_size = env.env.observation_space.shape[0]
    hidden_size = 128
    output_size = env.env.action_space.n
    agent = Agent(input_size, hidden_size, output_size)
    
    # Initialize the trainer
    trainer = Trainer(agent, env)
    

    # Start training
    trainer.train()

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()