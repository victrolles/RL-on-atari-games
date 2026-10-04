from agent import Agent
from trainer import Trainer
from env import Environment
from config import Config

def main():
    config = Config()

    # Initialize the environment
    env = Environment(config.env_name)

    # Initialize the agent
    input_size = env.train_env.observation_space.shape[0]
    output_size = env.train_env.action_space.n
    agent = Agent(config, input_size, output_size)
    
    # Initialize the trainer
    trainer = Trainer(agent, env, config)
    
    # Start training
    trainer.train()

    # Close the environment
    env.train_env.close()
    env.eval_env.close()

if __name__ == "__main__":
    main()