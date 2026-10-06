import sys
from pathlib import Path

from torchview import draw_graph

# This script is run directly from docs/trainings/dql_on_cartpole, so make the project
# root available for the root-level model modules.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from agent import Agent
from config import Config
from replay_buffer import ReplayBuffer

def main():
    config = Config(load_model=False)
    replay_buffer = ReplayBuffer(config.capacity, config.batch_size)

    # Create an instance of the Agent class
    agent = Agent(config, input_size=4, output_size=2, replay_buffer=replay_buffer)

    # Draw the model architecture
    draw_graph(agent.model,
               input_size=(1, 4),
               device="meta",
               graph_name="DQN_CartPole_Model",
               save_graph=True)

if __name__ == "__main__":
    main()