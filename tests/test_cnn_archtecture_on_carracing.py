import torch

from configs import load_config
from env import Environment
from agents.sac.cnn_networks import CNNPolicyNetwork, CNNQNetwork

config = load_config("configs/sac_on_carracing.yaml")
config.logging.run_dir = "./runs/test_carracing"

env = Environment(config)

try:
    state, _ = env.train_env.reset()

    print("Observation:", state.shape)
    print("Data type:", state.dtype)
    print("Action space:", env.train_env.action_space)

    actor = CNNPolicyNetwork(
        num_channels=4,
        hidden_size=256,
        num_actions=3
    )

    critic = CNNQNetwork(
        num_channels=4,
        num_actions=3,
        hidden_size=256
    )

    states = torch.as_tensor(state).unsqueeze(0)

    with torch.no_grad():
        actions, log_probs = actor.sample(states)
        q_values = critic(states, actions)

    print("Actions:", actions.shape)
    print("Log probabilities:", log_probs.shape)
    print("Q-values:", q_values.shape)

finally:
    env.train_env.close()
    env.eval_env.close()