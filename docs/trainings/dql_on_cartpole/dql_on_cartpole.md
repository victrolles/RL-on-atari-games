# CartPole — DQN

[← README](../../../README.md) · [Results and protocol](../README.md) · [Setup](../../setup.md)

Keep a pole upright by moving its cart left or right. This task with discrete actions serves as an introductory DQN experiment. The environment provides four observations and two possible actions; a `CartPole-v1` episode is limited to 500 steps in its standard configuration. [Gymnasium description](https://gymnasium.farama.org/environments/classic_control/cart_pole/).

## Before / after

<table>
  <tr><th align="center">Initial recording</th><th align="center">After training</th></tr>
  <tr>
    <td align="center"><img src="record_1.gif" alt="CartPole: initial recording" width="400"></td>
    <td align="center"><img src="record_2.gif" alt="CartPole: trained agent" width="400"></td>
  </tr>
</table>

## Results and runtime

| Metric | Value |
| :--- | :--- |
| Expected runtime | A few minutes, roughly 2–5 min on the reference machine |
| Runtime noted in historical documentation | 2 min 39 s |
| Runtime from the local W&B summary | 4 min 27 s |
| Latest mean evaluation reward from the local summary | 500.00 |
| Total interactions from the local summary | 305,540 |
| Local run | `run_7834` · W&B `gfulc9v8` |
| Reference hardware | NVIDIA GeForce GTX 1650 · 4 GB |

The latest recorded mean reaches the environment's episode duration limit. It describes that specific evaluation and does not measure stability across training runs. Historical and local runtimes come from different records; the [archived sources](../results.json) and [interpretation guidelines](../README.md) explain this distinction.

## Method and technologies

- **PyTorch**: a fully connected network with two hidden layers and ReLU activations to estimate Q-values.
- **DQN**: learning from a replay buffer and **epsilon-greedy** exploration after the initial random-action phase.
- **Adam** and **MSE**: optimizing the difference between the selected Q-value and its temporal-difference target.
- **CosineAnnealingWarmRestarts**: cyclic learning rate changes; see the [technical note](../../technologies/lr_scheduler.md).
- **Gymnasium**, **W&B**, and **RecordVideo**: simulation, metric tracking, and evaluation recording.

In this implementation, DQN targets are computed using the same network as the current Q-values; there is no separate target network. The [historical network diagram](architecture_model.png) remains available as an illustration.

## Charts and analysis

### Rewards

<table>
  <tr><th align="center">Training reward</th><th align="center">Evaluation reward</th></tr>
  <tr><td align="center"><img src="chart_training_reward.svg" alt="Training reward" width="400"></td><td align="center"><img src="chart_eval_reward.svg" alt="Evaluation reward" width="400"></td></tr>
</table>

Training reward gradually increases from around 20 to about 260 in the export. Evaluation also improves, with several intermediate setbacks, before reaching 500 at the final point. The gap between the two curves is consistent with a policy that still explores during training and uses deterministic action selection during evaluation. It does not, by itself, measure the agent's variance.

### Exploration and value estimation error

<table>
  <tr><th align="center">Epsilon-greedy exploration</th><th align="center">Q-network MSE loss</th></tr>
  <tr><td align="center"><img src="chart_epsilon.svg" alt="Epsilon-greedy exploration" width="400"></td><td align="center"><img src="chart_loss.svg" alt="Q-network MSE loss" width="400"></td></tr>
</table>

Epsilon steadily decreases from about 1 to 0.3: the minimum of 0.01 is not reached in the exported segment. The agent therefore continues trying random actions for a substantial part of training.

The loss shows a period of severe instability, with spikes on the order of `5 × 10⁸`, before returning to a much smaller scale. The improvement in rewards indicates useful learning in this run, but these spikes warrant monitoring value estimates and comparing multiple runs. The export alone cannot attribute this instability to a specific hyperparameter.

These figures are historical; the `Step` axis should not automatically be interpreted as an episode count. The current DQN agent no longer logs the epsilon curve.

## Reproduce the experiment

From the repository root, after [setup](../../setup.md):

```powershell
python main.py --config configs/dqn_on_cartpole.yaml
```

## Hyperparameters

Values from the [current configuration](../../../configs/dqn_on_cartpole.yaml), supplemented by the defaults in [`configs/config.py`](../../../configs/config.py). They describe the suggested run and do not guarantee the settings used for the historical charts.

### Environment and network

| Parameter | Value |
| :--- | :--- |
| `env.name` | `CartPole-v1` |
| `env.device` | `cuda` |
| `env.normalize` | `false` |
| `network.hidden_size` | `64` |

### Training and replay buffer

| Parameter | Value |
| :--- | :--- |
| `training.nb_episodes` | `10000` |
| `training.learning_starts` | `10000` |
| `training.random_steps` | `10000` |
| `training.train_freq` | `40` |
| `training.gradient_steps` | `2` |
| `training.evaluation_interval` | `100` |
| `replay.capacity` | `1000` |
| `replay.batch_size` | `64` |

`learning_starts`, `random_steps`, and `train_freq` are measured in environment interactions. `gradient_steps` is the number of updates per trigger; `evaluation_interval` is measured in episodes.

### Optimization

| Parameter | Value |
| :--- | :--- |
| `agent.learning_rate` | `0.01` |
| `agent.gamma` | `0.99` |
| `agent.epsilon` | `1.0` |
| `agent.epsilon_decay` | `0.999` |
| `agent.epsilon_min` | `0.01` |

Epsilon is multiplied by `epsilon_decay` at each agent update, down to `epsilon_min`. The scheduler uses `T_0=50`, `T_mult=1`, and `eta_min=0`; it advances at each update. These scheduler settings remain defined in the code.

Tracking uses the W&B project `atari-rl` and creates outputs under `runs/` by default. Evaluations consist of **20 deterministic episodes**, followed by **3 recorded episodes**, according to the [shared protocol](../README.md#current-training-protocol).
