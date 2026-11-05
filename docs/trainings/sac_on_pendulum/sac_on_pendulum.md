# Pendulum — Soft Actor-Critic

[← README](../../../README.md) · [Results and protocol](../README.md) · [Setup](../../setup.md)

Learn to swing up and then stabilize a pendulum by controlling continuous torque. The observation has three components, and the agent produces one action. The reward penalizes deviation from upright, speed, and effort: a less negative value indicates better performance. [Gymnasium description](https://gymnasium.farama.org/environments/classic_control/pendulum/).

## Before / after

<table>
  <tr><th align="center">Initial recording</th><th align="center">After training</th></tr>
  <tr>
    <td align="center"><img src="record_1.gif" alt="Pendulum: initial recording" width="400"></td>
    <td align="center"><img src="record_2.gif" alt="Pendulum: trained agent" width="400"></td>
  </tr>
</table>

## Results and runtime

| Metric | Value |
| :--- | :--- |
| Expected runtime | About 2 min on the reference machine |
| Runtime noted in historical documentation | 1 min 55 s |
| Runtime from the local W&B summary | 1 min 50 s |
| Latest mean evaluation reward from the local summary | −148.81 |
| Total interactions from the local summary | 8,000 |
| Local run | `run_4248` · W&B `lwta13av` |
| Reference hardware | NVIDIA GeForce GTX 1650 · 4 GB |

Pendulum is the shortest SAC experiment in the repository: the YAML file specifies 40 episodes. The local summary gives an encouraging result, but the historical charts show uneven progress. The [archived sources](../results.json) and [interpretation guidelines](../README.md) distinguish these records.

## Method and technologies

- **SAC**: a Gaussian actor, two critics, and two target critics, with soft target updates.
- **PyTorch**: fully connected networks with ReLU activations and reparameterized sampling, followed by a `tanh` transformation.
- **Adam**: optimizers for the actor, both critics, and the entropy coefficient `alpha`.
- **MSE** for the critics; an actor objective combining Q-value and entropy. The actor loss is not an MSE.
- **Gymnasium**, **W&B**, and **RecordVideo**: simulation, metrics, and videos.

Actions produced in `[-1, 1]` are rescaled by `RescaleAction` to Pendulum's torque range. The target entropy is `−1`, based on the action dimension. The current code does not use a learning rate scheduler for SAC.

## Charts and analysis

### Rewards

<table>
  <tr><th align="center">Training reward</th><th align="center">Evaluation reward</th></tr>
  <tr><td align="center"><img src="chart_reward_training.svg" alt="Training reward" width="400"></td><td align="center"><img src="chart_reward_evaluation.svg" alt="Evaluation reward" width="400"></td></tr>
</table>

Training reward moves from values often near −1,500 to several much better episodes, sometimes close to zero. Setbacks remain frequent: the improvement does not form a stable plateau.

The evaluation export shows a very good point around −100, followed by a drop below −1,200 and then a partial recovery. It is therefore insufficient to conclude that convergence is steady. Its final value differs from the local W&B summary at −148.81: these two records are presented separately.

### Actor and critic losses

<table>
  <tr><th align="center">Actor loss</th><th align="center">Exported critic loss</th></tr>
  <tr><td align="center"><img src="chart_loss_actor.svg" alt="Actor loss" width="400"></td><td align="center"><img src="chart_loss_critic.svg" alt="Exported critic loss" width="400"></td></tr>
</table>

The actor loss increases, while the critic loss has occasional spikes above 400. The SAC actor objective depends on the scale of Q-values and the entropy term: an increase does not directly imply worse performance. Critic spikes indicate periods of larger errors relative to value targets; rewards remain the main criterion for judging behavior.

### Value estimates and exploration

<table>
  <tr><th align="center">Mean Q-value</th><th align="center">Action log-probability</th></tr>
  <tr><td align="center"><img src="chart_mean_q.svg" alt="Mean Q-value" width="400"></td><td align="center"><img src="chart_log_prob.svg" alt="Action log-probability" width="400"></td></tr>
  <tr><th align="center">Policy standard deviation</th><th align="center">Log standard deviation</th></tr>
  <tr><td align="center"><img src="chart_std.svg" alt="Policy standard deviation" width="400"></td><td align="center"><img src="chart_log_std.svg" alt="Log standard deviation" width="400"></td></tr>
</table>

The mean Q-value becomes more negative over the export, which is consistent with learning negative returns in this environment. This proves neither an improvement nor a deterioration of the policy without comparison to observed returns.

The policy's standard deviation generally decreases from about 0.9 to 0.5, and its logarithm follows the same trend. On the sampled observations, the Gaussian distribution becomes more concentrated. Log-probability generally increases with fluctuations; these metrics describe policy exploration, not a success rate.

The eight SVGs retain their original `Step` axis. Their exact correspondence to episodes and to the local session has not been established.

## Reproduce the experiment

From the repository root, after [setup](../../setup.md):

```powershell
python main.py --config configs/sac_on_pendulum.yaml
```

To increase the episode budget for a new run:

```powershell
python main.py --config configs/sac_on_pendulum.yaml --episodes 80
```

## Hyperparameters

Values from the [current configuration](../../../configs/sac_on_pendulum.yaml), supplemented by the defaults in [`configs/config.py`](../../../configs/config.py). They describe the suggested run and do not guarantee the settings used for the historical charts.

### Environment and network

| Parameter | Value |
| :--- | :--- |
| `env.name` | `Pendulum-v1` |
| `env.device` | `cuda` |
| `env.normalize` | `true` |
| `network.hidden_size` | `64` |

### Training and replay buffer

| Parameter | Value |
| :--- | :--- |
| `training.nb_episodes` | `40` |
| `training.learning_starts` | `1000` |
| `training.random_steps` | `1000` |
| `training.train_freq` | `5` |
| `training.gradient_steps` | `5` |
| `training.evaluation_interval` | `10` |
| `replay.capacity` | `10000` |
| `replay.batch_size` | `64` |

`learning_starts`, `random_steps`, and `train_freq` are measured in environment interactions. `gradient_steps` is the number of updates per trigger; `evaluation_interval` is measured in episodes.

### Optimization

| Parameter | Value |
| :--- | :--- |
| `agent.actor_lr` | `0.003` |
| `agent.critic_lr` | `0.003` |
| `agent.alpha_lr` | `0.003` |
| `agent.gamma` | `0.99` |
| `agent.tau` | `0.005` |
| `agent.alpha` | `0.2` |

`alpha` is the initial entropy coefficient, which is then learned automatically. The configured `log_std` bounds are `-20` and `2`, but the constructor currently uses the effective bounds **−5 and 2**: these two YAML fields are not passed to it.

Tracking uses the W&B project `atari-rl` and creates outputs under `runs/` by default. Evaluations consist of **20 deterministic episodes**, followed by **3 recorded episodes**, according to the [shared protocol](../README.md#current-training-protocol).
