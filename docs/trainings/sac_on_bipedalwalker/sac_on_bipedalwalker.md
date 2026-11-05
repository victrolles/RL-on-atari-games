# BipedalWalker — Soft Actor-Critic

[← README](../../../README.md) · [Results and protocol](../README.md) · [Setup](../../setup.md)

Move a bipedal robot across uneven terrain by controlling its four joints. The environment provides 24 observations and four continuous actions in `[-1, 1]`. The project uses the standard `BipedalWalker-v3` variant, with hardcore mode disabled. [Gymnasium description](https://gymnasium.farama.org/environments/box2d/bipedal_walker/).

## Before / after

<table>
  <tr><th align="center">Initial recording</th><th align="center">After training</th></tr>
  <tr>
    <td align="center"><img src="record_1.gif" alt="BipedalWalker: initial recording" width="400"></td>
    <td align="center"><img src="record_2.gif" alt="BipedalWalker: trained agent" width="400"></td>
  </tr>
</table>

## Results and runtime

| Metric | Value |
| :--- | :--- |
| Expected runtime | About 10 h on the reference machine |
| First promising results, according to historical notes | About 1 h |
| Runtime noted in historical documentation | 9 h 30 min |
| Runtime from the local W&B summary | 9 h 33 min 52 s |
| Latest mean evaluation reward from the local summary | 334.42 |
| Total interactions from the local summary | 1,197,187 |
| Local run | `run_3025` · W&B `yqzyzht5` |
| Reference hardware | NVIDIA GeForce GTX 1650 · 4 GB |

The latest local evaluation achieves a high reward, but the exports also reveal substantial setbacks during learning. Ten hours is a planning estimate for the completed run, not a guaranteed runtime for all 5,000 configured episodes. The [archived sources](../results.json) provide the exact values.

## Method and technologies

- **SAC**: a Gaussian actor, two critics and their target networks, with a replay buffer.
- **PyTorch**: fully connected networks with two hidden layers of 512 units and ReLU activations.
- **Adam**: optimization of the actor, each critic, and the entropy temperature `alpha`.
- **MSE** for the critics; an entropy-regularized SAC objective for the actor, with a learned temperature and target entropy of `−4`.
- **Gymnasium / Box2D**, **W&B**, and **RecordVideo**: physics simulation, tracking, and recording.

Policy actions are bounded by `tanh` within the range expected by the environment; no additional rescaling is enabled. Updates are grouped: after the initial phase, the code performs 64 updates every 64 interactions. No learning rate scheduler is used for SAC.

## Charts and analysis

### Rewards

<table>
  <tr><th align="center">Training reward</th><th align="center">Evaluation reward</th></tr>
  <tr><td align="center"><img src="chart%20reward_training.svg" alt="Training reward" width="400"></td><td align="center"><img src="chart_reward_evaluation.svg" alt="Evaluation reward" width="400"></td></tr>
</table>

Training rewards move from a negative range to frequent plateaus around 330. Very poor episodes nevertheless persist throughout the export. The agent therefore learns effective behavior in many episodes, with occasional failures that remain substantial.

Evaluation supports this interpretation: several plateaus above 300 alternate with drops close to −100. The quality of the best stretches is insufficient to establish continuous stability. The charts alone cannot separate the effects of the policy, the terrain encountered, and successive updates.

### Optimization objectives

<table>
  <tr><th align="center">Actor loss</th><th align="center">Critic loss · export 1</th></tr>
  <tr><td align="center"><img src="chart_actor_loss.svg" alt="Actor loss" width="400"></td><td align="center"><img src="chart_critic_loss.svg" alt="Critic loss · export 1" width="400"></td></tr>
  <tr><th align="center">Critic loss · export 2</th></tr>
  <tr><td align="center"><img src="chart_critic_loss_2.svg" alt="Critic loss · export 2" width="400"></td></tr>
</table>

The actor loss generally stabilizes at a negative value after the initial phase. Both critic loss exports usually remain near the bottom of the displayed scale, with a clear spike around the horizontal coordinate `1.1M`. The actor also shows a disturbance in this region, followed by a return to its previous level.

These events are consistent with a temporary adjustment of value estimates; their cause cannot be identified from these charts alone. The names of the two critic exports do not specify which corresponds to `critic_1_loss` or `critic_2_loss`, so the figures remain labeled by export number.

### Entropy temperature

<table>
  <tr><th align="center">Entropy coefficient alpha</th><th align="center">Entropy temperature loss</th></tr>
  <tr><td align="center"><img src="chart_alpha.svg" alt="Entropy coefficient alpha" width="400"></td><td align="center"><img src="chart_alpha_loss.svg" alt="Entropy temperature loss" width="400"></td></tr>
</table>

The coefficient `alpha` quickly decreases from its initial value of 0.2, then stays near 0.01 with a few spikes. The local summary records `alpha ≈ 0.0106`. The entropy term therefore carries less weight while continuing to adapt. The associated loss oscillates within a relatively stable range after learning begins; this does not guarantee that the robot will never fail.

### Memory and progress

<table>
  <tr><th align="center">Replay buffer size</th><th align="center">Total interactions: Steps total</th></tr>
  <tr><td align="center"><img src="chart_buffer_size.svg" alt="Replay buffer size" width="400"></td><td align="center"><img src="chart_step_total.svg" alt="Total interactions: Steps total" width="400"></td></tr>
</table>

The buffer reaches its capacity of 300,000 transitions and then remains constant: new experiences replace the oldest ones. This plateau describes storage, not a halt in learning. `Steps total` continues to increase throughout the exported segment.

The horizontal `Step` axis is kept as exported by W&B. The progress chart and local summary cover different ranges; their final values are not treated as equivalent.

## Reproduce the experiment

From the repository root, after [setup](../../setup.md):

```powershell
python main.py --config configs/sac_on_bipedalwalker.yaml
```

Plan for a long run and keep `checkpoints/best.pt` to retain the best saved evaluation, even if later ones regress. The [resume command](../../../README.md#resume-training) lets you extend a run.

## Hyperparameters

Values from the [current configuration](../../../configs/sac_on_bipedalwalker.yaml), supplemented by the defaults in [`configs/config.py`](../../../configs/config.py). They describe the suggested run and do not guarantee the settings used for the historical charts.

### Environment and network

| Parameter | Value |
| :--- | :--- |
| `env.name` | `BipedalWalker-v3` |
| `env.device` | `cuda` |
| `env.normalize` | `false` |
| `network.hidden_size` | `512` |

### Training and replay buffer

| Parameter | Value |
| :--- | :--- |
| `training.nb_episodes` | `5000` |
| `training.learning_starts` | `10000` |
| `training.random_steps` | `10000` |
| `training.train_freq` | `64` |
| `training.gradient_steps` | `64` |
| `training.evaluation_interval` | `30` |
| `replay.capacity` | `300000` |
| `replay.batch_size` | `256` |

`learning_starts`, `random_steps`, and `train_freq` are measured in environment interactions. `gradient_steps` is the number of updates per trigger; `evaluation_interval` is measured in episodes.

### Optimization

| Parameter | Value |
| :--- | :--- |
| `agent.actor_lr` | `0.00073` |
| `agent.critic_lr` | `0.00073` |
| `agent.alpha_lr` | `0.00073` |
| `agent.gamma` | `0.98` |
| `agent.tau` | `0.02` |
| `agent.alpha` | `0.2` |

`alpha` is the initial entropy coefficient, which is then learned automatically. The configured `log_std` bounds are `-20` and `2`, but the constructor currently uses the effective bounds **−5 and 2**: these two YAML fields are not passed to it.

Tracking uses the W&B project `atari-rl` and creates outputs under `runs/` by default. Evaluations consist of **20 deterministic episodes**, followed by **3 recorded episodes**, according to the [shared protocol](../README.md#current-training-protocol).
