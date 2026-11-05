# CarRacing — Soft Actor-Critic with CNNs

[← README](../../../README.md) · [Results and protocol](../README.md) · [Setup](../../setup.md)

Learn to drive from track images by controlling steering, throttle, and braking. `CarRacing-v3` provides `96 × 96` RGB images and uses its continuous action space here. Tracks are generated at each reset. [Gymnasium description](https://gymnasium.farama.org/environments/box2d/car_racing/).

## Before / after

<table>
  <tr><th align="center">Initial recording</th><th align="center">After training</th></tr>
  <tr>
    <td align="center"><img src="record_1.gif" alt="CarRacing: initial recording" width="400"></td>
    <td align="center"><img src="record_2.gif" alt="CarRacing: trained agent" width="400"></td>
  </tr>
</table>

## Results and runtime

| Metric | Value |
| :--- | :--- |
| Expected runtime | About a day; allow up to 30 h on the reference machine |
| Estimate based on experience | One day |
| Runtime from the local W&B summary | 29 h 18 min 44 s |
| Latest mean evaluation reward from the local summary | 311.47 |
| Total interactions from the local summary | 999,443 |
| Local run | `run_8544` · W&B `w5q1kcxw` |
| Reference hardware | NVIDIA GeForce GTX 1650 · 4 GB |

This is the longest of the four experiments: the agent learns both a visual representation and a control policy. The latest local mean evaluation reward remains below some peaks in the charts, so it does not represent the best result achieved. The [archived sources](../results.json) preserve the exact metrics.

## Method and technologies

- **SAC**: a Gaussian actor, two critics, two target critics, and a replay buffer.
- **PyTorch CNNs**: convolutional encoders in the actor and critics to process images.
- **Adam**: optimization of the networks and the entropy temperature `alpha`.
- **MSE** for the critics; an entropy-regularized SAC objective for the actor, with a learned temperature and target entropy of `−3`.
- **Gymnasium / Box2D**, **W&B**, and **RecordVideo**: simulation, charts, and videos.

### Visual preprocessing and actions

The three environments used for training, evaluation, and video apply the same transformations in [`env.py`](../../../env.py):

1. Convert to grayscale.
2. Resize to `84 × 84` pixels.
3. Stack **4 frames**, producing an observation with shape `(4, 84, 84)`.
4. Convert pixels from `[0, 255]` to `[0, 1]` in the CNN encoder.

`RescaleAction` maps policy outputs in `[-1, 1]` to the native steering, throttle, and brake bounds. The current implementation applies neither frame skipping nor image augmentation. No learning rate scheduler is used for SAC.

## Charts and analysis

### Rewards

<table>
  <tr><th align="center">Training reward</th><th align="center">Evaluation reward</th></tr>
  <tr><td align="center"><img src="chart_reward_training.svg" alt="Training reward" width="400"></td><td align="center"><img src="chart_reward_evaluation.svg" alt="Evaluation reward" width="400"></td></tr>
</table>

Training reward moves out of an initially negative range and regularly reaches high scores, sometimes close to 900. Variability remains substantial: many episodes still score poorly, even late in the export. Isolated good episodes therefore do not demonstrate reliable driving across all tracks.

Evaluation initially improves substantially, then generally oscillates between about 200 and 500, with a peak near 600 and several setbacks. The latest local mean of 311.47 is consistent with learning that remains uneven. Track diversity and changes in the policy may contribute to this variability; the exports cannot quantify their respective effects.

### Actor and critic losses

<table>
  <tr><th align="center">Actor loss</th><th align="center">Exported critic loss</th></tr>
  <tr><td align="center"><img src="chart_actor_loss.svg" alt="Actor loss" width="400"></td><td align="center"><img src="chart_critic_loss.svg" alt="Exported critic loss" width="400"></td></tr>
</table>

The actor loss initially increases, then becomes negative and fluctuates around a new level. Its sign and magnitude depend on Q-values and entropy; they should not be interpreted as a driving error. The critic loss remains mostly near the bottom of the displayed scale, with occasional spikes, while rewards remain scattered.

The actor export covers a shorter horizontal range than the other three figures. Without complete identification of their sessions and axis settings, events cannot be precisely aligned across these charts. The `Step` axis is kept in its original form.

## Reproduce the experiment

From the repository root, after [setup](../../setup.md):

```powershell
python main.py --config configs/sac_on_carracing.yaml
```

Allow enough RAM for the image replay buffer and enough disk space for models and videos. Reducing `replay.capacity` or `replay.batch_size` can make a new run more accessible, while changing its learning conditions. Keep the configurations being compared.

## Hyperparameters

Values from the [current configuration](../../../configs/sac_on_carracing.yaml), supplemented by the defaults in [`configs/config.py`](../../../configs/config.py). They describe the suggested run and do not guarantee the settings used for the historical charts.

### Environment and network

| Parameter | Value |
| :--- | :--- |
| `env.name` | `CarRacing-v3` |
| `env.device` | `cuda` |
| `env.normalize` | `true` |
| `network.hidden_size` | `256` |

### Training and replay buffer

| Parameter | Value |
| :--- | :--- |
| `training.nb_episodes` | `1000` |
| `training.learning_starts` | `10000` |
| `training.random_steps` | `10000` |
| `training.train_freq` | `1` |
| `training.gradient_steps` | `1` |
| `training.evaluation_interval` | `25` |
| `replay.capacity` | `50000` |
| `replay.batch_size` | `128` |

`learning_starts`, `random_steps`, and `train_freq` are measured in environment interactions. `gradient_steps` is the number of updates per trigger; `evaluation_interval` is measured in episodes.

### Optimization

| Parameter | Value |
| :--- | :--- |
| `agent.actor_lr` | `0.0003` |
| `agent.critic_lr` | `0.0003` |
| `agent.alpha_lr` | `0.0003` |
| `agent.gamma` | `0.99` |
| `agent.tau` | `0.005` |
| `agent.alpha` | `0.2` |

`alpha` is the initial entropy coefficient, which is then learned automatically. The configured `log_std` bounds are `-20` and `2`, but the constructor currently uses the effective bounds **−5 and 2**: these two YAML fields are not passed to it.

Tracking uses the W&B project `atari-rl` and creates outputs under `runs/` by default. Evaluations consist of **20 deterministic episodes**, followed by **3 recorded episodes**, according to the [shared protocol](../README.md#current-training-protocol).

## Potential improvements

- Evaluate on more tracks and multiple seeds, recording score variability.
- Log speed, steering, throttle, braking, and episode length to better explain failures.
- Experiment with frame skipping and reward accumulation while respecting episode boundaries.
- Compare more efficient replay storage and visual augmentations such as random shifts or crops.

These ideas build on the experiment notes; they are not implemented in the repository.
