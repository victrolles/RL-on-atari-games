# Reference results and protocol

[← README](../../README.md) · [Setup](../setup.md)

## The four experiments

| Experiment page | Configuration | Recorded evaluation reward | Recorded W&B runtime |
| :--- | :--- | ---: | :--- |
| [CartPole · DQN](dql_on_cartpole/dql_on_cartpole.md) | [YAML](../../configs/dqn_on_cartpole.yaml) | 500.00 | 4 min 27 s |
| [Pendulum · SAC](sac_on_pendulum/sac_on_pendulum.md) | [YAML](../../configs/sac_on_pendulum.yaml) | −148.81 | 1 min 50 s |
| [BipedalWalker · SAC](sac_on_bipedalwalker/sac_on_bipedalwalker.md) | [YAML](../../configs/sac_on_bipedalwalker.yaml) | 334.42 | 9 h 33 min 52 s |
| [CarRacing · SAC + CNN](sac_on_carracing/sac_on_carracing.md) | [YAML](../../configs/sac_on_carracing.yaml) | 311.47 | 29 h 18 min 44 s |

These figures are the **latest values in the local W&B summaries**, and are not necessarily the best results achieved. They describe one run per environment, without averaging across independent training runs. Rewards for the four tasks are not directly comparable.

## Data sources

An extract of the local data is kept in [`results.json`](results.json) so the figures remain available after cloning: `runs/` is ignored by Git. This extract contains only the relevant metrics, run identifier, and hardware; it contains no credentials.

| Environment | Local experiment directory | W&B run ID |
| :--- | :--- | :--- |
| CartPole | `runs/CartPole-v1/run_7834` | `gfulc9v8` |
| Pendulum | `runs/Pendulum-v1/run_4248` | `lwta13av` |
| BipedalWalker | `runs/BipedalWalker-v3/run_3025` | `yqzyzht5` |
| CarRacing | `runs/CarRacing-v3/run_8544` | `w5q1kcxw` |

The original sources are the `wandb-summary.json` and `wandb-metadata.json` files in `runs/<environment>/<run>/wandb/<session>/files/`. The runtimes in the table correspond to `_runtime`, converted to hours, minutes, and seconds. They include elapsed time during W&B tracking, including evaluations and recordings; they do not measure gradient computation time alone.

All four sessions report **Windows 11**, **CPython 3.13.16**, and a **4 GB NVIDIA GeForce GTX 1650**. Historical notes also mentioned **2 min 39 s** for CartPole, **1 min 55 s** for Pendulum, and **9 h 30 min** for BipedalWalker. These values remain listed on the experiment pages because they do not always match the local session found.

## Interpret the videos and charts

Each experiment page presents the `record_1.gif` and `record_2.gif` GIFs as a comparison between an initial recording and a later one. Their exact episodes and their correspondence to the sessions above are not documented. A video illustrates behavior during one episode; it does not replace evaluation over multiple episodes.

The SVGs are W&B exports kept in each directory. The analysis focuses on their visible trends. Some figures are historical and differ from the local summaries, so their values are not combined with those in the table.

The **`Step` axis depends on the export**. In the current code, `wandb.log()` is called at every agent update, every evaluation, and at the end of each episode. W&B's internal step therefore does not automatically correspond to either the episode count or the number of environment interactions. The `Steps total` metric counts those interactions. Without an explicit axis setting in the export, horizontal coordinates should not be converted to episodes or seconds.

## Current training protocol

- **Training**: random actions for `random_steps`, then the agent's policy; learning starts at `learning_starts` if the buffer contains a full batch.
- **Updates**: every `train_freq` interactions, perform `gradient_steps` updates on batches sampled from the replay buffer.
- **Evaluation**: every `evaluation_interval` episodes, compute the mean over **20 deterministic episodes** in a separate environment.
- **Video**: then record **3 deterministic episodes** in a third environment.
- **Checkpoints**: write `latest.pt`, update `best.pt` if the score improves, and export weights to `models/` at every evaluation.

Training rewards include exploration; evaluation rewards use a deterministic policy. For SAC, this applies `tanh` to the policy mean; for DQN, it selects the action with the highest Q-value without epsilon-greedy exploration.

## Hyperparameters and reproducibility

The tables on the experiment pages describe the **YAML files currently under version control** and the defaults actually used. They do not claim to reconstruct the settings of every historical export.

A few limitations to keep in mind when reproducing or comparing a run:

- Random seeds are not set by the training loop. Repeat runs and record their variability before concluding that a change improves performance.
- Dependencies in `requirements.txt` are not pinned. Keep the installed versions with the results of each new experiment.
- The replay buffer and random number generator states are not saved in checkpoints. Resuming does not exactly reproduce an uninterrupted run.
- `config.yaml` is copied before the CLI `--episodes` override. The effective budget appears in the configuration sent to W&B; also keep the command.
- The SAC fields `agent.log_std_min` and `agent.log_std_max` exist in the configuration but are not currently passed to the policy constructor. The effective bounds remain **−5 and 2**, as defined in `PolicyNetwork`.
- The W&B project is fixed to `atari-rl` in `trainer.py`, regardless of `logging.wandb_project`.

For a stronger experimental comparison, keep the command, YAML file, versions, seeds, hardware, and rewards from multiple evaluations together. On CarRacing, track diversity makes evaluation on multiple tracks particularly useful.
