# Install and run the project

[← README](../README.md) · [Training commands](../README.md#run-training)

The main commands below target **Windows / PowerShell**, the environment used for the reference experiments.

## Prerequisites

| Tool | Purpose |
| :--- | :--- |
| Python 3.13 | Version used for local runs: CPython 3.13.16 |
| Git | Clone the repository |
| Docker Desktop | Host the local W&B server; start Docker before W&B |
| NVIDIA GPU and compatible driver | Accelerate PyTorch with CUDA, especially for CarRacing and BipedalWalker |
| Graphviz, optional | Generate network diagrams with Torchview |

Local runs used a **4 GB GeForce GTX 1650**. The code selects the CPU when `env.device` is `cpu` or CUDA is unavailable; runtimes may then increase substantially.

## 1. Clone the repository and create the virtual environment

```powershell
git clone https://github.com/victrolles/RL-on-atari-games.git
cd RL-on-atari-games
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If multiple Python versions are installed, use `py -3.13 -m venv .venv` instead of `python -m venv .venv`. If the repository is already cloned, start by creating the virtual environment.

Dependencies include PyTorch, Gymnasium and its `box2d` and `other` extras, W&B, NumPy, tqdm, Torchview, and Jupyter. PyYAML is imported by the configuration loader and installed through W&B's dependencies; MoviePy, used by `RecordVideo`, is provided through Gymnasium's `other` extra.

On Linux or macOS, activate the virtual environment with:

```bash
source .venv/bin/activate
```

### Adapt PyTorch to your hardware

[`requirements.txt`](../requirements.txt) specifies the additional index `https://download.pytorch.org/whl/cu132`. The PyTorch build must match your platform and driver: see the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) to adapt the installation.

Check the result:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

To explicitly select the CPU, edit the relevant YAML file:

```yaml
env:
  device: cpu
```

### If Box2D installation fails

If `pip` attempts a build and reports that SWIG is missing:

```powershell
python -m pip install swig
python -m pip install "gymnasium[box2d]"
python -m pip install -r requirements.txt
```

If a compiler error persists on Windows, install the C++ tools requested in the error message, then retry the installation. Prebuilt wheel availability depends on Python and the Gymnasium version; see the [official Box2D instructions](https://gymnasium.farama.org/environments/box2d/).

## 2. Configure the local W&B server

### Start the server and connect the client

With Docker Desktop running and the virtual environment activated:

```powershell
wandb server start
```

Open [http://localhost:8080](http://localhost:8080) and complete the initial setup wizard. If a license is requested, follow the [official W&B Server guide](https://github.com/wandb/server#quickstart) and enter it in `/system-admin`.

In the terminal that will launch training:

```powershell
$env:WANDB_BASE_URL = "http://localhost:8080"
wandb login --host=http://localhost:8080
```

Use the **API key of the account created on the local server**. The code calls `wandb.init()` and publishes its metrics to the **`atari-rl`** project. The project name is currently fixed in `trainer.py`; changing `logging.wandb_project` alone has no effect.

The `$env:...` variables apply to the current terminal and the processes it starts. The `wandb.env` file in the repository is not loaded automatically by the code.

On Linux or macOS, set the URL as follows:

```bash
export WANDB_BASE_URL="http://localhost:8080"
wandb login --host=http://localhost:8080
```

### Stop the server or change the port

```powershell
wandb server stop
```

If port 8080 is in use, stop the server before restarting it on another port:

```powershell
wandb server start --port 9090
$env:WANDB_BASE_URL = "http://localhost:9090"
wandb login --host=http://localhost:9090
```

The server stores its data in a persistent Docker volume named `wandb`. The `wandb server start --help` command describes the options available in the installed version.

### Offline mode

To train without a server or W&B connection:

```powershell
$env:WANDB_MODE = "offline"
python main.py --config configs/sac_on_pendulum.yaml
```

Metrics are kept in the experiment's `wandb/` directory. To return to online tracking:

```powershell
Remove-Item Env:WANDB_MODE -ErrorAction SilentlyContinue
```

To later upload an offline run to the configured server:

```powershell
wandb sync runs/Pendulum-v1/run_XXXX/wandb/offline-run-XXXX
```

Replace both directory names with those actually created, and check the server URL before syncing.

## 3. Verify the installation

These checks do not train any agents:

```powershell
python -c "import torch, gymnasium, wandb, yaml, moviepy; print('Dependencies available')"
python main.py --help
wandb server --help
```

To check all four environments and their physics dependencies:

```powershell
@'
import gymnasium as gym

for name in ("CartPole-v1", "Pendulum-v1", "BipedalWalker-v3", "CarRacing-v3"):
    env = gym.make(name)
    observation, _ = env.reset(seed=0)
    print(name, "observations:", observation.shape, "actions:", env.action_space)
    env.close()
'@ | python -
```

Then run [one of the four training commands](../README.md#run-training), again from the repository root. Pendulum is a short first experiment.

## 4. Jupyter and optional tools

Jupyter and ipykernel are already included in the dependencies. Register the virtual environment as a kernel:

```powershell
python -m ipykernel install --user --name rl-env --display-name "Python (RL Environment)"
jupyter lab
```

For Torchview diagrams, also install the [Graphviz](https://graphviz.org/download/) application and make the `dot` command available on your `PATH`. The Python `graphviz` package alone does not provide this application.

## Common troubleshooting

| Symptom | Action |
| :--- | :--- |
| PowerShell blocks `Activate.ps1` | Use `.\.venv\Scripts\python.exe` and `.\.venv\Scripts\wandb.exe` directly instead of `python` and `wandb` in the commands |
| `wandb` or a dependency cannot be found | Check the active virtual environment, then rerun `python -m pip install -r requirements.txt` |
| `ModuleNotFoundError: yaml` or `moviepy` | Explicitly install them with `python -m pip install PyYAML moviepy` in the virtual environment |
| Docker cannot be reached | Start Docker Desktop, then rerun `wandb server start` |
| W&B connection refused | Check the URL, port, and local account key; retry with `wandb login --relogin --host=http://localhost:8080` |
| PyTorch reports `CUDA available: False` | Check the driver and installed PyTorch build, or use `env.device: cpu` |
| Insufficient memory on CarRacing | Reduce `replay.capacity` for RAM and `replay.batch_size` for GPU memory, then create a new experiment |
| No checkpoint after a short run | Wait at least `training.evaluation_interval` episodes; saves are tied to evaluations |
