# Setup

## Github

```
git init
```

## Dependancies

- Docker (for Wandb)
- Graphviz (for Torchview)
- MoviePy (to record videos)

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

```
wandb.log()
```

```
C:\Users\svetl\AppData\Local\Programs\Python\Python313\pythonw.exe -m venv .venv313
.venv313\Scripts\activate
pip install -r requirements313.txt
```

# Setup jupyter notebook

Install python libraries
```
pip install jupyter ipykernel
```

Create the jupyter kernel
```
python -m ipykernel install --user --name rl-env --display-name "Python (RL Environment)"
```