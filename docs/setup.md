# Installer et lancer le projet

[← README](../README.md) · [Commandes d'entraînement](../README.md#lancer-un-entraînement)

Les commandes principales ci-dessous ciblent **Windows / PowerShell**, l'environnement utilisé pour les expériences de référence.

## Prérequis

| Outil | Utilisation |
| :--- | :--- |
| Python 3.13 | Version utilisée par les essais locaux : CPython 3.13.16 |
| Git | Récupérer le dépôt |
| Docker Desktop | Héberger le serveur W&B local ; démarrer Docker avant W&B |
| GPU NVIDIA et pilote compatible | Accélérer PyTorch avec CUDA, particulièrement pour CarRacing et BipedalWalker |
| Graphviz, facultatif | Générer les schémas de réseaux avec Torchview |

Les essais locaux ont été exécutés avec une **GeForce GTX 1650 de 4 Go**. Le code sélectionne le CPU lorsque `env.device` vaut `cpu` ou que CUDA est indisponible ; les durées peuvent alors augmenter fortement.

## 1. Récupérer le dépôt et créer le venv

```powershell
git clone https://github.com/victrolles/RL-on-atari-games.git
cd RL-on-atari-games
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si plusieurs versions de Python sont installées, utiliser `py -3.13 -m venv .venv` à la place de `python -m venv .venv`. Dans un dépôt déjà cloné, commencer directement par la création du venv.

Les dépendances couvrent PyTorch, Gymnasium et ses extras `box2d` et `other`, W&B, NumPy, tqdm, Torchview et Jupyter. PyYAML est importé par le chargeur de configuration et installé via les dépendances de W&B ; MoviePy, utilisé par `RecordVideo`, est fourni via l'extra Gymnasium `other`.

Sous Linux ou macOS, activer le venv avec :

```bash
source .venv/bin/activate
```

### Adapter PyTorch au matériel

[`requirements.txt`](../requirements.txt) déclare un index additionnel `https://download.pytorch.org/whl/cu132`. Le choix d'une distribution PyTorch doit correspondre à la plateforme et au pilote : consulter le [sélecteur officiel PyTorch](https://pytorch.org/get-started/locally/) pour adapter l'installation.

Vérifier le résultat :

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA disponible:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Pour choisir explicitement le CPU, modifier le YAML concerné :

```yaml
env:
  device: cpu
```

### Si l'installation de Box2D échoue

Si `pip` tente une compilation et signale l'absence de SWIG :

```powershell
python -m pip install swig
python -m pip install "gymnasium[box2d]"
python -m pip install -r requirements.txt
```

Si une erreur de compilateur persiste sous Windows, installer les outils C++ demandés par le message d'erreur, puis relancer l'installation. La disponibilité de roues précompilées dépend de Python et de la version de Gymnasium ; voir les [instructions Box2D officielles](https://gymnasium.farama.org/environments/box2d/).

## 2. Configurer le serveur W&B local

### Démarrer et connecter le client

Avec Docker Desktop démarré et le venv activé :

```powershell
wandb server start
```

Ouvrir [http://localhost:8080](http://localhost:8080) et terminer l'assistant initial. Si une licence est demandée, suivre le [guide officiel W&B Server](https://github.com/wandb/server#quickstart) et la renseigner dans `/system-admin`.

Dans le terminal qui lancera l'entraînement :

```powershell
$env:WANDB_BASE_URL = "http://localhost:8080"
wandb login --host=http://localhost:8080
```

Utiliser la **clé API du compte créé sur le serveur local**. Le code appelle `wandb.init()` et publie ses métriques dans le projet **`atari-rl`**. Le nom du projet est actuellement fixé dans `trainer.py` ; modifier `logging.wandb_project` seul n'a pas d'effet.

Les variables `$env:...` s'appliquent au terminal courant et aux processus qu'il lance. Le fichier `wandb.env` présent dans le dépôt n'est pas chargé automatiquement par le code.

Sous Linux ou macOS, définir l'URL ainsi :

```bash
export WANDB_BASE_URL="http://localhost:8080"
wandb login --host=http://localhost:8080
```

### Arrêter ou changer le port

```powershell
wandb server stop
```

Si le port 8080 est occupé, arrêter le serveur avant de le relancer sur un autre port :

```powershell
wandb server start --port 9090
$env:WANDB_BASE_URL = "http://localhost:9090"
wandb login --host=http://localhost:9090
```

Le serveur utilise un volume Docker persistant nommé `wandb` pour ses données. La commande `wandb server start --help` décrit les options disponibles dans la version installée.

### Mode hors ligne

Pour entraîner sans serveur ni connexion W&B :

```powershell
$env:WANDB_MODE = "offline"
python main.py --config configs/sac_on_pendulum.yaml
```

Les métriques sont conservées dans le dossier `wandb/` de l'expérience. Pour revenir au suivi en ligne :

```powershell
Remove-Item Env:WANDB_MODE -ErrorAction SilentlyContinue
```

Pour envoyer ensuite une expérience hors ligne au serveur configuré :

```powershell
wandb sync runs/Pendulum-v1/run_XXXX/wandb/offline-run-XXXX
```

Remplacer les deux noms de dossiers par ceux réellement créés, et vérifier l'URL du serveur avant la synchronisation.

## 3. Vérifier l'installation

Ces vérifications n'entraînent aucun agent :

```powershell
python -c "import torch, gymnasium, wandb, yaml, moviepy; print('Dépendances disponibles')"
python main.py --help
wandb server --help
```

Pour vérifier les quatre environnements et leurs dépendances physiques :

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

Lancer ensuite [l'une des quatre commandes d'entraînement](../README.md#lancer-un-entraînement), toujours depuis la racine du dépôt. Pendulum constitue une première expérience courte.

## 4. Jupyter et outils facultatifs

Jupyter et ipykernel figurent déjà dans les dépendances. Enregistrer le venv comme noyau :

```powershell
python -m ipykernel install --user --name rl-env --display-name "Python (RL Environment)"
jupyter lab
```

Pour les schémas Torchview, installer également l'application [Graphviz](https://graphviz.org/download/) et rendre la commande `dot` accessible dans le `PATH`. Le paquet Python `graphviz` ne fournit pas à lui seul cette application.

## Dépannage courant

| Symptôme | Action |
| :--- | :--- |
| PowerShell bloque `Activate.ps1` | Utiliser directement `.\.venv\Scripts\python.exe` et `.\.venv\Scripts\wandb.exe` à la place de `python` et `wandb` dans les commandes |
| `wandb` ou une dépendance est introuvable | Vérifier le venv actif, puis relancer `python -m pip install -r requirements.txt` |
| `ModuleNotFoundError: yaml` ou `moviepy` | Installer explicitement `python -m pip install PyYAML moviepy` dans le venv |
| Docker n'est pas joignable | Démarrer Docker Desktop, puis relancer `wandb server start` |
| Connexion W&B refusée | Vérifier l'URL, le port et la clé du compte local ; réessayer avec `wandb login --relogin --host=http://localhost:8080` |
| PyTorch annonce `CUDA disponible: False` | Vérifier le pilote et la distribution PyTorch installée, ou utiliser `env.device: cpu` |
| Mémoire insuffisante sur CarRacing | Réduire `replay.capacity` pour la RAM et `replay.batch_size` pour la mémoire GPU, puis créer une nouvelle expérience |
| Aucun checkpoint après un lancement court | Attendre au moins `training.evaluation_interval` épisodes ; les sauvegardes sont liées aux évaluations |
