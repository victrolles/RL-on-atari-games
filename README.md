# Deep Reinforcement Learning — DQN & SAC

Implémentations en **PyTorch** de DQN et Soft Actor-Critic pour apprendre à équilibrer, marcher et conduire dans quatre environnements **Gymnasium**. Les expériences partagent une boucle d'entraînement, des configurations YAML, un suivi **Weights & Biases** et des sauvegardes de modèles.

## Résultats en action

<table>
  <tr>
    <th align="center">CartPole · DQN</th>
    <th align="center">Pendulum · SAC</th>
  </tr>
  <tr>
    <td align="center">
      <a href="docs/trainings/dql_on_cartpole/dql_on_cartpole.md"><img src="docs/trainings/dql_on_cartpole/record_2.gif" alt="Agent DQN entraîné sur CartPole" width="400"></a><br>
      <a href="docs/trainings/dql_on_cartpole/dql_on_cartpole.md">Résultats, courbes et hyperparamètres</a>
    </td>
    <td align="center">
      <a href="docs/trainings/sac_on_pendulum/sac_on_pendulum.md"><img src="docs/trainings/sac_on_pendulum/record_2.gif" alt="Agent SAC entraîné sur Pendulum" width="400"></a><br>
      <a href="docs/trainings/sac_on_pendulum/sac_on_pendulum.md">Résultats, courbes et hyperparamètres</a>
    </td>
  </tr>
  <tr>
    <th align="center">BipedalWalker · SAC</th>
    <th align="center">CarRacing · SAC + CNN</th>
  </tr>
  <tr>
    <td align="center">
      <a href="docs/trainings/sac_on_bipedalwalker/sac_on_bipedalwalker.md"><img src="docs/trainings/sac_on_bipedalwalker/record_2.gif" alt="Agent SAC entraîné sur BipedalWalker" width="400"></a><br>
      <a href="docs/trainings/sac_on_bipedalwalker/sac_on_bipedalwalker.md">Résultats, courbes et hyperparamètres</a>
    </td>
    <td align="center">
      <a href="docs/trainings/sac_on_carracing/sac_on_carracing.md"><img src="docs/trainings/sac_on_carracing/record_2.gif" alt="Agent SAC avec encodeurs CNN entraîné sur CarRacing" width="400"></a><br>
      <a href="docs/trainings/sac_on_carracing/sac_on_carracing.md">Résultats, courbes et hyperparamètres</a>
    </td>
  </tr>
</table>

| Environnement | Agent | Dernière récompense moyenne d'évaluation¹ | Temps à prévoir² |
| :--- | :--- | ---: | :--- |
| `CartPole-v1` | DQN · réseau dense | 500,00 | Quelques minutes, environ 2 à 5 min |
| `Pendulum-v1` | SAC · réseaux denses | −148,81 | Environ 2 min |
| `BipedalWalker-v3` | SAC · réseaux denses | 334,42 | Environ 10 h |
| `CarRacing-v3` | SAC · encodeurs convolutifs | 311,47 | Environ une journée, 29 h sur l'essai relevé |

¹ Valeurs des résumés W&B locaux, archivées dans [les résultats de référence](docs/trainings/README.md). Les récompenses ont des échelles différentes selon l'environnement. Les GIF et les courbes historiques ne sont pas tous rattachés avec certitude à ces mêmes essais.

² Ordres de grandeur sur une **NVIDIA GeForce GTX 1650 de 4 Go**, sous Windows et Python 3.13. Ils décrivent les essais disponibles, qui n'ont pas nécessairement épuisé le budget d'épisodes configuré. Le matériel, les paramètres, les évaluations et les enregistrements influencent la durée. Chaque fiche précise les mesures retrouvées.

[Setup](#setup) · [Lancer un entraînement](#lancer-un-entraînement) · [Configurer une expérience](#configurer-une-expérience) · [Documentation](#documentation)

## Setup

Prérequis : **Python 3.13** pour retrouver l'environnement de travail utilisé, **Git** et **Docker Desktop** démarré pour le serveur W&B local. Un GPU NVIDIA compatible CUDA est utile pour les deux expériences longues ; le code utilise le CPU si CUDA est indisponible.

Depuis PowerShell :

```powershell
git clone https://github.com/victrolles/RL-on-atari-games.git
cd RL-on-atari-games
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Le fichier de dépendances ajoute l'index PyTorch **CUDA 13.2**. Pour adapter l'installation au matériel, résoudre un problème Box2D ou configurer Jupyter, consulter le [guide d'installation détaillé](docs/setup.md).

Lancer ensuite le serveur **Weights & Biases local** et connecter le client :

```powershell
wandb server start
$env:WANDB_BASE_URL = "http://localhost:8080"
wandb login --host=http://localhost:8080
```

Ouvrir [localhost:8080](http://localhost:8080), terminer la configuration initiale du serveur et utiliser la clé API du compte local pour la connexion. Les expériences apparaîtront dans le projet **`atari-rl`**. Voir le [démarrage officiel de W&B Server](https://github.com/wandb/server#quickstart), notamment pour l'activation de la licence locale.

Pour travailler sans serveur, le [guide de setup](docs/setup.md#mode-hors-ligne) décrit le mode hors ligne.

## Lancer un entraînement

Exécuter **une commande au choix**, depuis la racine du dépôt, avec l'environnement virtuel activé et W&B configuré :

```powershell
# CartPole : DQN, quelques minutes
python main.py --config configs/dqn_on_cartpole.yaml

# Pendulum : SAC, environ 2 minutes
python main.py --config configs/sac_on_pendulum.yaml

# BipedalWalker : SAC, environ 10 heures
python main.py --config configs/sac_on_bipedalwalker.yaml

# CarRacing : SAC avec CNN, environ une journée
python main.py --config configs/sac_on_carracing.yaml
```

Toujours fournir `--config` pour un nouvel entraînement : le chemin utilisé par défaut dans `main.py`, `configs/sac.yaml`, n'existe pas dans le dépôt.

### Ajuster le nombre d'épisodes

```powershell
python main.py --config configs/sac_on_pendulum.yaml --episodes 80
```

`--episodes` définit le **nombre total d'épisodes visé**. Un lancement très court peut se terminer avant la première évaluation ou avant le début des mises à jour ; il ne produira alors pas de checkpoint.

### Reprendre un entraînement

Remplacer `run_XXXX` par le dossier affiché au lancement :

```powershell
python main.py --checkpoint runs/Pendulum-v1/run_XXXX/checkpoints/latest.pt --episodes 80
```

Utiliser `best.pt` pour repartir du meilleur checkpoint évalué. La reprise recharge le `config.yaml` de l'expérience : **ne pas combiner `--checkpoint` et `--config`**. Elle restaure les réseaux, les optimiseurs et la progression, mais le replay buffer repart vide. Elle crée également une nouvelle session W&B.

## Configurer une expérience

**Pour modifier les entraînements, éditer simplement le fichier `.yaml` correspondant dans [`configs/`](configs/).** Les champs omis utilisent les valeurs par défaut de [`configs/config.py`](configs/config.py).

| Section | Paramètres utiles | Rôle |
| :--- | :--- | :--- |
| `env` | `name`, `device`, `normalize` | Environnement, CPU/CUDA et remise à l'échelle des actions |
| `network` | `hidden_size` | Largeur des couches denses |
| `training` | `nb_episodes`, `learning_starts`, `random_steps` | Budget, début de l'apprentissage et exploration initiale |
| `training` | `train_freq`, `gradient_steps`, `evaluation_interval` | Fréquence des mises à jour et des évaluations |
| `replay` | `capacity`, `batch_size` | Taille de la mémoire d'expérience et des lots |
| `agent` | Taux d'apprentissage, `gamma`, paramètres DQN ou SAC | Réglages de l'algorithme |
| `logging` | `run_dir` | Répertoire parent des expériences |

Par exemple, dans la configuration de Pendulum :

```yaml
training:
  nb_episodes: 80
  evaluation_interval: 10
```

Cet extrait montre les champs à modifier dans le fichier existant. Les évaluations utilisent actuellement **20 épisodes déterministes**, suivis de **3 épisodes enregistrés** ; ces nombres sont définis dans [`trainer.py`](trainer.py). Les fiches détaillent tous les hyperparamètres et les quelques réglages qui restent fixés dans le code.

### Retrouver les sorties

```text
runs/<environnement>/run_XXXX/
├── config.yaml         # Configuration copiée à la création de l'expérience
├── checkpoints/
│   ├── latest.pt       # Dernière évaluation sauvegardée
│   └── best.pt         # Meilleure moyenne d'évaluation sauvegardée
├── models/             # Poids et métadonnées exportés aux évaluations
├── records/            # Vidéos MP4 des épisodes enregistrés
└── wandb/              # Journaux locaux W&B
```

Les sauvegardes et les vidéos sont produites à chaque `evaluation_interval`, sans sauvegarde finale supplémentaire garantie entre deux évaluations. `runs/` est ignoré par Git ; les rendus destinés à la documentation sont conservés dans `docs/trainings/`.

Le fichier de configuration est copié **avant** l'application de `--episodes` : en cas de surcharge en ligne de commande, conserver aussi la commande utilisée. Le script `inference.py` est encore une ébauche ; les évaluations et vidéos utilisables passent actuellement par l'entraînement.

## Organisation du projet

```text
agents/          # Agents DQN et SAC, réseaux denses et convolutifs
configs/         # Configurations YAML et chargement typé
docs/            # Installation, résultats et notes techniques
models/          # Sélection de poids conservés dans le dépôt
tests/           # Scripts et notebooks d'exploration et de vérification
env.py           # Environnements, prétraitements et enregistrement vidéo
main.py          # Point d'entrée et reprise des expériences
replay_buffer.py # Mémoire d'expérience partagée
trainer.py       # Entraînement, évaluation, métriques et sauvegardes
```

## Documentation

- [Installation détaillée, W&B, CUDA et Jupyter](docs/setup.md)
- [Résultats de référence et protocole de lecture](docs/trainings/README.md)
- [CartPole — DQN](docs/trainings/dql_on_cartpole/dql_on_cartpole.md)
- [Pendulum — SAC](docs/trainings/sac_on_pendulum/sac_on_pendulum.md)
- [BipedalWalker — SAC](docs/trainings/sac_on_bipedalwalker/sac_on_bipedalwalker.md)
- [CarRacing — SAC avec CNN](docs/trainings/sac_on_carracing/sac_on_carracing.md)
- [Note technique : scheduler du taux d'apprentissage](docs/technologies/lr_scheduler.md)

Malgré le nom historique du dépôt, ces quatre expériences utilisent les environnements **Classic Control** et **Box2D** de Gymnasium.
