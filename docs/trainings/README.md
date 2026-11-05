# Résultats de référence et protocole

[← README](../../README.md) · [Installation](../setup.md)

## Les quatre expériences

| Fiche | Configuration | Récompense d'évaluation relevée | Durée W&B relevée |
| :--- | :--- | ---: | :--- |
| [CartPole · DQN](dql_on_cartpole/dql_on_cartpole.md) | [YAML](../../configs/dqn_on_cartpole.yaml) | 500,00 | 4 min 27 s |
| [Pendulum · SAC](sac_on_pendulum/sac_on_pendulum.md) | [YAML](../../configs/sac_on_pendulum.yaml) | −148,81 | 1 min 50 s |
| [BipedalWalker · SAC](sac_on_bipedalwalker/sac_on_bipedalwalker.md) | [YAML](../../configs/sac_on_bipedalwalker.yaml) | 334,42 | 9 h 33 min 52 s |
| [CarRacing · SAC + CNN](sac_on_carracing/sac_on_carracing.md) | [YAML](../../configs/sac_on_carracing.yaml) | 311,47 | 29 h 18 min 44 s |

Les chiffres sont les **dernières valeurs présentes dans les résumés W&B locaux**, pas nécessairement les meilleurs résultats atteints. Ils décrivent un essai par environnement, sans moyenne entre plusieurs entraînements indépendants. Les récompenses des quatre tâches ne sont pas directement comparables.

## Provenance des chiffres

Un extrait des données locales est conservé dans [`results.json`](results.json), afin que les chiffres restent consultables après clonage : `runs/` est ignoré par Git. Cet extrait reprend uniquement les métriques utiles, l'identifiant de l'essai et le matériel ; il ne contient aucune clé de connexion.

| Environnement | Dossier local de l'expérience | Identifiant W&B |
| :--- | :--- | :--- |
| CartPole | `runs/CartPole-v1/run_7834` | `gfulc9v8` |
| Pendulum | `runs/Pendulum-v1/run_4248` | `lwta13av` |
| BipedalWalker | `runs/BipedalWalker-v3/run_3025` | `yqzyzht5` |
| CarRacing | `runs/CarRacing-v3/run_8544` | `w5q1kcxw` |

Les sources d'origine sont les fichiers `wandb-summary.json` et `wandb-metadata.json`, dans `runs/<environnement>/<run>/wandb/<session>/files/`. Les durées du tableau correspondent à `_runtime`, converti en heures, minutes et secondes. Elles incluent le temps écoulé pendant le suivi W&B, avec les évaluations et les enregistrements ; elles ne constituent pas une mesure du seul calcul des gradients.

Les quatre sessions indiquent **Windows 11**, **CPython 3.13.16** et une **NVIDIA GeForce GTX 1650 de 4 Go**. Les notes historiques mentionnaient aussi **2 min 39 s** pour CartPole, **1 min 55 s** pour Pendulum et **9 h 30** pour BipedalWalker. Ces valeurs restent signalées dans les fiches, car elles ne coïncident pas toujours avec la session locale retrouvée.

## Lire les vidéos et les courbes

Chaque fiche présente les GIF `record_1.gif` et `record_2.gif` comme une comparaison entre un enregistrement initial et un enregistrement ultérieur. Leur épisode exact et leur correspondance avec les sessions ci-dessus ne sont pas documentés. Une vidéo illustre un comportement sur un épisode ; elle ne remplace pas une évaluation sur plusieurs épisodes.

Les SVG sont les exports W&B conservés dans chaque dossier. L'analyse porte sur leurs tendances visibles. Certaines figures sont historiques et diffèrent des résumés locaux : leurs valeurs ne sont donc pas fusionnées avec celles du tableau.

L'axe **`Step` dépend de l'export**. Dans le code actuel, `wandb.log()` est appelé à chaque mise à jour de l'agent, à chaque évaluation et en fin d'épisode. L'étape interne W&B ne correspond donc automatiquement ni au nombre d'épisodes ni au nombre d'interactions avec l'environnement. La métrique `Steps total` compte ces interactions. Sans réglage d'axe explicite dans l'export, on évite de convertir une abscisse en épisodes ou en secondes.

## Protocole du code actuel

- **Entraînement** : actions aléatoires pendant `random_steps`, puis politique de l'agent ; apprentissage à partir de `learning_starts` si le buffer contient un lot complet.
- **Mises à jour** : toutes les `train_freq` interactions, effectuer `gradient_steps` mises à jour sur des lots tirés du replay buffer.
- **Évaluation** : tous les `evaluation_interval` épisodes, calculer la moyenne sur **20 épisodes déterministes** dans un environnement distinct.
- **Vidéo** : enregistrer ensuite **3 épisodes déterministes** dans un troisième environnement.
- **Sauvegardes** : écrire `latest.pt`, améliorer éventuellement `best.pt` et exporter des poids dans `models/` à chaque évaluation.

La récompense d'entraînement comporte l'exploration ; celle d'évaluation utilise une politique déterministe. En SAC, celle-ci applique `tanh` à la moyenne de la politique ; en DQN, elle choisit l'action de Q-valeur maximale sans exploration epsilon-greedy.

## Hyperparamètres et reproductibilité

Les tables des fiches décrivent les **YAML actuellement versionnés** et les valeurs par défaut réellement utilisées. Elles ne prétendent pas reconstituer les paramètres de chaque export historique.

Quelques limites utiles pour reproduire ou comparer un essai :

- Les graines aléatoires ne sont pas fixées par la boucle d'entraînement. Refaire plusieurs essais et conserver leur dispersion avant de conclure à une amélioration.
- Les dépendances de `requirements.txt` ne sont pas épinglées. Conserver les versions installées avec les résultats d'une nouvelle expérience.
- Le replay buffer et les états des générateurs aléatoires ne sont pas sauvegardés dans les checkpoints. Une reprise ne reproduit pas exactement une exécution continue.
- `config.yaml` est copié avant la surcharge CLI `--episodes`. Le budget effectif figure dans la configuration transmise à W&B ; conserver également la commande.
- Les champs SAC `agent.log_std_min` et `agent.log_std_max` existent dans la configuration, mais ne sont actuellement pas transmis au constructeur de la politique. Les bornes effectives restent **−5 et 2**, définies dans `PolicyNetwork`.
- Le projet W&B est fixé à `atari-rl` dans `trainer.py`, indépendamment de `logging.wandb_project`.

Pour une comparaison expérimentale plus solide, conserver ensemble la commande, le YAML, les versions, les graines, le matériel et les récompenses de plusieurs évaluations. Sur CarRacing, la variété des circuits rend particulièrement utile l'évaluation sur plusieurs pistes.
