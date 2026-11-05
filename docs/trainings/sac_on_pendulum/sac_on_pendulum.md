# Pendulum — Soft Actor-Critic

[← README](../../../README.md) · [Résultats et protocole](../README.md) · [Installation](../../setup.md)

Apprendre à redresser puis stabiliser un pendule en contrôlant un couple continu. L'observation comporte trois composantes et l'agent produit une action. La récompense pénalise l'écart à la verticale, la vitesse et l'effort : une valeur moins négative indique une meilleure performance. [Description Gymnasium](https://gymnasium.farama.org/environments/classic_control/pendulum/).

## Avant / après

<table>
  <tr><th align="center">Enregistrement initial</th><th align="center">Après entraînement</th></tr>
  <tr>
    <td align="center"><img src="record_1.gif" alt="Pendulum : enregistrement initial" width="400"></td>
    <td align="center"><img src="record_2.gif" alt="Pendulum : agent entraîné" width="400"></td>
  </tr>
</table>

## Résultats et durée

| Indicateur | Valeur |
| :--- | :--- |
| Temps à prévoir | Environ 2 min sur la machine de référence |
| Durée notée dans la documentation historique | 1 min 55 s |
| Durée du résumé W&B local | 1 min 50 s |
| Dernière moyenne d'évaluation du résumé local | −148,81 |
| Interactions cumulées du résumé local | 8 000 |
| Essai local | `run_4248` · W&B `lwta13av` |
| Matériel de référence | NVIDIA GeForce GTX 1650 · 4 Go |

Pendulum est l'expérience SAC la plus courte du dépôt : le YAML prévoit 40 épisodes. Le résumé local donne un résultat encourageant, mais les courbes historiques montrent une progression irrégulière. Les [sources archivées](../results.json) et le [protocole de lecture](../README.md) distinguent ces relevés.

## Méthode et technologies

- **SAC** : un acteur gaussien, deux critiques et deux critiques cibles, avec mise à jour douce des cibles.
- **PyTorch** : réseaux denses à activations ReLU et échantillonnage reparamétré, suivi d'une transformation `tanh`.
- **Adam** : optimiseurs de l'acteur, des deux critiques et du coefficient d'entropie `alpha`.
- **MSE** pour les critiques ; objectif de l'acteur combinant Q-valeur et entropie. La perte de l'acteur n'est pas une MSE.
- **Gymnasium**, **W&B** et **RecordVideo** : simulation, métriques et vidéos.

Les actions produites dans `[-1, 1]` sont remises à l'échelle par `RescaleAction` dans l'intervalle de couple de Pendulum. L'entropie cible vaut `−1`, selon la dimension de l'action. Aucun scheduler de taux d'apprentissage n'est utilisé pour SAC dans le code actuel.

## Courbes et analyse

### Récompenses

<table>
  <tr><th align="center">Récompense d'entraînement</th><th align="center">Récompense d'évaluation</th></tr>
  <tr><td align="center"><img src="chart_reward_training.svg" alt="Récompense d'entraînement" width="400"></td><td align="center"><img src="chart_reward_evaluation.svg" alt="Récompense d'évaluation" width="400"></td></tr>
</table>

La récompense d'entraînement passe de valeurs souvent proches de −1 500 à plusieurs épisodes nettement meilleurs, parfois proches de zéro. Les rechutes restent nombreuses : l'amélioration ne constitue pas un plateau stable.

L'export d'évaluation montre un très bon point autour de −100, suivi d'un recul sous −1 200, puis d'une récupération partielle. Il ne suffit donc pas à conclure à une convergence régulière. Sa dernière valeur diffère du résumé W&B local à −148,81 : ces deux relevés sont présentés séparément.

### Pertes de l'acteur et du critique

<table>
  <tr><th align="center">Perte de l'acteur</th><th align="center">Perte du critique exportée</th></tr>
  <tr><td align="center"><img src="chart_loss_actor.svg" alt="Perte de l'acteur" width="400"></td><td align="center"><img src="chart_loss_critic.svg" alt="Perte du critique exportée" width="400"></td></tr>
</table>

La perte de l'acteur augmente, tandis que celle du critique comporte des pics ponctuels dépassant 400. L'objectif SAC de l'acteur dépend de l'échelle des Q-valeurs et du terme d'entropie : sa hausse ne signifie pas directement une baisse de performance. Les pics du critique signalent des périodes de plus grande erreur sur les cibles de valeur ; les récompenses restent le critère principal pour juger le comportement.

### Valeurs estimées et exploration

<table>
  <tr><th align="center">Q-valeur moyenne</th><th align="center">Log-probabilité des actions</th></tr>
  <tr><td align="center"><img src="chart_mean_q.svg" alt="Q-valeur moyenne" width="400"></td><td align="center"><img src="chart_log_prob.svg" alt="Log-probabilité des actions" width="400"></td></tr>
  <tr><th align="center">Écart-type de la politique</th><th align="center">Log de l'écart-type</th></tr>
  <tr><td align="center"><img src="chart_std.svg" alt="Écart-type de la politique" width="400"></td><td align="center"><img src="chart_log_std.svg" alt="Log de l'écart-type" width="400"></td></tr>
</table>

La Q-valeur moyenne devient plus négative au cours de l'export, ce qui est cohérent avec l'apprentissage de retours négatifs dans cet environnement. Cela ne prouve ni une amélioration ni une dégradation de la politique sans comparaison aux retours observés.

L'écart-type de la politique diminue globalement d'environ 0,9 à 0,5, et son logarithme suit la même tendance. Sur les observations échantillonnées, la distribution gaussienne devient plus concentrée. La log-probabilité augmente globalement avec des fluctuations ; ces métriques décrivent l'exploration de la politique, pas un taux de réussite.

Les huit SVG sont conservés avec leur axe `Step` d'origine. Leur correspondance exacte avec les épisodes et avec la session locale n'est pas établie.

## Reproduire l'expérience

Depuis la racine du dépôt, après le [setup](../../setup.md) :

```powershell
python main.py --config configs/sac_on_pendulum.yaml
```

Pour prolonger le budget lors d'un nouvel essai :

```powershell
python main.py --config configs/sac_on_pendulum.yaml --episodes 80
```

## Hyperparamètres

Valeurs de la [configuration actuelle](../../../configs/sac_on_pendulum.yaml), complétées par les défauts de [`configs/config.py`](../../../configs/config.py). Elles décrivent le lancement proposé, sans garantir les réglages des courbes historiques.

### Environnement et réseau

| Paramètre | Valeur |
| :--- | :--- |
| `env.name` | `Pendulum-v1` |
| `env.device` | `cuda` |
| `env.normalize` | `true` |
| `network.hidden_size` | `64` |

### Entraînement et replay buffer

| Paramètre | Valeur |
| :--- | :--- |
| `training.nb_episodes` | `40` |
| `training.learning_starts` | `1000` |
| `training.random_steps` | `1000` |
| `training.train_freq` | `5` |
| `training.gradient_steps` | `5` |
| `training.evaluation_interval` | `10` |
| `replay.capacity` | `10000` |
| `replay.batch_size` | `64` |

`learning_starts`, `random_steps` et `train_freq` sont exprimés en interactions avec l'environnement. `gradient_steps` est le nombre de mises à jour par déclenchement ; `evaluation_interval` est exprimé en épisodes.

### Optimisation

| Paramètre | Valeur |
| :--- | :--- |
| `agent.actor_lr` | `0.003` |
| `agent.critic_lr` | `0.003` |
| `agent.alpha_lr` | `0.003` |
| `agent.gamma` | `0.99` |
| `agent.tau` | `0.005` |
| `agent.alpha` | `0.2` |

`alpha` est la valeur initiale du coefficient d'entropie, ensuite appris automatiquement. Les bornes `log_std` configurées sont `-20` et `2`, mais le constructeur utilise actuellement les bornes effectives **−5 et 2** : ces deux champs YAML ne lui sont pas transmis.

Le suivi utilise le projet W&B `atari-rl` et crée les sorties sous `runs/` par défaut. Les évaluations comprennent **20 épisodes déterministes**, puis **3 épisodes enregistrés**, selon le [protocole commun](../README.md#protocole-du-code-actuel).
