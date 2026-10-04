# DQN On CartPole

![Bot playing CartPole](medias\eval-episode-22.gif)

Note: Most of the environment description and technical details in this section are adapted from the official [Gymnasium documentation](https://gymnasium.farama.org/environments/classic_control/cart_pole/). The explanations and implementation-specific details have been added or adapted where relevant to this project.

## The game

This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in “[Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem](https://ieeexplore.ieee.org/document/6313077)”. A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces in the left and right direction on the cart.

### Action Space

The action is a ndarray with shape (1,) which can take values {0, 1} indicating the direction of the fixed force the cart is pushed with.

- 0: Push cart to the left
- 1: Push cart to the right

Eg:
```
Action : 0
```

### Observation Space

The observation is a ndarray with shape (4,) with the values corresponding to the following positions and velocities:

| Num | Observation              | Min                    | Max                   |
|-----|---------------------------|------------------------|-----------------------|
| 0   | Cart Position             | -4.8                   | 4.8                   |
| 1   | Cart Velocity              | -Inf                   | Inf                   |
| 2   | Pole Angle                 | ~ -0.418 rad (-24°)    | ~ 0.418 rad (24°)     |
| 3   | Pole Angular Velocity      | -Inf                   | Inf                   |

Eg:
```
Observation: [ 0.03651603 -0.15809521 -0.04385493  0.23530404]
```

## The model

The model used is a dense neural network made with torch.nn modules

![architecture of the model](medias\DQN_CartPole_Model.gv.png)