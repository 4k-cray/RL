# PA #3 (Chapter 5) - Off-policy Monte-Carlo Evaluation

## Algorithm Details
| Monte-Carlo Evaluation |             |
| ---------------------- | ----------- |
| **On/off policy:**     | Off-policy  |
| **Target policy:**     | None        |
| **Policy updates:**    | N/A         |
| **Control or Prediction:** | Prediction |
| **Observation space:** | Discrete |
| **Action space:**      | Discrete |
| **Objective:**         | Learn $V_{\pi}$ using $\pi_{behavioral}$

## Learning Objectives
* Learn the value function of a policy using trajectories collected
using a different policy.
* Learn difference between ordinary and weighted importance sampling.

## Introduction
In this programming assignment, you will implement 2 algorithms introduced in Chapter 5. General instructions and requirements can be found in `monte_carlo.py`. For this assignment, one of the algorithms is described using psuedocode in the textbook.

You'll need to implement an off-policy Monte-Carlo evaluation method. You need to write two different versions:
* Ordinary Importance Sampling *(Description on **Page 109**)*
* Weighted Importance Sampling *(**Page 110**)*

We suggest implementing the weighted version first, then adapting it to ordinary importance sampling (as described on page 109). Note: the change required here is very small.

Both of these are done in an "every-visit" manner.

## Coding Details
### `monte_carlo.py`

Open `assignments/monte_carlo.py`. You will need to implement 2 functions:
* `off_policy_mc_prediction_ordinary_importance_sampling()`
* `off_policy_mc_prediction_weighted_importance_sampling()`

First, you can run the tests, which will fail:

`python test.py monte_carlo`

Now, implement the functions until your tests are passing. The tests cover 2 environments, the `OneStateMDP-v0` and `GridWorld2x2-v0` from the previous environment.

### Visualizing your results
You can also run your code using an arbitrary policy and number of episodes. This is useful for debugging:
`python run.py monte_carlo --target_policy {POLICY_NAME} --behavior_policy {POLICY_NAME} --num_episodes {N}`

We have 3 different policies that can be used:
* `RandomPolicy` - an equiprobable random policy
* `{ENVIRONMENT}OptimalPolicy` - an optimal policy for a given environment
* `UnbalancedRandomPolicy` - a random policy with different weights for each action.

You should be able to use any of these as the behavioral policy. However, the choice of target policy should be what changes your results. For example, when using the ideal policy, your V should match the ideal value function ($V^*$).

>**Important:** some of the tests may fail from time to time due to stochasticity.

## Deliverables
You will turn in one file to Gradescope:
* `monte_carlo.py`