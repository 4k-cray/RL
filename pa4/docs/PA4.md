# PA #4 (Chapter 7) - N-Step TD & N-Step SARSA

## Algorithm Details
| N-Step TD |             |
| ---------------------- | ----------- |
| **On/off policy:**     | On-policy  |
| **Target policy:**     | None        |
| **Policy updates:**    | n-Step         |
| **Control or Prediction:** | Prediction |
| **Observation space:** | Discrete |
| **Action space:**      | Discrete |
| **Objective:**         | Learn $V_{\pi}$


| N-Step SARSA |             |
| ---------------------- | ----------- |
| **On/off policy:**     | Off-policy  |
| **Target policy:**     | None        |
| **Policy updates:**    | n-Step         |
| **Control or Prediction:** | Control |
| **Observation space:** | Discrete |
| **Action space:**      | Discrete |
| **Objective:**         | Learn $Q^{*}$, $\pi^{*}$ using $\pi_{behavioral}$

## Learning Objectives
* Learn the value function for a policy $V_{\pi}$ using trajectories collected using an unknown policy.
* Learn the ideal state-action function $Q^{*}$ and the ideal policy $\pi^{*}$, using trajectories from a behavioral policy.

### Description
In this programming assignment, you will implement 2 algorithms introduced in Chapter 7. General instructions and requirements can be found in `n_step_bootstrap.py`. For this assignment, both of the algorithms are described in the textbook using psuedocode:

* On-policy n-step TD for evaluating a policy (**Page 144**).
* Off-policy n-step SARSA for learning an optimal policy (**Page 149**).

> **IMPORTANT!** There is an errata in the 2018 edition of the textbook, which is corrected in the 2020 edition. Please see the authors' [errata and notes](http://incompleteideas.net/book/errata.html) for page 149.

## Coding Details
### on_policy_n_step_td()
For this function, you are implementing the on-policy n-Step TD algorithm.
Trajectories are already generated for you using (to you) an unknown policy.
Your goal is to learn the value function for the policy that is generating
these trajectories.

Insert your code. For your inner loop, start with the `run.py` script. Try running: `python run.py n_step_bootstrap --help` to see the options available.
You may wish to start with a simple environment, like the `GridWorld2x2-v0`.
You also may wish to initially limit the number of episodes run.

Lastly, the default hyperparameters use a value of `n=1`. Once you have this creating expected results, you'll want to switch to `n=2` and perform testing.
If you algorithm is correct, you should see similar results to `n=1`.

Once you feel good about the implementation, you can run `python test.py n_step_bootstrap`. 

We run a simple test in the local grader, for OneStateMDP with n=2.

### class NStepSARSA
Similar to PA #1, we will again utilize the Solver interface. Here, you
will implement the `train_episode()` function.

You can choose to run the entire episode up-front, or you can start training
after n number of steps. The choice is yours, both implementations should work
correctly.

Implement the algorithm. Similarly to TD, you may wish to limit the number of episodes, or change the N value when debugging your code. At the end, you can
run the same test script.

>**Important:** some of the tests may fail from time to time due to stochasticity.

### Note on `policy_deterministic_greedy.py`
You will also need to turn in your `policy_deterministic_greedy.py` from PA2.

## Deliverables
You will turn in two files to Gradescope:
* `n_step_bootstrap.py`
* `policy_deterministic_greedy.py`

Each algorithm in `n_step_bootstrap.py` is worth 50% of the assignment grade.

## Tips & Tricks
It is very important to implement algorithms very carefully since these algorithms can fail due to minor implementation errors, which are almost impossible to debug. Students who implement the algorithm as faithfully to the psuedocode as they can tend to finish this assignment sooner and with fewer bugs.
