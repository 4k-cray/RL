from typing import Iterable, Tuple
import gymnasium as gym
import numpy as np

from interfaces.policy import RandomPolicy
from interfaces.solver import Solver, Hyperparameters

from assignments.policy_deterministic_greedy import Policy_DeterministicGreedy

def on_policy_n_step_td(
    trajs: Iterable[Iterable[Tuple[int,int,int,int]]],
    n: int,
    alpha: float,
    initV: np.array,
    gamma: float = 1.0
) -> Tuple[np.array]:
    """
    Runs the on-policy n-step TD algorithm to estimate the value function for a given policy.

    Sutton & Barto, p. 144, "n-step TD Prediction"

    Parameters:
        trajs (list): N trajectories generated using an unknown policy. Each trajectory is a 
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        n (int): The number of steps (the "n" in n-step TD)
        alpha (float): The learning rate
        initV (np.ndarray): initial V values; np array shape of [nS]
        gamma (float): The discount factor

    Returns:
        V (np.ndarray): $v_pi$ function; numpy array shape of [nS]
    """

    #####################
    # TODO: Implement On Policy n-Step TD algorithm
    # sampling (Hint: Sutton Book p. 144)
    #####################

    V = np.array(initV, dtype=float)
    for episode in trajs:
        T = len(episode)
        for t in range(T):
            G = 0.0
            for i in range(t, min(t+n, T)):
                state, action, reward, next_state = episode[i]
                G += (gamma ** (i-t)) * reward
            if t+n < T:
                state, action, reward, next_state = episode[t+n]
                G += (gamma ** n) * V[state]
            state, action, reward, next_state = episode[t]
            V[state] += alpha * (G - V[state])
    

    return V


class NStepSARSAHyperparameters(Hyperparameters):
    """ Hyperparameters for NStepSARSA algorithm """
    def __init__(self, gamma: float, alpha: float, n: int):
        """
        Parameters:
            gamma (float): The discount factor
            alpha (float): The learning rate
            n (int): The number of steps (the "n" in n-step SARSA)
        """
        super().__init__(gamma)
        self.alpha = alpha
        """The learning rate"""
        self.n = n
        """The number of steps (the "n" in n-step SARSA)"""

class NStepSARSA(Solver):
    """
    Solver for N-Step SARSA algorithm, good for discrete state and action spaces.

    Off-policy algorithm, using weighted importance sampling.
    """
    def __init__(self, env: gym.Env, hyperparameters: NStepSARSAHyperparameters):
        super().__init__("NStepSARSA", env, hyperparameters)
        self.pi = Policy_DeterministicGreedy(np.ones((env.observation_space.n, env.action_space.n)))

    def action(self, state):
        """
        Chooses an action based on the current policy.

        Parameters:
            state (int): The current state
        
        Returns:
            int: The action to take
        """
        return self.pi.action(state)

    def train_episode(self):
        """
        Trains the agent for a single episode.

        Returns:
            float: The total (undiscounted) reward for the episode
        """

        #####################
        # TODO: Implement Off Policy n-Step SARSA algorithm
        #   - Hint: Sutton Book p. 149
        #   - Hint: You'll need to build your trajectories using a behavior policy (RandomPolicy)
        #   - Hint: You can use the `pi.action_prob(state, action)` and `bpi.action_prob(state, action)` methods to get the action probabilities.
        #   - Hint: Be sure to check both terminated and truncated variables.
        #####################
        
        episode_G = 0.0
        n = self.hyperparameters.n
        alpha = self.hyperparameters.alpha
        gamma = self.hyperparameters.gamma
        bpi = RandomPolicy(self.env.action_space.n)

        state, _ = self.env.reset()
        states = [state]
        actions = [bpi.action(state)]
        rewards = [0.0]  # rewards[i] is R_i; there is no R_0.
        T = float('inf')
        t = 0

        while True:
            if t < T:
                next_state, reward, terminated, truncated, _ = self.env.step(actions[t])
                states.append(next_state)
                rewards.append(reward)
                episode_G += reward

                if terminated or truncated:
                    T = t + 1
                else:
                    actions.append(bpi.action(next_state))

            tau = t - n + 1
            if tau >= 0:
                # Exclude A_tau, but include the bootstrap action A_(tau+n).
                rho = 1.0
                for i in range(tau + 1, min(tau + n, T - 1) + 1):
                    rho *= (self.pi.action_prob(states[i], actions[i])
                            / bpi.action_prob(states[i], actions[i]))

                G = 0.0
                for i in range(tau + 1, min(tau + n, T) + 1):
                    G += gamma ** (i - tau - 1) * rewards[i]
                if tau + n < T:
                    G += gamma ** n * self.pi.Q[states[tau + n], actions[tau + n]]

                state, action = states[tau], actions[tau]
                self.pi.Q[state, action] += alpha * rho * (G - self.pi.Q[state, action])
                # self.pi computes its greedy action directly from the updated Q.

            if tau == T - 1:
                break
            t += 1

        return episode_G
