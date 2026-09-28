import numpy as np
import gymnasium as gym

from interfaces.policy import Policy

def Q2V(env: gym.Env, Q: np.ndarray, pi: Policy):
    # Compute V based on Q and policy pi in environment env
    V = np.zeros(env.observation_space.n)
    for s in range(env.observation_space.n):
        V[s] = 0
        for a in range(env.action_space.n):
            V[s] += pi.action_prob(s, a) * Q[s, a]
    return V