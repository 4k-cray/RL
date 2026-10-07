import numpy as np
from typing import override
from interfaces.policy import Policy

class Policy_DeterministicGreedy(Policy):
    def __init__(self, Q: np.ndarray[np.float64]):
        """
        Parameters:
        - Q (np.ndarray): Q function; numpy array shape of [nS,nA]
        """
        self.Q = Q

    @override
    def action(self, state: int) -> int:
        """
        Chooses the action that maximizes the Q function for the given state.

        Parameters:
            - state (int): state index

        Returns:
            - int: index of the action to take
        """

        ### TODO: Implement the action method ###
        return max(range(self.Q.shape[1]), key=lambda a: self.Q[state, a])


    @override
    def action_prob(self, state: int, action: int) -> float:
        """
        Returns the probability of taking the action if we are in the given state.

        Since this is a greedy policy, this will be a 1 or a 0.

        Parameters:
            - state (int): state index
            - action (int): action index

        Returns:
            - float: the probability of taking the action in the given state
        """

        ### TODO: Implement the action_prob method ###
        greedy_action = self.action(state)
        return 1.0 if action == greedy_action else 0.0