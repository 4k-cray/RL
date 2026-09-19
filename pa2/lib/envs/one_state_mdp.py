import numpy as np
import gymnasium as gym

from interfaces.policy import Policy

class OneStateMDP(gym.Env): # MDP introduced at Fig 5.4 in Sutton Book
    def __init__(self):
        super().__init__()

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(2)
        self.gamma = 1.0

        self.terminal_states = [1]
        self.P = self._build_trans_mat()

    def _build_trans_mat(self):
        P = np.zeros((self.observation_space.n, self.action_space.n), dtype=object)

        # State 0, Action 0
        P[0, 0] = [
            (0.9, 0, 0, False),
            (0.1, 1, 1, True) # this is the only way to receive a reward
        ]

        # If you take the other action, you loop back to the same state
        P[0, 1] = [
            (1.0, 1, 0, True)
        ]

        # Terminal state
        for n in range(self.action_space.n):
            P[1, n] = [
                (1.0, 1, 0, True)
            ]

        return P

    def render(self):
        pass

    def reset(self):
        self._state = 0
        return self._state, {}

    def step(self, action):
        assert action in list(range(self.action_space.n)), "Invalid Action"
        assert self._state not in self.terminal_states, "Episode has ended!"

        # grab the transition probabilities [(prob, next_state, reward, done)]
        transitions: list = self.P[self._state, action]
        
        # sample an index from transitions based on the prob
        probs, _, _, _ = zip(*transitions)

        i = np.random.choice(np.arange(len(transitions)), p=list(probs))
        prob, next_state, reward, done = transitions[i]
        
        self.state = next_state
        return self.state, reward, done, False, {}

# Off-policy evaluation test with optimal policy
class OneStateMDPOptimalPolicy(Policy):
    def action_prob(self, state, action):
        return 1.0 if self.action(state) == action else 0.0

    def action(self, state):
        return 0