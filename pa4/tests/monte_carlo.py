import numpy as np
import unittest
import random

from interfaces.random_policy import RandomPolicy
from lib.envs.grid_world_2x2 import GridWorld2x2, GridWorld2x2OptimalPolicy
from lib.envs.one_state_mdp import OneStateMDP, OneStateMDPOptimalPolicy
from lib.envs.generate_trajectories import generate_trajectories

from assignments.monte_carlo import off_policy_mc_prediction_ordinary_importance_sampling as mc_ois
from assignments.monte_carlo import off_policy_mc_prediction_weighted_importance_sampling as mc_wis

random.seed(0)

class BaseMCTests(unittest.TestCase):
    def _test(self, fn, env, pi, bpi, Q_star, num_episodes=25000):
        trajs = generate_trajectories(env, bpi, num_episodes)

        Q = fn(
            env.observation_space,
            env.action_space,
            trajs,
            bpi,
            pi,
            np.zeros((env.observation_space.n, env.action_space.n)))
        self.assertTrue(
            np.allclose(Q, Q_star, rtol=1e-1, atol=0.2),
            f'Expected {Q_star}, got {Q}'
        )

class TestOrdinaryImportanceSampling(BaseMCTests):
    """Ordinary Importance Sampling"""
    def test_mc_ois_on_policy(self):
        """env=OneStateMDP, behavior_policy=RandomPolicy, eval_policy=RandomPolicy"""
        env = OneStateMDP()
        self._test(
            mc_ois,
            env,
            RandomPolicy(env.action_space.n),
            RandomPolicy(env.action_space.n),
            np.array([[0.1, 0.], [0., 0.]])
        )
        
    def test_mc_ois_off_policy(self):
        """env=OneStateMDP, behavior_policy=RandomPolicy, eval_policy=OneStateMDPOptimalPolicy"""
        env = OneStateMDP()
        self._test(
            mc_ois,
            env,
            RandomPolicy(env.action_space.n),
            OneStateMDPOptimalPolicy(),
            np.array([[1.0, 0.], [0., 0.]])
        )

    def test_mc_ois_on_policy_small_gridworld(self):
        """env=GridWorld2x2, behavior_policy=RandomPolicy, eval_policy=RandomPolicy"""
        env = GridWorld2x2()
        self._test(
            mc_ois,
            env,
            RandomPolicy(env.action_space.n),
            RandomPolicy(env.action_space.n),
            np.array([
                [ -9, -9, -7, -7 ],
                [ -9, -7, -7, -1 ],
                [ -7, -9, -1, -7 ],
                [  0,  0,  0,  0 ]
            ])
        )
        
    def test_mc_ois_off_policy_small_gridworld(self):
        """env=GridWorld2x2, behavior_policy=RandomPolicy, eval_policy=GridWorld2x2OptimalPolicy"""
        env = GridWorld2x2()
        self._test(
            mc_ois,
            env,
            RandomPolicy(env.action_space.n),
            GridWorld2x2OptimalPolicy(),
            np.array([
                [  0,  0, -1.85, -1.85 ],
                [  0,  0,  0,    -1    ],
                [  0,  0, -1,     0    ],
                [  0,  0,  0,     0    ]
            ])
        )

    def test_mc_ois_off_policy_small_gridworld_unbalanced(self):
        """env=OneStateMDP, behavior_policy=UnequalWeightPolicy, eval_policy=RandomPolicy"""
        env = OneStateMDP()
        pi = RandomPolicy(env.action_space.n)
        bpi = RandomPolicy(env.action_space.n, p = np.array([1/3., 2/3.]))
        num_episodes=25000

        Q_star = np.array( [[0.18942553, 0.        ], [0.,         0.        ]])

        trajs = generate_trajectories(env, bpi, num_episodes)

        Q = mc_ois(
            env.observation_space,
            env.action_space,
            trajs,
            bpi,
            pi,
            np.zeros((env.observation_space.n, env.action_space.n)))
        
        self.assertTrue(
            np.allclose(Q, Q_star, rtol=1e-1, atol=0.1),
            f'Expected {Q_star}, got {Q}'
        )
       
class TestWeightedImportanceSampling(BaseMCTests):
    """Weighted Importance Sampling"""

    def test_mc_wis_on_policy(self):
        """env=OneStateMDP, behavior_policy=RandomPolicy, eval_policy=RandomPolicy"""
        env = OneStateMDP()
        self._test(
            mc_wis,
            env,
            RandomPolicy(env.action_space.n),
            RandomPolicy(env.action_space.n),
            np.array([[0.1, 0.], [0., 0.]])
        )
       
    def test_mc_wis_off_policy(self):
        """env=OneStateMDP, behavior_policy=RandomPolicy, eval_policy=OneStateMDPOptimalPolicy"""
        env = OneStateMDP()
        self._test(
            mc_wis,
            env,
            RandomPolicy(env.action_space.n),
            OneStateMDPOptimalPolicy(),
            np.array([[1.0, 0.], [0., 0.]])
        )

    def test_mc_wis_on_policy_small_gridworld(self):
        """env=GridWorld2x2, behavior_policy=RandomPolicy, eval_policy=RandomPolicy"""
        env = GridWorld2x2()
        self._test(
            mc_wis,
            env,
            RandomPolicy(env.action_space.n),
            RandomPolicy(env.action_space.n),
            np.array([
                [ -9, -9, -7, -7 ],
                [ -9, -7, -7, -1 ],
                [ -7, -9, -1, -7 ],
                [  0,  0,  0,  0 ]
            ])
        )
        
    def test_mc_wis_off_policy_small_gridworld(self):
        """env=GridWorld2x2, behavior_policy=RandomPolicy, eval_policy=GridWorld2x2OptimalPolicy"""
        env = GridWorld2x2()
        self._test(
            mc_wis,
            env,
            RandomPolicy(env.action_space.n),
            GridWorld2x2OptimalPolicy(),
            np.array([
                [  0,  0, -2, -2 ],
                [  0,  0,  0, -1 ],
                [  0,  0, -1,  0 ],
                [  0,  0,  0,  0 ]
            ])
        )

    def test_mc_wis_off_policy_small_gridworld_unbalanced(self):
        """env=GridWorld2x2, behavior_policy=UnequalWeightPolicy, eval_policy=RandomPolicy"""
        env = GridWorld2x2()
        self._test(
            mc_wis,
            env,
            RandomPolicy(env.action_space.n),
            RandomPolicy(env.action_space.n, p = [1/6., 1/6., 1/3., 1/3.]),
            np.array([[-8.91827884, -8.86863992, -6.94731186, -6.93853978],
                [-8.91900359, -6.94731294, -7.05168604, -1.        ],
                [-7.13228961, -9.07741789, -1.,         -6.97228902],
                [ 0.,          0.,          0.,          0.        ]]),
            num_episodes=200000
        )

class TestBoth(unittest.TestCase):
    """Both Monte Carlo Methods"""

    def test_wis_same_as_ois_in_on_policy_case(self):
        """Test whether two implementation are equal in on-policy case"""
        env = OneStateMDP()
        behavior_policy = RandomPolicy(env.action_space.n)

        trajs = generate_trajectories(env, behavior_policy, num_episodes=25000)

        # On-policy evaluation test
        Q_ordinary = mc_ois(env.observation_space, env.action_space, trajs,behavior_policy,behavior_policy,np.zeros((env.observation_space.n,env.action_space.n)))
        Q_weighted = mc_wis(env.observation_space, env.action_space, trajs,behavior_policy,behavior_policy,np.zeros((env.observation_space.n,env.action_space.n)))
        
        self.assertTrue(
            np.allclose(Q_ordinary, Q_weighted),
            f'Both implementation should be equal in on policy case. Got ois={Q_ordinary} and wis={Q_weighted}.'
        )
        