import numpy as np
import unittest

from interfaces.random_policy import RandomPolicy
from lib.envs.one_state_mdp import OneStateMDP
from lib.envs.generate_trajectories import generate_trajectories
from lib.run_solver import run_solver

from assignments.n_step_bootstrap import NStepSARSA, NStepSARSAHyperparameters
from assignments.n_step_bootstrap import on_policy_n_step_td as ntd

class TestNStepTD(unittest.TestCase):
    """on_policy_n_step_td()"""
    def test_ntd(self):
        """Basic Test w/ OneStateMDP"""
        env = OneStateMDP()
        behavior_policy = RandomPolicy(env.action_space.n)
        trajs = generate_trajectories(env, behavior_policy)

        V_est_td = ntd(trajs,2,0.005,np.zeros((env.observation_space.n)), gamma=1.0)
        assert np.allclose(V_est_td,np.array([0.1,0.]),1e-5,1e-1), 'due to stochasticity, this test might fail'

class TestNStepSARSA(unittest.TestCase):
    """off_policy_n_step_sarsa()"""
    def test_nsarsa_frozen_lake(self):
        """FrozenLake-v1"""
        results = run_solver(
            env_name = "FrozenLake-v1",
            solver = NStepSARSA,
            hyperparameters = NStepSARSAHyperparameters(
                gamma = 0.99,
                alpha = 0.005,
                n = 1
            ),
            num_episodes = 75000,
            render = False
        )

        self.assertTrue(results['mean'] > 0.70, f"Expected mean reward > 0.70, got {results['mean']}")

    def test_nsarsa_cliff_walking(self):
        """Taxi-v3"""
        results = run_solver(
            env_name = "Taxi-v3",
            solver = NStepSARSA,
            hyperparameters = NStepSARSAHyperparameters(
                gamma = 0.99,
                alpha = 0.005,
                n = 1
            ),
            num_episodes = 25000,
            render = False
        )

        self.assertTrue(results['mean'] >= 6.0, f"Expected mean reward > 0, got {results['mean']}")
