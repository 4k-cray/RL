################################
# Hide warnings about wrappers because they are annoying and not directly
# caused by our code.
import warnings
warnings.showwarning = lambda *args, **kwargs: None
################################

import argparse
import numpy as np
import gymnasium as gym
from typing import Literal
from lib.envs.register_custom_grid_worlds import register_custom_grid_worlds
from interfaces.random_policy import RandomPolicy
from lib.envs.grid_world_2x2 import GridWorld2x2OptimalPolicy


MONTE_CARLO_POLICY_NAMES = ['equiprobable', 'unbalanced', 'ideal', 'down_or_right']

def main():
    #########################################################################
    # Create the top-level parser & setup subparsers for each assignment
    parser = argparse.ArgumentParser(description="Visualize your algorithms in different environments and with different configurations.")
    subparsers = parser.add_subparsers(dest='assignment', required=True)
    #########################################################################


    #########################################################################
    # Bandits
    parser_bandits = subparsers.add_parser('bandit', help="Bandits w/ 10-armed testbed")
    parser_bandits.add_argument('--num_iterations', type=int, required=False, default=10000, help="Number of iterations to run")
    parser_bandits.add_argument('--num_simulations', type=int, required=False, default=300, help="Number of simulations to run")
    parser_bandits.add_argument('--k', type=int, required=False, default=10, help="Number of arms in the bandit")
    parser_bandits.set_defaults(func=action_for_bandits)
    #########################################################################

    #########################################################################
    # Dynamic Programming
    register_custom_grid_worlds()
    environment_names = [
        'OneStateMDP-v0',
        'GridWorld2x2-v0',
        'WrappedFrozenLake-v0',
        'WrappedFrozenLakeSlippery-v0',
        'WrappedTaxi-v0',
        'WrappedCliffWalking-v0'
    ]
    
    parser_dp = subparsers.add_parser('dp', help="Value Iteration and Value Prediction - solving an MDP")
    parser_dp.add_argument('--environment', type=str, required=False, default='GridWorld2x2-v0', choices=environment_names, help="Environment to visualize")
    parser_dp.set_defaults(func=action_for_dp)

    parser_monte_carlo = subparsers.add_parser('monte_carlo', help="Monte-Carlo importance sampling - off-policy evaluation")
    parser_monte_carlo.add_argument('--environment', type=str, required=False, default='GridWorld2x2-v0', choices=environment_names, help="Environment to visualize")
    parser_monte_carlo.add_argument('--num_episodes', type=int, required=False, default=10000, help="Number of episodes/trajectories to generate")
    parser_monte_carlo.add_argument('--target_policy', type=str, required=False, choices=MONTE_CARLO_POLICY_NAMES, default='equiprobable', help="Target policy to evaluate")
    parser_monte_carlo.add_argument('--behavior_policy', type=str, required=False, choices=MONTE_CARLO_POLICY_NAMES, default='equiprobable', help="Behavior policy to generate trajectories")
    parser_monte_carlo.set_defaults(func=action_for_monte_carlo)
    
    parser_n_step_bootstrap = subparsers.add_parser('n_step_bootstrap', help="On-policy n-step TD and off-policy n-step SARSA")
    parser_n_step_bootstrap.add_argument('--environment', type=str, required=False, default='GridWorld2x2-v0', choices=environment_names, help="Environment to visualize")
    parser_n_step_bootstrap.add_argument('--num_episodes', type=int, required=False, default=None, help="Number of episodes/trajectories to generate")
    parser_n_step_bootstrap.add_argument('--n', type=int, required=False, default=None, help="n-Value (number of steps before bootstrapping)")
    parser_n_step_bootstrap.add_argument('--alpha_td', type=float, required=False, default=0.005, help="Learning rate for TD")
    parser_n_step_bootstrap.add_argument('--alpha_sarsa', type=float, required=False, default=None, help="Learning rate for SARSA")
    parser_n_step_bootstrap.add_argument('--gamma', type=float, required=False, default=None, help="Discount factor")
    parser_n_step_bootstrap.set_defaults(func=action_for_n_step_bootstrap)
    #########################################################################

    #########################################################################
    # Parse the arguments, call the function
    args = parser.parse_args()
    args.func(args)
    #########################################################################

def action_for_bandits(args):
    from lib.visualize.bandit import visualize
    visualize(
        num_simulations = args.num_simulations,
        num_iterations = args.num_iterations,
        k = args.k
    )

def action_for_dp(args):
    from lib.visualize.dp import visualize
    visualize(args.environment)


def action_for_monte_carlo(args):
    from lib.visualize.monte_carlo import visualize
    env = gym.make(args.environment)

    def get_policy_for_env(env_name: str, env: gym.Env, policy_name: Literal['equiprobable', 'unbalanced', 'ideal', 'down_or_right']):
        def unequal_weights(n):
            weights = np.arange(1, n+1)
            return weights / weights.sum() # Normalize so that they sum to 1

        policies = {
            'equiprobable': RandomPolicy(env.action_space.n),
            'unbalanced': RandomPolicy(env.action_space.n, unequal_weights(env.action_space.n))
        }

        if env_name == "GridWorld2x2-v0":
            policies['ideal'] = GridWorld2x2OptimalPolicy()
            policies['down_or_right'] = RandomPolicy(env.action_space.n, [0, 0, 0.5, 0.5])

        if policy_name not in policies.keys():
            raise ValueError(f"Policy {policy_name} not found for environment {env_name}. Try one of: {', '.join(policies.keys())}")

        return policies[policy_name]

    target_policy = get_policy_for_env(args.environment, env, args.target_policy)
    behavior_policy = get_policy_for_env(args.environment, env, args.behavior_policy)

    visualize(
        env = env,
        num_episodes=args.num_episodes,
        target_policy=target_policy,
        behavior_policy=behavior_policy
    )

def action_for_n_step_bootstrap(args):
    from lib.visualize.n_step_bootstrap import visualize
    behavior_policy = RandomPolicy(gym.make(args.environment).action_space.n)

    from assignments.n_step_bootstrap import NStepSARSAHyperparameters

    hyperparameters = {
        'GridWorld2x2-v0': NStepSARSAHyperparameters(
            gamma = 0.99,
            alpha = 0.01,
            n = 1
        ),
        'WrappedFrozenLake-v0': NStepSARSAHyperparameters(
            gamma = 0.99,
            alpha = 0.005,
            n = 1
        ),
        'WrappedFrozenLakeSlippery-v0': NStepSARSAHyperparameters(
            gamma = 0.99,
            alpha = 0.005,
            n = 1
        ),
        'WrappedTaxi-v0': NStepSARSAHyperparameters(
            gamma = 0.99,
            alpha = 0.005,
            n = 1
        )
    }[args.environment]

    num_episodes = args.num_episodes or (
        {
            'WrappedTaxi-v0': 25000,
            'WrappedFrozenLake-v0': 10000,
            'WrappedFrozenLakeSlippery-v0': 75000,
            'GridWorld2x2-v0': 10000
        }[args.environment]
    )

    hyperparameters.gamma = args.gamma or hyperparameters.gamma
    hyperparameters.alpha = args.alpha_sarsa or hyperparameters.alpha
    hyperparameters.n = args.n or hyperparameters.n

    visualize(
        env_name = args.environment,
        behavior_policy=behavior_policy,
        num_episodes=num_episodes,
        alpha_td=args.alpha_td,
        hyperparameters=hyperparameters,
        save_video = args.save_video
    )


if __name__ == "__main__":
    main()
