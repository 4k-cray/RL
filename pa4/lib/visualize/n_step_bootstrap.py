import numpy as np
import gymnasium as gym

from interfaces.policy import Policy
from lib.envs.wrapped_gridworld import WrappedGridWorld
from lib.envs.generate_trajectories import generate_trajectories
from lib.Q2V import Q2V

from lib.run_solver import run_solver
from assignments.n_step_bootstrap import NStepSARSA, NStepSARSAHyperparameters
from assignments.n_step_bootstrap import on_policy_n_step_td as ntd

def visualize(
    env_name: str,
    behavior_policy: Policy,
    num_episodes: int,
    alpha_td: float,
    hyperparameters: NStepSARSAHyperparameters,
    save_video: bool = False
):
    env = gym.make(env_name)

    print(f"Generating episodes based on {behavior_policy}")
    trajs = generate_trajectories(env, behavior_policy, num_episodes)
    print("Done!")
    print()
    print("---------------------------------------")
    print(env)
    print("---------------------------------------")
    print()

    print(f"---------------------------------------")
    print(f"     {hyperparameters.n}-step TD and SARSA")
    print()

    # n-step TD
    V = ntd(trajs, hyperparameters.n, alpha_td, np.zeros((env.observation_space.n)), gamma=1.0)
    print(env.visualize_v(V, f"V - On-Policy TD"))
    print()

    # Off-policy SARSA
    result = run_solver(
        env_name = env_name,
        solver = NStepSARSA,
        hyperparameters = hyperparameters,
        num_episodes = num_episodes,
        render = True,
        save_video=save_video
    )
    solver: NStepSARSA = result['solver']

    print(env.visualize_v(Q2V(env, solver.pi.Q, solver.pi), f"V - Off-Policy SARSA"))
    print()
    print(env.visualize_q(solver.pi.Q, f"Q - Off-Policy SARSA"))
    print()
    print(env.visualize_policy(solver.pi, "π - Off-Policy SARSA"))
    print(f"---------------------------------------")
    print()
