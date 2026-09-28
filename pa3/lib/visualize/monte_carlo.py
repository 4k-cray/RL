import numpy as np

from interfaces.policy import Policy
from lib.envs.wrapped_gridworld import WrappedGridWorld
from lib.envs.generate_trajectories import generate_trajectories
from lib.Q2V import Q2V

from assignments.monte_carlo import off_policy_mc_prediction_ordinary_importance_sampling as mc_ois
from assignments.monte_carlo import off_policy_mc_prediction_weighted_importance_sampling as mc_wis

def visualize(
    env: WrappedGridWorld,
    num_episodes: int,
    target_policy: Policy,
    behavior_policy: Policy
):
    print("---------------------------------------")
    print(env)
    print("---------------------------------------")
    print()

    # Sample with random policy
    print("Generating episodes based on random policy")
    trajs = generate_trajectories(env, behavior_policy, num_episodes)
    print("Done!")
    print()

    Q_ordinary = mc_ois(env.observation_space, env.action_space, trajs, behavior_policy, target_policy, np.zeros((env.observation_space.n, env.action_space.n)))
    Q_weighted = mc_wis(env.observation_space, env.action_space, trajs, behavior_policy, target_policy, np.zeros((env.observation_space.n, env.action_space.n)))
    
    print("Off-policy Monte-Carlo evaluation")
    print(f"  Behavior Policy: {behavior_policy}")
    print(f"  Target Policy: {target_policy}")
    print("---------------------------------------")
    print(env.visualize_v(Q2V(env, Q_ordinary, target_policy), "V - ordinary importance sampling"))
    print(env.visualize_q(Q_ordinary, "Q - ordinary importance sampling"))
    print()

    print(env.visualize_v(Q2V(env, Q_weighted, target_policy), "V - weighted importance sampling"))
    print(env.visualize_q(Q_weighted, "Q - weighted importance sampling"))
    print()