import numpy as np
import gymnasium as gym

from interfaces.random_policy import RandomPolicy
from lib.envs.wrapped_gridworld import WrappedGridWorld

from assignments.dp import value_iteration, value_prediction

def visualize(env_name: str):
    env: WrappedGridWorld = gym.make(env_name)

    too_big_to_visualize_envs = [
        "WrappedTaxi-v0",
        "WrappedCliffWalking-v0"
    ]

    behavior_policy = RandomPolicy(env.action_space.n)
    initV = np.zeros(env.observation_space.n)

    if not env_name in too_big_to_visualize_envs:
        print("---------------------------------------")
        print(env)
        print("---------------------------------------")
        print()

    V, Q = value_prediction(env, behavior_policy, initV, 1e-12, env.gamma)
    if not env_name in too_big_to_visualize_envs:
        print(f"Value Prediction (using {behavior_policy.__class__.__name__})")
        print("---------------------------------------")
        print()
        print(env.get_wrapper_attr('visualize_v')(V, "V_π"))
        print()
        print(env.get_wrapper_attr('visualize_q')(Q, "Q_π"))
        print("---------------------------------------")
        print()
    else:
        print(f"Skipping value_prediction() visualization of {env_name} due to size.")

    initV = np.zeros(env.observation_space.n)
    V, pi = value_iteration(env, initV, 1e-12, gamma=env.gamma)

    if not env_name in too_big_to_visualize_envs:
        print("Value Iteration")
        print("---------------------------------------")
        print()
        print(env.get_wrapper_attr('visualize_v')(V, "V*"))
        print()
        print(env.get_wrapper_attr('visualize_policy')(pi, "π*"))
        print("---------------------------------------")
    else:
        print(f"Skipping value_iteration() visualization of {env_name} due to size.")

    ## Single episode of the ideal policy
    env = gym.make(env_name, render_mode='human')
    state, _ = env.reset()
    env.render()
    done = terminated = False
    G = 0.0
    while not done and not terminated:
        action = pi.action(state)
        state, R, done, terminated, _ = env.step(action)
        G += R
    print("Received reward:", G)
