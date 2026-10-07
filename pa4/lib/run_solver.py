from typing import Type, Literal
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from datetime import datetime
from interfaces.solver import Solver, Hyperparameters
from lib.envs.capture_video_wrapper import CaptureVideoWrapper
from lib.eval_episode import eval_episode

def run_solver(
    env_name: Literal["LunarLander-v2", "CartPole-v1", "MountainCar-v0", "FrozenLake-v1", "Taxi-v3", "CliffWalking-v0", "CartPoleDQN-v0"],
    solver: Type[Solver],
    hyperparameters: Hyperparameters,
    num_episodes: int,
    render: bool = False,
    save_video: bool = False,
    post_solve_fn: callable = None,
    post_episode_fn: callable = None,
    success_condition: callable = None,
    env = None
):
    train_env = CaptureVideoWrapper(gym.make(env_name, render_mode='rgb_array'), None)
    render_env = gym.make(env_name, render_mode='human')

    solver = solver(train_env, hyperparameters)

    print("Running a solver...")
    print("Environment:", env_name)
    print("Solver:", solver.name)
    print("Max episodes:", num_episodes)
    print("Hyperparameters:", hyperparameters.__dict__)

    CAPTURE_EPISODES = [
        1,
        num_episodes // 5,
        2 * num_episodes // 5,
        3 * num_episodes // 5,
        4 * num_episodes // 5,
        num_episodes
    ]

    for episode in tqdm(range(num_episodes)):
        if save_video and ((episode+1) in CAPTURE_EPISODES):
            train_env.get_wrapper_attr('activate')()

        G = solver.train_episode()

        if post_episode_fn:
            post_episode_fn(solver)

        if save_video and ((episode+1) in CAPTURE_EPISODES):
            success = success_condition(G) if success_condition else None
            train_env.get_wrapper_attr('save_frames_to_tensorboard')(episode+1, fps=30, success=success)

    if post_solve_fn:
        post_solve_fn(solver)

    Gs = [eval_episode(solver) for _ in tqdm(range(100))]

    if render:
        solver.env = render_env
        eval_episode(solver)

    result = {
        'mean': np.mean(Gs),
        'median': np.median(Gs),
        'max': np.max(Gs),
        'min': np.min(Gs)
    }

    print(f'Mean: {result["mean"]}')
    print(f'Median: {result["median"]}')
    print(f'Max: {result["max"]}')
    print(f'Min: {result["min"]}')

    print()
    if save_video:
        filename = f'{solver.name}_{env_name}.mp4'
        train_env.get_wrapper_attr('save_to_video')(filename, fps=30)
        print(f"Video saved to {filename}")

    result['solver'] = solver

    return result