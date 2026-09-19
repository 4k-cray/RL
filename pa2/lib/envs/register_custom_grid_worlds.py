import gymnasium as gym
import numpy as np

from lib.envs.wrapped_gridworld import WrappedGridWorld, WrappedMDP, Actions
from lib.envs.grid_world_2x2 import GridWorld2x2
from lib.envs.one_state_mdp import OneStateMDP

##### Register overrides for the gym classes
def create_wrapped_cliff_walking(render_mode = None):
    return WrappedGridWorld(
        env= gym.make('CliffWalking-v0', render_mode=render_mode),
        rows=4,
        columns=12,
        action_remap = [
            Actions.UP,
            Actions.RIGHT,
            Actions.DOWN,
            Actions.LEFT
        ],
        gamma=0.01
    )

def create_wrapped_taxi(render_mode = None):
    return WrappedGridWorld(
        env=gym.make("Taxi-v3", render_mode=render_mode),
        rows=5,
        columns=100,
        action_remap = [
            Actions.DOWN,
            Actions.UP,
            Actions.RIGHT,
            Actions.LEFT
        ],
        gamma=0.99
    )

def create_wrapped_frozen_lake(render_mode = None):
    return WrappedGridWorld(
        env=gym.make("FrozenLake-v1", is_slippery=False, render_mode=render_mode),
        rows=4,
        columns=4,
        action_remap = [
            Actions.LEFT,
            Actions.DOWN,
            Actions.RIGHT,
            Actions.UP
        ],
        gamma=0.99
    )

def create_wrapped_frozen_lake_slippery(render_mode = None):
    return WrappedGridWorld(
        env=gym.make("FrozenLake-v1", render_mode=render_mode),
        rows=4,
        columns=4,
        action_remap = [
            Actions.LEFT,
            Actions.DOWN,
            Actions.RIGHT,
            Actions.UP
        ],
        gamma=0.99
    )

def create_wrapped_grid_world_2x2(render_mode = None):
    return  WrappedGridWorld(GridWorld2x2(), rows=2, columns=2, gamma=1.0)

def create_wrapped_one_state_mdp(render_mode = None):
    return WrappedMDP(OneStateMDP(), shape=np.array((1, 2)), gamma=1.0)

def register_custom_grid_worlds():
    gym.register(
        id='WrappedCliffWalking-v0',  # Unique ID for your environment
        entry_point=create_wrapped_cliff_walking
    )

    gym.register(
        id='WrappedTaxi-v0',  # Unique ID for your environment
        entry_point=create_wrapped_taxi
    )

    gym.register(
        id='WrappedFrozenLakeSlippery-v0',  # Unique ID for your environment
        entry_point=create_wrapped_frozen_lake_slippery
    )

    gym.register(
        id = 'WrappedFrozenLake-v0',
        entry_point = create_wrapped_frozen_lake
    )

    gym.register(
        id = 'GridWorld2x2-v0',
        entry_point = create_wrapped_grid_world_2x2
    )

    gym.register(
        id = 'OneStateMDP-v0',
        entry_point = create_wrapped_one_state_mdp
    )