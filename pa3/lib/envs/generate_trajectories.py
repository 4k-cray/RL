import gymnasium as gym
from tqdm import tqdm
from interfaces.policy import Policy

# Gather experience using behavior policy
def generate_trajectories(env: gym.Env, behavior_policy: Policy, num_episodes = 100000):

    trajs = []
    for _ in tqdm(range(num_episodes)):
        s_0, _ = env.reset()
        done = terminated = False
        states, actions, rewards =\
            [s_0], [], []

        while not done and not terminated:
            a = behavior_policy.action(states[-1])
            s, r, done, terminated, _ = env.step(a)

            states.append(s)
            actions.append(a)
            rewards.append(r)

        traj = list(zip(states[:-1],actions,rewards,states[1:]))
        trajs.append(traj)

    return trajs