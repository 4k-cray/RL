from interfaces.solver import Solver

def eval_episode(solver: Solver) -> float:
    G = 0.0
    state, _ = solver.env.reset()
    done = False

    while not done: 
        action = solver.action(state)
        state, reward, terminated, truncated, _ = solver.env.step(action)
        G += reward
        done = terminated or truncated

    return G