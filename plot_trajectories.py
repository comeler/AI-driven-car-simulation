"""
plot_trajectories.py: overlay several episodes' trajectories on the track,
comparing the trained AI against baseline policies (random, and the best
fixed-turn policy from Etape 3). Static matplotlib plot for after-the-fact
analysis - not a live pygame animation like watch_best.py.
"""

import random

import numpy as np
import matplotlib.pyplot as plt

import config as cfg
from env import CarEnv, ACTION_NAMES
from neural_network import NeuralNetwork


def collect_trajectory(env, policy_fn):
    state = env.reset()
    done = False
    info = {}
    while not done:
        action = policy_fn(state)
        state, reward, done, info = env.step(action)
    return list(env.car.trajectory), info


def main():
    env = CarEnv()
    weights = np.load(cfg.BEST_WEIGHTS_PATH)
    net = NeuralNetwork(env.state_size, cfg.HIDDEN_SIZE, env.action_size, weights=weights)

    policies = {
        "trained AI": lambda s: net.choose_action(s),
        "random": lambda s: random.randrange(env.action_size),
        "fixed 'right'": lambda s: ACTION_NAMES.index("right"),
    }
    colors = {"trained AI": "tab:blue", "random": "tab:gray", "fixed 'right'": "tab:orange"}

    fig, ax = plt.subplots(figsize=(8, 6))

    inner = env.track.inner + [env.track.inner[0]]
    outer = env.track.outer + [env.track.outer[0]]
    ax.plot(*zip(*inner), color="black", linewidth=1)
    ax.plot(*zip(*outer), color="black", linewidth=1)

    for name, policy_fn in policies.items():
        for i in range(cfg.PLOT_EPISODES_PER_POLICY):
            trajectory, info = collect_trajectory(env, policy_fn)
            xs, ys = zip(*trajectory)
            label = f"{name} (laps={info['laps_completed']})" if i == 0 else None
            ax.plot(xs, ys, color=colors[name], alpha=0.7, linewidth=1.5, label=label)

    ax.set_title("Trajectories: trained AI vs baseline policies")
    ax.invert_yaxis()  # match screen coordinates (y grows downward)
    ax.set_aspect("equal")
    ax.legend()
    plt.tight_layout()
    plt.savefig("trajectories.png", dpi=150)
    print("Saved trajectories.png")


if __name__ == "__main__":
    main()
