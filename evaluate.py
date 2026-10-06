"""
evaluate.py: run a trained network through many episodes (no rendering) and
report aggregate metrics. A single watch_best.py run tells you almost
nothing about consistency - this tells you how it performs on average, and
how often/why it fails.
"""

import numpy as np

import config as cfg
from env import CarEnv
from neural_network import NeuralNetwork


def run_episode(env, net):
    state = env.reset()
    total_reward = 0.0
    steps = 0
    done = False
    info = {}
    while not done:
        action = net.choose_action(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
    return total_reward, steps, info


def main():
    weights = np.load(cfg.BEST_WEIGHTS_PATH)
    env = CarEnv()
    net = NeuralNetwork(env.state_size, cfg.HIDDEN_SIZE, env.action_size, weights=weights)

    rewards, steps_list, laps_list = [], [], []
    off_track_count = 0

    for _ in range(cfg.EVAL_NUM_EPISODES):
        total_reward, steps, info = run_episode(env, net)
        rewards.append(total_reward)
        steps_list.append(steps)
        laps_list.append(info["laps_completed"])
        if not info["on_track"]:
            off_track_count += 1

    timeout_count = cfg.EVAL_NUM_EPISODES - off_track_count
    finished_at_least_1_lap = sum(1 for laps in laps_list if laps >= 1)

    print(f"Evaluated over {cfg.EVAL_NUM_EPISODES} episodes ({cfg.BEST_WEIGHTS_PATH}):\n")
    print(f"  Avg reward:            {np.mean(rewards):8.2f}  (min {np.min(rewards):7.2f}, max {np.max(rewards):7.2f})")
    print(f"  Avg episode length:    {np.mean(steps_list):8.1f} steps  (~{np.mean(steps_list) * env.dt:.1f}s)")
    print(f"  Episodes with >=1 lap: {finished_at_least_1_lap}/{cfg.EVAL_NUM_EPISODES}")
    print(f"  Avg laps completed:    {np.mean(laps_list):8.2f}")
    print(f"  Ended off-track:       {off_track_count:3d}/{cfg.EVAL_NUM_EPISODES}  ({100 * off_track_count / cfg.EVAL_NUM_EPISODES:.0f}%)")
    print(f"  Ended by timeout:      {timeout_count:3d}/{cfg.EVAL_NUM_EPISODES}  ({100 * timeout_count / cfg.EVAL_NUM_EPISODES:.0f}%)")


if __name__ == "__main__":
    main()
