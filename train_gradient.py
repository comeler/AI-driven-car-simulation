"""
train_gradient.py: trains the network with policy gradient (REINFORCE)
instead of the genetic algorithm. Every setting lives in config.py (the
PG_* values) - edit that file, then run this one (works from a terminal or
Spyder's Run button).

Each update: drive PG_EPISODES_PER_UPDATE episodes with sampled actions,
take one gradient step, then test the GREEDY policy once (how watch_best.py
will drive it). Whenever that greedy episode beats the best so far, the
weights are saved to config.PG_BEST_WEIGHTS_PATH - so stopping with Ctrl-C /
Spyder's stop button never loses the best network found.
"""

import numpy as np

import config as cfg
from env import CarEnv
from neural_network import NeuralNetwork
from policy_gradient import PolicyGradientTrainer, evaluate_greedy


def main():
    if cfg.PG_SEED is not None:
        np.random.seed(cfg.PG_SEED)   # seeds the initial random weights

    env = CarEnv()
    net = NeuralNetwork(env.state_size, cfg.HIDDEN_SIZE, env.action_size)

    loaded = False
    if cfg.PG_START_MODE == "best_weights":
        try:
            net.set_weights(np.load(cfg.PG_BEST_WEIGHTS_PATH))
            loaded = True
            print(f"Loaded starting weights from {cfg.PG_BEST_WEIGHTS_PATH}.")
        except FileNotFoundError:
            print(f"No weights found at {cfg.PG_BEST_WEIGHTS_PATH} - starting fresh instead.")
    elif cfg.PG_START_MODE != "fresh":
        raise ValueError(
            f"Unknown PG_START_MODE '{cfg.PG_START_MODE}' in config.py - use 'fresh' or 'best_weights'."
        )

    if not loaded:
        # Shrink the output layer so the initial policy is close to uniform
        # (every action about equally likely). Large random output weights
        # would make the starting policy nearly deterministic, so the car
        # would barely explore.
        net.w2 = net.w2 * cfg.PG_INIT_OUTPUT_SCALE

    trainer = PolicyGradientTrainer(
        net,
        learning_rate=cfg.PG_LEARNING_RATE,
        gamma=cfg.PG_DISCOUNT,
        entropy_coef=cfg.PG_ENTROPY_COEF,
        grad_clip=cfg.PG_GRAD_CLIP,
        seed=cfg.PG_SEED,             # seeds the sampling of actions
    )

    best_reward, best_laps, _ = evaluate_greedy(env, net)
    np.save(cfg.PG_BEST_WEIGHTS_PATH, net.get_weights())
    print(f"Starting point: greedy reward = {best_reward:8.2f}, laps = {best_laps}")
    print(f"(best weights are saved to {cfg.PG_BEST_WEIGHTS_PATH} whenever the greedy reward improves)\n")

    for update in range(1, cfg.PG_NUM_UPDATES + 1):
        stats = trainer.update(env, cfg.PG_EPISODES_PER_UPDATE)

        greedy_note = ""
        if update % cfg.PG_EVAL_EVERY == 0:
            greedy_reward, greedy_laps, greedy_steps = evaluate_greedy(env, net)
            if greedy_reward > best_reward:
                best_reward, best_laps = greedy_reward, greedy_laps
                np.save(cfg.PG_BEST_WEIGHTS_PATH, net.get_weights())
                greedy_note = "  <- new best saved"
            greedy_note = (f" | greedy={greedy_reward:8.2f} (laps {greedy_laps}, "
                           f"{greedy_steps} steps){greedy_note}")

        print(f"Update {update:4d} | sampled mean reward={stats['mean_reward']:8.2f} | "
              f"mean laps={stats['mean_laps']:4.2f} | mean length={stats['mean_length']:6.1f} | "
              f"|grad|={stats['grad_norm']:.3f}{greedy_note}")

    print(f"\nDone. Best greedy reward = {best_reward:.2f} ({best_laps} laps), "
          f"saved in {cfg.PG_BEST_WEIGHTS_PATH}.")


if __name__ == "__main__":
    main()
