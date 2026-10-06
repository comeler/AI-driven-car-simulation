"""
train.py: runs the genetic algorithm against CarEnv. Every setting lives in
config.py - edit that file, then just run this one (works identically from a
terminal or by pressing Run in Spyder - no command-line flags needed).

Two files are written as training progresses:
  - config.BEST_WEIGHTS_PATH  - the single best-ever genome (what
    watch_best.py / evaluate.py / plot_trajectories.py load).
  - config.HOF_PATH / HOF_SUMMARY_PATH - the top config.HOF_SIZE genomes
    ever seen (reloadable archive + human-readable text summary).

To pause a long run: just stop the script (Ctrl-C or Spyder's stop button).
The Hall of Fame is saved every generation, so nothing since the last
printed line is lost. To continue later, set START_MODE = "hall_of_fame" in
config.py and run train.py again.
"""

import numpy as np

import config as cfg
from env import CarEnv
from genetic_algorithm import Population, HallOfFame
from neural_network import NeuralNetwork


def seed_population_from_hof(pop, hof):
    """Fill a freshly created Population using the Hall of Fame: the HOF
    genomes themselves (unmutated), then mutated copies of them to fill out
    the rest of pop.pop_size. Used for START_MODE = 'hall_of_fame'."""
    num_weights = NeuralNetwork(pop.input_size, pop.hidden_size, pop.output_size).num_weights
    seeded = np.zeros((pop.pop_size, num_weights))
    for i in range(pop.pop_size):
        base = np.array(hof.weights[i % len(hof.weights)])
        if i < len(hof.weights):
            seeded[i] = base  # keep the HOF genomes themselves unmutated
        else:
            noise = np.random.randn(num_weights) * pop.mutation_strength
            seeded[i] = base + noise
    pop.load_all_weights(seeded)


def main():
    env = CarEnv()
    hof = HallOfFame(size=cfg.HOF_SIZE)
    pop = Population(
        cfg.POP_SIZE, env.state_size, cfg.HIDDEN_SIZE, env.action_size,
        mutation_rate=cfg.MUTATION_RATE, mutation_strength=cfg.MUTATION_STRENGTH,
        elite_fraction=cfg.ELITE_FRACTION,
    )

    start_gen = 0

    if cfg.START_MODE == "hall_of_fame":
        try:
            hof.load(cfg.HOF_PATH)
            print(f"Loaded Hall of Fame from {cfg.HOF_PATH}: {len(hof.weights)} genomes, "
                  f"best fitness so far = {hof.best_fitness:.2f}")
            seed_population_from_hof(pop, hof)
            start_gen = max(hof.generations) + 1
        except FileNotFoundError:
            print(f"No Hall of Fame found at {cfg.HOF_PATH} - starting fresh instead.")
    elif cfg.START_MODE != "fresh":
        raise ValueError(
            f"Unknown START_MODE '{cfg.START_MODE}' in config.py - use 'fresh' or 'hall_of_fame'."
        )

    for gen in range(start_gen, start_gen + cfg.NUM_GENERATIONS):
        fitnesses = pop.evaluate(env)
        avg_fit = float(np.mean(fitnesses))
        _, best_fit_this_gen = pop.best_network()

        hof.consider(pop.get_all_weights(), fitnesses, generation=gen)
        hof.save(cfg.HOF_PATH, cfg.HOF_SUMMARY_PATH)
        np.save(cfg.BEST_WEIGHTS_PATH, hof.best_weights)

        print(f"Gen {gen:3d} | best={best_fit_this_gen:8.2f} | avg={avg_fit:8.2f} | "
              f"all-time best={hof.best_fitness:8.2f}")

        pop.evolve(hall_of_fame=hof)

    print(f"\nDone. Best-ever genome saved to {cfg.BEST_WEIGHTS_PATH} (fitness={hof.best_fitness:.2f}).")
    print(f"Hall of Fame: {cfg.HOF_PATH} (reloadable) / {cfg.HOF_SUMMARY_PATH} (readable).")


if __name__ == "__main__":
    main()
