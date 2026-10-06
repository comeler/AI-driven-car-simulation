"""
Population: a genetic algorithm operating on a population of NeuralNetwork
individuals. No gradients anywhere - "training" means evaluate everyone's
fitness (total reward from one episode), keep the best, breed the next
generation from them with crossover + mutation, and repeat.
"""

import numpy as np

from neural_network import NeuralNetwork


class Population:
    def __init__(self, pop_size, input_size, hidden_size, output_size,
                 mutation_rate=0.15, mutation_strength=0.4, elite_fraction=0.2):
        self.pop_size = pop_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.mutation_rate = mutation_rate         # probability each individual weight gets mutated
        self.mutation_strength = mutation_strength  # size of the random nudge when it does
        self.elite_fraction = elite_fraction        # fraction of top performers kept as-is + used as parents

        self.networks = [
            NeuralNetwork(input_size, hidden_size, output_size) for _ in range(pop_size)
        ]
        self.fitnesses = [0.0] * pop_size

    def evaluate(self, env):
        """Run every individual through one full episode, record its total
        reward as fitness. This is the slow part - one episode per network."""
        for i, net in enumerate(self.networks):
            state = env.reset()
            total_reward = 0.0
            done = False
            while not done:
                action = net.choose_action(state)
                state, reward, done, info = env.step(action)
                total_reward += reward
            self.fitnesses[i] = total_reward
        return self.fitnesses

    def best_network(self):
        best_idx = int(np.argmax(self.fitnesses))
        return self.networks[best_idx], self.fitnesses[best_idx]

    def get_all_weights(self):
        """Every individual's flat weight vector, stacked into one 2D array -
        used to checkpoint the whole population, not just the best one."""
        return np.array([net.get_weights() for net in self.networks])

    def load_all_weights(self, weights_array):
        """Inverse of get_all_weights(): restore every individual from a
        previously saved population checkpoint."""
        for net, w in zip(self.networks, weights_array):
            net.set_weights(w)

    def evolve(self, hall_of_fame=None):
        """Build the next generation. If hall_of_fame is given, elites and
        breeding parents are drawn from that all-time-best archive instead
        of just this generation's top performers - a great genome found many
        generations ago is never lost to genetic drift, and mediocre luck in
        one generation can't erase good genomes found earlier."""
        num_elite = max(1, int(self.pop_size * self.elite_fraction))

        if hall_of_fame is not None and len(hall_of_fame.weights) > 0:
            elite_weights = [np.array(w) for w in hall_of_fame.weights[:num_elite]]
        else:
            order = np.argsort(self.fitnesses)[::-1]  # best fitness first, this generation only
            elite_weights = [self.networks[i].get_weights() for i in order[:num_elite]]

        new_networks = []

        # Elitism: carry the best performers over unchanged, so a generation
        # can never do worse than the best genome known so far.
        for w in elite_weights:
            net = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
            net.set_weights(w.copy())
            new_networks.append(net)

        # Fill the rest of the population with children of the elites.
        while len(new_networks) < self.pop_size:
            parent_a = elite_weights[np.random.randint(len(elite_weights))]
            parent_b = elite_weights[np.random.randint(len(elite_weights))]
            child = self._crossover(parent_a, parent_b)
            child = self._mutate(child)
            net = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
            net.set_weights(child)
            new_networks.append(net)

        self.networks = new_networks
        self.fitnesses = [0.0] * self.pop_size

    def _crossover(self, weights_a, weights_b):
        """Uniform crossover: each weight independently comes from parent A or B."""
        mask = np.random.rand(len(weights_a)) < 0.5
        return np.where(mask, weights_a, weights_b)

    def _mutate(self, weights):
        """Add gaussian noise to a random subset of weights."""
        mutate_mask = np.random.rand(len(weights)) < self.mutation_rate
        noise = np.random.randn(len(weights)) * self.mutation_strength
        return weights + mutate_mask * noise


class HallOfFame:
    """Persistent archive of the best individuals EVER seen, across all of
    training - not just whichever generation happens to be running now. Two
    jobs: (1) supply Population.evolve() with all-time-best parents instead
    of just the current generation's top performers, and (2) let you stop
    and later restart training from the best genomes found so far, without
    a full population checkpoint or any command-line flags - just set
    START_MODE in config.py and run train.py again."""

    def __init__(self, size=20):
        self.size = size
        self.weights = []       # list of flat weight vectors, best first
        self.fitnesses = []     # matching fitness values
        self.generations = []   # which generation each entry was found in

    def consider(self, weights_list, fitnesses_list, generation):
        """Merge one generation's individuals into the archive, keeping only
        the overall top `size` BY FITNESS AND DISTINCT genome. Elitism keeps
        the same best genome unchanged across generations, so without
        de-duplication the same handful of genomes would fill most of the
        20 slots instead of 20 genuinely different ones."""
        combined = list(zip(self.fitnesses, self.weights, self.generations))
        combined += [(float(f), np.array(w), generation) for f, w in zip(fitnesses_list, weights_list)]
        combined.sort(key=lambda item: item[0], reverse=True)

        deduped = []
        for fit, weights, gen in combined:
            is_duplicate = any(np.allclose(weights, kept_w, atol=1e-9) for _, kept_w, _ in deduped)
            if not is_duplicate:
                deduped.append((fit, weights, gen))
            if len(deduped) >= self.size:
                break

        self.fitnesses = [item[0] for item in deduped]
        self.weights = [item[1] for item in deduped]
        self.generations = [item[2] for item in deduped]

    @property
    def best_fitness(self):
        return self.fitnesses[0] if self.fitnesses else -np.inf

    @property
    def best_weights(self):
        return self.weights[0] if self.weights else None

    def save(self, weights_path, summary_path=None):
        """Save the archive (reloadable .npz), and optionally a plain-text
        summary meant for you to read, not the program."""
        np.savez(
            weights_path,
            weights=np.array(self.weights),
            fitnesses=np.array(self.fitnesses),
            generations=np.array(self.generations),
        )
        if summary_path:
            with open(summary_path, "w") as f:
                f.write(f"Hall of Fame - top {len(self.weights)} genomes ever seen\n\n")
                f.write(f"{'rank':>4} | {'fitness':>10} | {'found at generation':>20}\n")
                for i, (fit, gen) in enumerate(zip(self.fitnesses, self.generations)):
                    f.write(f"{i + 1:4d} | {fit:10.2f} | {gen:20d}\n")

    def load(self, weights_path):
        """Raises FileNotFoundError if weights_path doesn't exist yet - the
        caller decides whether to fall back to a fresh start."""
        data = np.load(weights_path)
        self.weights = list(data["weights"])
        self.fitnesses = list(data["fitnesses"])
        self.generations = list(data["generations"])
