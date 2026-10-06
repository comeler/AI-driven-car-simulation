"""
NeuralNetwork: a small feedforward network (input -> hidden -> output),
implemented in plain numpy - no TensorFlow needed. It exposes get_weights()/
set_weights() as one flat vector, which is exactly what the genetic algorithm
needs to mutate and crossover networks without caring about their internal
matrix shapes.
"""

import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, weights=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        if weights is not None:
            self.set_weights(weights)
        else:
            # Small random init - keeps early behavior close to "do nothing much",
            # rather than wildly saturated tanh outputs.
            self.w1 = np.random.randn(input_size, hidden_size) * 0.5
            self.b1 = np.zeros(hidden_size)
            self.w2 = np.random.randn(hidden_size, output_size) * 0.5
            self.b2 = np.zeros(output_size)

    def forward(self, state):
        """One forward pass: state (list of floats) -> raw score per action."""
        x = np.asarray(state, dtype=np.float64)
        hidden = np.tanh(x @ self.w1 + self.b1)   # tanh: bounded, well-behaved hidden activation
        output = hidden @ self.w2 + self.b2       # linear output: one score per action
        return output

    def choose_action(self, state):
        """Pick the action with the highest score (greedy - no exploration
        needed here, since the genetic algorithm explores via mutation, not
        via randomness at decision time)."""
        return int(np.argmax(self.forward(state)))

    def get_weights(self):
        """Flatten every weight/bias into a single 1D vector."""
        return np.concatenate([
            self.w1.flatten(), self.b1.flatten(),
            self.w2.flatten(), self.b2.flatten(),
        ])

    def set_weights(self, flat_weights):
        """Inverse of get_weights(): rebuild w1, b1, w2, b2 from a flat vector."""
        flat = np.asarray(flat_weights, dtype=np.float64)
        i = 0

        n = self.input_size * self.hidden_size
        self.w1 = flat[i:i + n].reshape(self.input_size, self.hidden_size)
        i += n

        n = self.hidden_size
        self.b1 = flat[i:i + n]
        i += n

        n = self.hidden_size * self.output_size
        self.w2 = flat[i:i + n].reshape(self.hidden_size, self.output_size)
        i += n

        n = self.output_size
        self.b2 = flat[i:i + n]
        i += n

    @property
    def num_weights(self):
        return (self.input_size * self.hidden_size + self.hidden_size
                + self.hidden_size * self.output_size + self.output_size)
