"""
NeuralNetwork: a small feedforward network (input -> hidden -> output),
implemented in plain numpy - no TensorFlow/PyTorch needed.

Two ways to use it:
  - Genetic algorithm: get_weights()/set_weights() expose every weight as one
    flat vector, which is what mutation/crossover work on.
  - Policy gradient (REINFORCE): the output layer is read as action
    PROBABILITIES (softmax), and policy_gradient() backpropagates through the
    whole network to get the gradient of the loss with respect to every weight.

The weight layout (w1, b1, w2, b2, flattened in that order) is identical for
both, so a weights file saved by one method can be loaded by the other.
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
            self.w1 = np.random.randn(input_size, hidden_size) * 0.5
            self.b1 = np.zeros(hidden_size)
            self.w2 = np.random.randn(hidden_size, output_size) * 0.5
            self.b2 = np.zeros(output_size)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, state):
        """State (or a batch of states) -> raw score (logit) per action."""
        x = np.asarray(state, dtype=np.float64)
        hidden = np.tanh(x @ self.w1 + self.b1)
        return hidden @ self.w2 + self.b2

    def choose_action(self, state):
        """Greedy: the action with the highest score. Used for evaluation and
        for watching the trained car (no randomness)."""
        return int(np.argmax(self.forward(state)))

    @staticmethod
    def _softmax(logits):
        shifted = logits - np.max(logits, axis=-1, keepdims=True)  # numerical stability
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def action_probabilities(self, state):
        """Softmax of the logits: the probability the policy gives each action."""
        return self._softmax(self.forward(state))

    def sample_action(self, state, rng):
        """Stochastic: pick an action at random according to its probability.
        Used while TRAINING with policy gradient, so the car explores."""
        probs = self.action_probabilities(state)
        return int(rng.choice(len(probs), p=probs))

    # ------------------------------------------------------------------
    # Backpropagation (policy gradient)
    # ------------------------------------------------------------------
    def policy_gradient(self, states, actions, advantages, entropy_coef=0.0):
        """Gradient of the REINFORCE loss with respect to every weight, as one
        flat vector in the same order as get_weights().

            loss = -mean( advantage_t * log pi(action_t | state_t) )
                   - entropy_coef * mean( entropy of pi(. | state_t) )

        Gradient DESCENT on this loss makes actions with positive advantage
        more likely and actions with negative advantage less likely; the
        entropy term rewards staying uncertain, which keeps exploration alive.
        """
        X = np.asarray(states, dtype=np.float64)        # (T, input)
        a = np.asarray(actions, dtype=np.int64)         # (T,)
        adv = np.asarray(advantages, dtype=np.float64)  # (T,)
        T = X.shape[0]

        # Forward pass, keeping the hidden activations for the backward pass.
        hidden = np.tanh(X @ self.w1 + self.b1)         # (T, hidden)
        probs = self._softmax(hidden @ self.w2 + self.b2)  # (T, actions)

        # d loss / d logits. For softmax + log-likelihood this collapses to
        # -(advantage) * (one_hot(action) - probs).
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(T), a] = 1.0
        d_logits = -adv[:, None] * (one_hot - probs)

        if entropy_coef:
            log_probs = np.log(probs + 1e-12)
            entropy = -np.sum(probs * log_probs, axis=1, keepdims=True)
            d_logits += entropy_coef * probs * (log_probs + entropy)

        d_logits /= T

        # Backpropagate through the output layer, then the tanh hidden layer.
        dw2 = hidden.T @ d_logits
        db2 = d_logits.sum(axis=0)
        d_hidden = d_logits @ self.w2.T
        d_z1 = d_hidden * (1.0 - hidden ** 2)           # tanh'(z) = 1 - tanh(z)^2
        dw1 = X.T @ d_z1
        db1 = d_z1.sum(axis=0)

        return np.concatenate([dw1.ravel(), db1, dw2.ravel(), db2])

    # ------------------------------------------------------------------
    # Weights as one flat vector
    # ------------------------------------------------------------------
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
