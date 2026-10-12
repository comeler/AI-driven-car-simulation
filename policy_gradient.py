"""
policy_gradient.py: REINFORCE, the simplest policy-gradient algorithm.

Idea: let the car drive using a STOCHASTIC policy (actions sampled from the
network's output probabilities), record everything it did and the rewards it
got, then nudge the weights so actions that led to high return become more
likely and actions that led to low return become less likely. That nudge is
a gradient descent step - computed by NeuralNetwork.policy_gradient().

Contrast with the genetic algorithm: there, weights change by random
mutation and selection; here, they change in the direction the gradient says
will increase expected reward.
"""

import numpy as np


class Adam:
    """Adam optimizer: gradient descent with a running average of the gradient
    (momentum) and of its square (per-weight step size scaling). Far less
    sensitive to the learning rate than plain gradient descent."""

    def __init__(self, num_params, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(num_params)
        self.v = np.zeros(num_params)
        self.t = 0

    def step(self, params, grad):
        """Return the updated parameters (one gradient DESCENT step)."""
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)


def collect_episode(env, net, rng):
    """Drive one full episode, SAMPLING each action from the policy.
    Returns the states seen, actions taken, rewards received, and the final
    info dict (laps completed, etc.)."""
    state = env.reset()
    states, actions, rewards = [], [], []
    done = False
    info = {}
    while not done:
        action = net.sample_action(state, rng)
        next_state, reward, done, info = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state
    return states, actions, rewards, info


def discounted_returns(rewards, gamma):
    """G_t = r_t + gamma * G_{t+1}: the reward from step t onward, with
    later rewards counting a bit less (gamma < 1)."""
    returns = np.zeros(len(rewards))
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def evaluate_greedy(env, net):
    """One episode using the GREEDY policy (argmax, no randomness) - exactly
    how watch_best.py and evaluate.py will drive the car. The environment is
    deterministic, so this is the network's true performance. Returns
    (total_reward, laps_completed, steps)."""
    state = env.reset()
    total_reward = 0.0
    steps = 0
    done = False
    info = {}
    while not done:
        state, reward, done, info = env.step(net.choose_action(state))
        total_reward += reward
        steps += 1
    return total_reward, info["laps_completed"], steps


class PolicyGradientTrainer:
    def __init__(self, net, learning_rate, gamma, entropy_coef, grad_clip, seed=None):
        self.net = net
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.grad_clip = grad_clip
        self.optimizer = Adam(net.num_weights, learning_rate)
        self.rng = np.random.default_rng(seed)

    def update(self, env, num_episodes):
        """Collect num_episodes episodes with the current policy, then take
        ONE gradient step using all of them together. Returns a stats dict."""
        all_states, all_actions, all_returns = [], [], []
        ep_rewards, ep_lengths, ep_laps = [], [], []

        for _ in range(num_episodes):
            states, actions, rewards, info = collect_episode(env, self.net, self.rng)
            all_states.extend(states)
            all_actions.extend(actions)
            all_returns.append(discounted_returns(rewards, self.gamma))
            ep_rewards.append(sum(rewards))
            ep_lengths.append(len(rewards))
            ep_laps.append(info["laps_completed"])

        # Advantage = how much better than average this step's return was.
        # Normalizing across the whole batch keeps gradient size stable no
        # matter what scale the rewards in config.py happen to have.
        returns = np.concatenate(all_returns)
        advantages = (returns - returns.mean()) / (returns.std() + 1e-8)

        grad = self.net.policy_gradient(all_states, all_actions, advantages, self.entropy_coef)

        grad_norm = float(np.linalg.norm(grad))
        if self.grad_clip > 0 and grad_norm > self.grad_clip:
            grad = grad * (self.grad_clip / grad_norm)

        self.net.set_weights(self.optimizer.step(self.net.get_weights(), grad))

        return {
            "mean_reward": float(np.mean(ep_rewards)),
            "mean_length": float(np.mean(ep_lengths)),
            "mean_laps": float(np.mean(ep_laps)),
            "grad_norm": grad_norm,
        }
