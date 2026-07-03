"""
Multi-Armed Bandit Playground
==============================

Simulates the classic "multi-armed bandit" problem — a row of slot machines
(arms) each paying out with a hidden, unknown probability — and compares how
three different exploration strategies learn to find the best arm:

  1. Epsilon-Greedy      : mostly exploit, occasionally explore at random
  2. UCB1 (Upper Confidence Bound): explore arms with high uncertainty
  3. Thompson Sampling   : Bayesian approach, sample from a Beta posterior

This is the foundational problem behind recommendation systems, A/B testing,
ad placement, and the "explore vs. exploit" dilemma that shows up everywhere
in reinforcement learning.

No API key, no internet connection, and no external dataset required —
everything is simulated with numpy.

Run:
    python bandit_playground.py
    python bandit_playground.py --steps 5000 --trials 200 --arms 10
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# The environment: a set of Bernoulli "slot machines" with hidden win rates
# ---------------------------------------------------------------------------
class BernoulliBandit:
    """A row of slot machines. Arm i pays out 1 with probability true_probs[i]."""

    def __init__(self, true_probs):
        self.true_probs = np.array(true_probs)
        self.n_arms = len(true_probs)
        self.best_prob = self.true_probs.max()

    def pull(self, arm):
        """Pull an arm, return a random 0/1 reward drawn from its true probability."""
        return 1 if np.random.random() < self.true_probs[arm] else 0


# ---------------------------------------------------------------------------
# Strategy 1: Epsilon-Greedy
# ---------------------------------------------------------------------------
class EpsilonGreedyAgent:
    """With probability epsilon, pick a random arm (explore).
    Otherwise, pick the arm with the best average reward so far (exploit)."""

    name = "Epsilon-Greedy"

    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)      # how many times each arm was pulled
        self.values = np.zeros(n_arms)      # running average reward per arm

    def select_arm(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)          # explore
        return int(np.argmax(self.values))                  # exploit best-known arm

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        # incremental mean update: new_avg = old_avg + (reward - old_avg) / n
        self.values[arm] += (reward - self.values[arm]) / n


# ---------------------------------------------------------------------------
# Strategy 2: UCB1 (Upper Confidence Bound)
# ---------------------------------------------------------------------------
class UCB1Agent:
    """Picks the arm that maximizes: average_reward + confidence_bonus.
    The bonus shrinks the more an arm has been tried, encouraging exploration
    of arms we're still uncertain about."""

    name = "UCB1"

    def __init__(self, n_arms, c=2.0):
        self.n_arms = n_arms
        self.c = c
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_pulls = 0

    def select_arm(self):
        # Try every arm at least once before trusting the formula
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        ucb_values = self.values + self.c * np.sqrt(
            np.log(self.total_pulls) / self.counts
        )
        return int(np.argmax(ucb_values))

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.total_pulls += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


# ---------------------------------------------------------------------------
# Strategy 3: Thompson Sampling
# ---------------------------------------------------------------------------
class ThompsonSamplingAgent:
    """Bayesian approach. Keeps a Beta(alpha, beta) belief distribution over
    each arm's win rate. To choose an arm, it samples one guess from each
    arm's belief and picks the arm with the highest sampled value. Beliefs
    sharpen automatically as evidence (wins/losses) accumulates."""

    name = "Thompson Sampling"

    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)   # prior successes (starts at 1 = uniform prior)
        self.beta = np.ones(n_arms)    # prior failures

    def select_arm(self):
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm, reward):
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------
def run_simulation(agent_class, bandit, n_steps, agent_kwargs=None):
    """Run one agent against one bandit for n_steps, return reward + regret history."""
    agent_kwargs = agent_kwargs or {}
    agent = agent_class(bandit.n_arms, **agent_kwargs)

    rewards = np.zeros(n_steps)
    regrets = np.zeros(n_steps)

    for t in range(n_steps):
        arm = agent.select_arm()
        reward = bandit.pull(arm)
        agent.update(arm, reward)

        rewards[t] = reward
        # regret = how much worse this pull was vs. always picking the best arm
        regrets[t] = bandit.best_prob - bandit.true_probs[arm]

    return rewards, regrets


def average_over_trials(agent_class, true_probs, n_steps, n_trials, agent_kwargs=None):
    """Repeat the simulation n_trials times (fresh bandit + agent each time)
    and average the results, since a single run is noisy."""
    all_rewards = np.zeros((n_trials, n_steps))
    all_regrets = np.zeros((n_trials, n_steps))

    for trial in range(n_trials):
        bandit = BernoulliBandit(true_probs)
        rewards, regrets = run_simulation(agent_class, bandit, n_steps, agent_kwargs)
        all_rewards[trial] = rewards
        all_regrets[trial] = regrets

    return all_rewards.mean(axis=0), all_regrets.mean(axis=0)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(results, true_probs, out_path="bandit_results.png"):
    """results: dict of {agent_name: (avg_rewards, avg_regrets)}"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"Epsilon-Greedy": "#e74c3c", "UCB1": "#3498db", "Thompson Sampling": "#2ecc71"}

    # Panel 1: cumulative regret (lower = better; the key metric in bandit research)
    ax = axes[0]
    for name, (rewards, regrets) in results.items():
        ax.plot(np.cumsum(regrets), label=name, color=colors.get(name), linewidth=2)
    ax.set_title("Cumulative Regret (lower is better)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Total regret")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: average reward over time (rolling window smooths the noise)
    ax = axes[1]
    window = 50
    for name, (rewards, regrets) in results.items():
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(smoothed, label=name, color=colors.get(name), linewidth=2)
    ax.axhline(true_probs.max(), color="gray", linestyle="--", label="Best possible")
    ax.set_title(f"Average Reward (rolling window={window})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward rate")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 3: the true win probabilities of each arm, so readers can see the setup
    ax = axes[2]
    arms = np.arange(len(true_probs))
    bars = ax.bar(arms, true_probs, color="#9b59b6", alpha=0.8)
    best_arm = np.argmax(true_probs)
    bars[best_arm].set_color("#f39c12")
    ax.set_title("True Win Probability per Arm (hidden from agents)")
    ax.set_xlabel("Arm index")
    ax.set_ylabel("True probability")
    ax.set_xticks(arms)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved plot to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Multi-Armed Bandit Playground")
    parser.add_argument("--arms", type=int, default=8, help="Number of bandit arms")
    parser.add_argument("--steps", type=int, default=3000, help="Steps per trial")
    parser.add_argument("--trials", type=int, default=150, help="Trials to average over")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Generate random but fixed "true" win probabilities for each arm.
    # These represent the hidden payout rates the agents must discover.
    true_probs = np.round(np.random.uniform(0.05, 0.95, size=args.arms), 3)
    best_arm = int(np.argmax(true_probs))

    print("=" * 60)
    print("MULTI-ARMED BANDIT PLAYGROUND")
    print("=" * 60)
    print(f"Arms: {args.arms} | Steps/trial: {args.steps} | Trials averaged: {args.trials}")
    print(f"True win probabilities: {true_probs}")
    print(f"Best arm: #{best_arm} (p={true_probs[best_arm]})\n")

    agents = {
        "Epsilon-Greedy": (EpsilonGreedyAgent, {"epsilon": 0.1}),
        "UCB1": (UCB1Agent, {"c": 2.0}),
        "Thompson Sampling": (ThompsonSamplingAgent, {}),
    }

    results = {}
    for name, (agent_class, kwargs) in agents.items():
        print(f"Running {name}...")
        rewards, regrets = average_over_trials(
            agent_class, true_probs, args.steps, args.trials, kwargs
        )
        results[name] = (rewards, regrets)
        total_regret = regrets.sum()
        avg_reward = rewards.mean()
        print(f"  -> Total avg regret: {total_regret:.2f} | Avg reward rate: {avg_reward:.4f}")

    print("\nSummary (lower regret & higher reward = better):")
    print(f"{'Strategy':<20}{'Total Regret':<16}{'Avg Reward':<12}")
    print("-" * 48)
    for name, (rewards, regrets) in results.items():
        print(f"{name:<20}{regrets.sum():<16.2f}{rewards.mean():<12.4f}")

    plot_results(results, true_probs)


if __name__ == "__main__":
    main()
