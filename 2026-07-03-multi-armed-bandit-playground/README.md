# Multi-Armed Bandit Playground

**Category:** Wild card (algorithms / decision-making under uncertainty)

A from-scratch simulation of the classic **multi-armed bandit** problem — a row of slot machines, each with a hidden, unknown payout rate — used to compare three famous "explore vs. exploit" strategies:

1. **Epsilon-Greedy** — mostly exploit the best-known arm, occasionally explore at random
2. **UCB1 (Upper Confidence Bound)** — favor arms we're still uncertain about
3. **Thompson Sampling** — Bayesian approach that samples from a Beta posterior over each arm's win rate

This is the same underlying problem behind real-world systems like A/B testing, ad placement, recommendation engines, and clinical trial design — anywhere a system has to balance trying new options against sticking with what already works.

## Why it's interesting

The "explore vs. exploit" trade-off is one of the most fundamental ideas in decision-making and reinforcement learning, but it's rarely shown side-by-side with real numbers. This project runs all three strategies against the *same* hidden bandit, averages over many trials to smooth out noise, and plots cumulative regret so you can see — not just intuit — why Thompson Sampling tends to win in practice despite being the simplest to state.

## Tech stack & key concepts

- **numpy** — vectorized simulation, Beta/Bernoulli sampling
- **matplotlib** — 3-panel comparison dashboard
- Concepts: exploration vs. exploitation, regret minimization, incremental mean updates, confidence bounds, Bayesian posterior updating (Beta-Bernoulli conjugacy)

No API key, internet connection, or external dataset needed — everything is simulated.

## Installation

```bash
pip install -r requirements.txt
```

## How to run

```bash
# Default run: 8 arms, 3000 steps, averaged over 150 trials
python bandit_playground.py

# Customize the experiment
python bandit_playground.py --arms 10 --steps 5000 --trials 200 --seed 7
```

Arguments:
- `--arms` — number of slot machines (default 8)
- `--steps` — pulls per trial (default 3000)
- `--trials` — independent trials to average over, since a single run is noisy (default 150)
- `--seed` — random seed for reproducibility (default 42)

## Example output

```
============================================================
MULTI-ARMED BANDIT PLAYGROUND
============================================================
Arms: 8 | Steps/trial: 3000 | Trials averaged: 150
True win probabilities: [0.387 0.906 0.709 0.589 0.19  0.19  0.102 0.83 ]
Best arm: #1 (p=0.906)

Running Epsilon-Greedy...
  -> Total avg regret: 77.95 | Avg reward rate: 0.8314
Running UCB1...
  -> Total avg regret: 156.59 | Avg reward rate: 0.7439
Running Thompson Sampling...
  -> Total avg regret: 23.62 | Avg reward rate: 0.8832

Summary (lower regret & higher reward = better):
Strategy            Total Regret    Avg Reward
------------------------------------------------
Epsilon-Greedy      77.95           0.8314
UCB1                156.59          0.7439
Thompson Sampling   23.62           0.8832

Saved plot to bandit_results.png
```

The script also saves `bandit_results.png`, a 3-panel chart showing cumulative regret over time, smoothed average reward vs. the theoretical best, and the (normally hidden) true win probability of each arm.

## How it works

**The environment (`BernoulliBandit`)** — each arm pays out `1` with some fixed but unknown probability and `0` otherwise. The agent never sees these probabilities directly; it only observes the rewards it receives.

**Epsilon-Greedy** keeps a running average reward per arm. With probability `epsilon` (default 0.1) it picks a uniformly random arm to explore; otherwise it exploits by picking the arm with the current highest average. Simple, but its exploration never slows down or gets smarter about *which* arm to try.

**UCB1** picks the arm maximizing `average_reward + c * sqrt(log(total_pulls) / arm_pulls)`. The second term is a "confidence bonus" that shrinks the more an arm has been tried — so arms that are rarely tested keep getting a boost, guaranteeing every arm eventually gets attention, while confident, well-tested arms are exploited more.

**Thompson Sampling** treats each arm's win rate as a random variable with a `Beta(alpha, beta)` belief distribution, starting from a uniform prior `Beta(1, 1)`. On each round, it draws one random sample from every arm's belief and pulls the arm with the highest sample. Wins increment `alpha`, losses increment `beta` — so beliefs sharpen automatically as evidence accumulates, naturally balancing exploration and exploitation without any tunable parameter.

**Regret** is the metric used to compare them: at every step, regret is `best_possible_probability - true_probability_of_chosen_arm`. Cumulative regret over time is the standard way researchers measure how quickly a bandit algorithm converges on the optimal arm — the flatter the curve, the better.
