# Reinforcement Learning: Q-Learning Grid World

A complete, from-scratch implementation of **Q-Learning** — one of the most fundamental model-free reinforcement learning algorithms — applied to a procedurally generated grid world.

The agent starts in the top-left corner and must find the optimal path to the goal in the bottom-right corner, navigating around walls and avoiding hazardous cells.

---

## What it does

- Generates a random NxN grid with walls (impassable), hazards (penalty), and a goal cell
- Trains a tabular Q-learning agent using ε-greedy exploration that decays over time
- After training, extracts and visualises the **learned policy** (best action per cell) as a heatmap
- Saves a **learning curve** showing cumulative reward over episodes
- Saves **path frames** showing the agent walking its discovered route step by step

---

## Tech Stack & Key Concepts

| Concept | Details |
|---|---|
| **Algorithm** | Tabular Q-Learning (off-policy TD control) |
| **Exploration** | ε-greedy with exponential decay |
| **Environment** | Custom GridWorld (no gym dependency) |
| **Reward shaping** | Step penalty (−0.04), hazard (−1.0), goal (+10.0) |
| **Libraries** | `numpy`, `matplotlib` only |

The Bellman update at each step:

```
Q(s, a) ← Q(s, a) + α [ r + γ · max_a' Q(s', a') − Q(s, a) ]
```

---

## Installation

```bash
pip install -r requirements.txt
```

No API keys, no external datasets, no GPU required.

---

## How to Run

```bash
# Default: 8×8 grid, 2000 training episodes
python gridworld_qlearning.py

# Larger grid, more training
python gridworld_qlearning.py --size 12 --episodes 5000

# Custom hyperparameters
python gridworld_qlearning.py --size 10 --alpha 0.05 --gamma 0.99 --seed 7

# All options
python gridworld_qlearning.py --help
```

Outputs are saved to `./output/` (or specify `--out my_folder`).

---

## Example Output

```
=======================================================
  Q-Learning Grid World
=======================================================
  Grid size : 8x8
  Episodes  : 2000
  alpha=0.1  gamma=0.95  seed=42

  Grid contains 9 walls and 4 hazards.

Training ...
  Episode   200/2000  avg_reward(last 100)=-2.847  eps=0.549
  Episode   400/2000  avg_reward(last 100)=-1.203  eps=0.301
  Episode   600/2000  avg_reward(last 100)=+5.441  eps=0.165
  Episode   800/2000  avg_reward(last 100)=+7.812  eps=0.091
  Episode  2000/2000  avg_reward(last 100)=+9.037  eps=0.010

Greedy path: 14 steps | Reached goal: YES

Saving visualisations ...
  Saved: output/policy_heatmap.png
  Saved: output/learning_curve.png
  Saved 6 path frames → output/path_frames/
```

The agent typically converges to a near-optimal path within 500–1000 episodes on an 8×8 grid.

---

## How It Works

### 1. Environment
The grid is procedurally generated using BFS to guarantee the goal is always reachable. Walls are placed randomly but rejected if they would block all paths to the goal.

### 2. Q-Table
A 2D array `Q[state, action]` initialised to zero. Each state is encoded as `row * grid_size + col`.

### 3. ε-Greedy Exploration
Early in training (ε ≈ 1.0), the agent explores randomly. As ε decays toward 0.01, it increasingly exploits what it has learned.

### 4. Bellman Update
After every step the agent updates the Q-value for the (state, action) pair using the TD error — the difference between the expected and observed future reward.

### 5. Greedy Policy Extraction
After training, the policy is read off the Q-table: for each state, choose `argmax_a Q(s, a)`. The heatmap overlays this policy on a colour-coded map of max Q-values.

---

## Files

```
2026-06-07-rl-gridworld-qlearning/
├── gridworld_qlearning.py   # complete implementation (~450 lines)
├── requirements.txt
└── README.md
```

Output (generated on run):
```
output/
├── policy_heatmap.png       # Q-value heatmap + best-action arrows + path
├── learning_curve.png       # reward per episode, smoothed
└── path_frames/
    ├── frame_00.png         # agent at start
    ├── frame_01.png
    └── ...                  # step-by-step path snapshots
```
