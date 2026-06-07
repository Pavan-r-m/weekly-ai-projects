"""
Reinforcement Learning: Q-Learning in a Grid World
===================================================
A complete implementation of Q-Learning — a model-free RL algorithm —
applied to a customizable grid world environment.

The agent learns to navigate from a start cell to a goal cell while:
  - Avoiding walls (impassable cells)
  - Avoiding hazard cells (large negative reward)
  - Finding the shortest safe path (positive reward on goal)

After training, the script saves:
  - A policy heatmap (learned Q-values and best actions)
  - A learning curve (cumulative reward per episode)
  - Frame-by-frame snapshots of the agent traversing its learned path

Usage:
    python gridworld_qlearning.py            # default 8x8 grid
    python gridworld_qlearning.py --size 12  # larger 12x12 grid
    python gridworld_qlearning.py --episodes 3000 --size 10
"""

import argparse
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")                        # headless rendering (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path


# ─────────────────────────────────────────────
# 1. GRID WORLD ENVIRONMENT
# ─────────────────────────────────────────────

class GridWorld:
    """
    A rectangular grid where the agent moves N/S/E/W.

    Cell types
    ----------
    0 : empty   (passable, small step penalty)
    1 : wall    (impassable)
    2 : hazard  (passable, large negative reward)
    3 : goal    (episode ends, large positive reward)
    """

    REWARD_STEP   = -0.04
    REWARD_HAZARD = -1.0
    REWARD_GOAL   = +10.0

    ACTIONS      = [0, 1, 2, 3]          # up, down, left, right
    ACTION_NAMES = ["up", "down", "left", "right"]
    DELTAS       = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, size: int = 8, seed: int = 42):
        self.size   = size
        self.rng    = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.grid   = self._generate_grid()
        self.start  = (0, 0)
        self.goal   = (size - 1, size - 1)
        self.state  = self.start

    # ── Grid generation ──────────────────────────────────────────────
    def _generate_grid(self) -> np.ndarray:
        size = self.size
        grid = np.zeros((size, size), dtype=int)

        # Walls (~15 % of cells)
        n_walls = int(0.15 * size * size)
        for _ in range(n_walls * 5):
            r, c = self.rng.randrange(size), self.rng.randrange(size)
            if (r, c) in {(0, 0), (size-1, size-1)}:
                continue
            if grid[r, c] == 0:
                grid[r, c] = 1
                if not self._is_reachable(grid, (0, 0), (size-1, size-1)):
                    grid[r, c] = 0
            if int(np.sum(grid == 1)) >= n_walls:
                break

        # Hazards (~8 % of cells)
        n_hazards = int(0.08 * size * size)
        placed = 0
        for _ in range(n_hazards * 5):
            r, c = self.rng.randrange(size), self.rng.randrange(size)
            if grid[r, c] == 0 and (r, c) not in {(0, 0), (size-1, size-1)}:
                grid[r, c] = 2
                placed += 1
            if placed >= n_hazards:
                break

        grid[size-1, size-1] = 3          # goal cell
        return grid

    def _is_reachable(self, grid, start, goal) -> bool:
        """BFS reachability check."""
        size = grid.shape[0]
        visited, queue = set(), [start]
        while queue:
            r, c = queue.pop()
            if (r, c) == goal:
                return True
            if (r, c) in visited:
                continue
            visited.add((r, c))
            for dr, dc in self.DELTAS.values():
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and grid[nr, nc] != 1:
                    queue.append((nr, nc))
        return False

    # ── Environment interface ─────────────────────────────────────────
    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action: int):
        r, c = self.state
        dr, dc = self.DELTAS[action]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.size and 0 <= nc < self.size and self.grid[nr, nc] != 1:
            self.state = (nr, nc)
        cell = self.grid[self.state]
        if cell == 3:
            return self.state, self.REWARD_GOAL,   True
        elif cell == 2:
            return self.state, self.REWARD_HAZARD, False
        else:
            return self.state, self.REWARD_STEP,   False

    def n_states(self):
        return self.size * self.size

    def state_index(self, state):
        return state[0] * self.size + state[1]


# ─────────────────────────────────────────────
# 2. Q-LEARNING AGENT
# ─────────────────────────────────────────────

class QLearningAgent:
    """
    Tabular Q-Learning with epsilon-greedy exploration.

    Update rule:
      Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
    """

    def __init__(self, n_states, n_actions,
                 alpha=0.1, gamma=0.95,
                 epsilon=1.0, eps_min=0.01, eps_decay=0.995):
        self.n_actions = n_actions
        self.alpha     = alpha
        self.gamma     = gamma
        self.epsilon   = epsilon
        self.eps_min   = eps_min
        self.eps_decay = eps_decay
        self.Q         = np.zeros((n_states, n_actions))

    def select_action(self, state_idx: int) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.Q[state_idx]))

    def update(self, s, a, r, s_next, done):
        best_next = 0.0 if done else float(np.max(self.Q[s_next]))
        td_target = r + self.gamma * best_next
        self.Q[s, a] += self.alpha * (td_target - self.Q[s, a])

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)


# ─────────────────────────────────────────────
# 3. TRAINING LOOP
# ─────────────────────────────────────────────

def train(env: GridWorld, agent: QLearningAgent,
          n_episodes: int, max_steps: int = 300):
    rewards_history = []
    for ep in range(n_episodes):
        state       = env.reset()
        total_reward = 0.0
        for step in range(max_steps):
            s_idx  = env.state_index(state)
            action = agent.select_action(s_idx)
            next_state, reward, done = env.step(action)
            ns_idx = env.state_index(next_state)
            agent.update(s_idx, action, reward, ns_idx, done)
            state        = next_state
            total_reward += reward
            if done:
                break
        agent.decay_epsilon()
        rewards_history.append(total_reward)
        if (ep + 1) % max(1, n_episodes // 10) == 0:
            avg = np.mean(rewards_history[-100:])
            print(f"  Episode {ep+1:>5}/{n_episodes}  "
                  f"avg_reward(last 100)={avg:+.3f}  eps={agent.epsilon:.3f}")
    return rewards_history


# ─────────────────────────────────────────────
# 4. GREEDY ROLLOUT
# ─────────────────────────────────────────────

def greedy_path(env: GridWorld, agent: QLearningAgent, max_steps: int = 300):
    """Follow greedy policy; return list of (row, col) cells visited."""
    state = env.reset()
    path  = [state]
    for _ in range(max_steps):
        s_idx      = env.state_index(state)
        action     = int(np.argmax(agent.Q[s_idx]))
        next_state, _, done = env.step(action)
        path.append(next_state)
        state = next_state
        if done:
            break
    return path


# ─────────────────────────────────────────────
# 5. VISUALISATION
# ─────────────────────────────────────────────

CELL_COLORS = {0: "#F0F4F8", 1: "#2D3748", 2: "#FC8181", 3: "#68D391"}


def _draw_grid_base(ax, env: GridWorld):
    """Render cell backgrounds, walls, hazards, goal symbol."""
    size = env.size
    labels = {1: "■", 2: "☠", 3: "★"}
    for r in range(size):
        for c in range(size):
            cell  = env.grid[r, c]
            color = CELL_COLORS.get(cell, CELL_COLORS[0])
            rect  = mpatches.FancyBboxPatch(
                (c + 0.05, size - r - 1 + 0.05), 0.9, 0.9,
                boxstyle="round,pad=0.02",
                facecolor=color, edgecolor="#CBD5E0", linewidth=0.5)
            ax.add_patch(rect)
            if cell in labels:
                ax.text(c + 0.5, size - r - 0.5, labels[cell],
                        ha="center", va="center",
                        fontsize=10, color="white", fontweight="bold")
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_policy(env: GridWorld, agent: QLearningAgent, path, out_dir: Path):
    """Save policy heatmap: Q-value colours + best-action arrows + path overlay."""
    size  = env.size
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#1A202C")
    ax.set_facecolor("#1A202C")

    max_q = np.max(agent.Q, axis=1).reshape(size, size)
    vmin, vmax = float(max_q.min()), float(max_q.max())
    cmap = LinearSegmentedColormap.from_list(
        "qcmap", ["#2D3748", "#4299E1", "#F6E05E"])

    labels = {1: "■", 2: "☠", 3: "★"}
    lcolors = {1: "#718096", 2: "#FC8181", 3: "#F6E05E"}

    for r in range(size):
        for c in range(size):
            cell = env.grid[r, c]
            if cell == 1:
                facecolor = CELL_COLORS[1]
            else:
                t = (max_q[r, c] - vmin) / (vmax - vmin + 1e-9)
                facecolor = cmap(float(t))
            rect = mpatches.FancyBboxPatch(
                (c + 0.05, size - r - 1 + 0.05), 0.9, 0.9,
                boxstyle="round,pad=0.02",
                facecolor=facecolor, edgecolor="#2D3748", linewidth=0.8)
            ax.add_patch(rect)

            if cell not in (1, 3):
                s_idx  = env.state_index((r, c))
                best_a = int(np.argmax(agent.Q[s_idx]))
                dr, dc = GridWorld.DELTAS[best_a]
                ax.annotate("",
                    xy=(c + 0.5 + dc*0.28, size-r-0.5 - dr*0.28),
                    xytext=(c + 0.5, size-r-0.5),
                    arrowprops=dict(arrowstyle="-|>", color="white",
                                    lw=1.1, mutation_scale=9))
            if cell in labels:
                ax.text(c+0.5, size-r-0.5, labels[cell],
                        ha="center", va="center", fontsize=11,
                        color=lcolors[cell], fontweight="bold")

    # Path overlay
    pr = [size - r - 0.5 for r, _ in path]
    pc = [c + 0.5 for _, c in path]
    ax.plot(pc, pr, color="#F6AD55", linewidth=2.5, alpha=0.9, zorder=5,
            label="Learned path")
    ax.plot(pc[0], pr[0], "o", color="#68D391", markersize=10, zorder=6,
            label="Start")
    ax.plot(pc[-1], pr[-1], "*", color="#F6E05E", markersize=14, zorder=6,
            label="Goal reached")

    ax.set_xlim(0, size); ax.set_ylim(0, size)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Learned Policy — Q-Value Heatmap + Best Actions",
                 color="white", fontsize=14, pad=12)
    leg = ax.legend(loc="upper left", framealpha=0.3,
                    labelcolor="white", fontsize=9)
    leg.get_frame().set_edgecolor("#718096")
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Max Q-value", color="white", fontsize=10)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    fpath = out_dir / "policy_heatmap.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {fpath}")


def plot_learning_curve(rewards, out_dir: Path):
    """Save rolling-average reward curve."""
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#1A202C")
    ax.set_facecolor("#263044")

    eps  = np.arange(1, len(rewards) + 1)
    win  = max(1, len(rewards) // 50)
    smoothed = np.convolve(rewards, np.ones(win)/win, mode="valid")

    ax.plot(eps, rewards, color="#4299E1", alpha=0.25, linewidth=0.6,
            label="Raw reward")
    ax.plot(eps[:len(smoothed)], smoothed, color="#F6AD55",
            linewidth=2.0, label=f"Smoothed (w={win})")
    ax.axhline(0, color="#718096", linewidth=0.8, linestyle="--")

    ax.set_xlabel("Episode", color="white", fontsize=11)
    ax.set_ylabel("Total Reward", color="white", fontsize=11)
    ax.set_title("Q-Learning Training Curve", color="white", fontsize=13)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#4A5568")
    leg = ax.legend(framealpha=0.2, labelcolor="white", fontsize=9)
    leg.get_frame().set_edgecolor("#718096")

    fpath = out_dir / "learning_curve.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {fpath}")


def plot_path_frames(env: GridWorld, path, out_dir: Path, n_frames: int = 6):
    """Save n_frames snapshots of the agent walking its learned path."""
    frames_dir = out_dir / "path_frames"
    frames_dir.mkdir(exist_ok=True)
    size = env.size
    indices = np.linspace(0, len(path)-1, min(n_frames, len(path)),
                          dtype=int)
    for fi, pi in enumerate(indices):
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor("#1A202C")
        ax.set_facecolor("#1A202C")
        _draw_grid_base(ax, env)

        past = path[:pi+1]
        pr   = [size - r - 0.5 for r, _ in past]
        pc   = [c + 0.5 for _, c in past]
        ax.plot(pc, pr, color="#F6AD55", linewidth=2, alpha=0.8, zorder=5)
        r, c = path[pi]
        ax.plot(c + 0.5, size - r - 0.5, "o",
                color="#63B3ED", markersize=14, zorder=6)
        ax.set_title(f"Step {pi} / {len(path)-1}",
                     color="white", fontsize=11)
        fpath = frames_dir / f"frame_{fi:02d}.png"
        fig.savefig(fpath, dpi=100, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"  Saved {len(indices)} path frames → {frames_dir}/")


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Q-Learning Grid World")
    parser.add_argument("--size",     type=int,   default=8,    help="Grid side length")
    parser.add_argument("--episodes", type=int,   default=2000, help="Training episodes")
    parser.add_argument("--alpha",    type=float, default=0.1,  help="Learning rate")
    parser.add_argument("--gamma",    type=float, default=0.95, help="Discount factor")
    parser.add_argument("--seed",     type=int,   default=42,   help="Random seed")
    parser.add_argument("--out",      type=str,   default="output", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  Q-Learning Grid World")
    print("=" * 55)
    print(f"  Grid size : {args.size}x{args.size}")
    print(f"  Episodes  : {args.episodes}")
    print(f"  alpha={args.alpha}  gamma={args.gamma}  seed={args.seed}")
    print()

    env   = GridWorld(size=args.size, seed=args.seed)
    agent = QLearningAgent(
        n_states  = env.n_states(),
        n_actions = len(GridWorld.ACTIONS),
        alpha     = args.alpha,
        gamma     = args.gamma,
        epsilon   = 1.0,
        eps_min   = 0.01,
        eps_decay = 0.997,
    )

    walls   = int(np.sum(env.grid == 1))
    hazards = int(np.sum(env.grid == 2))
    print(f"  Grid contains {walls} walls and {hazards} hazards.\n")

    print("Training ...")
    rewards = train(env, agent, n_episodes=args.episodes)

    path    = greedy_path(env, agent)
    reached = env.grid[path[-1]] == 3
    print(f"\nGreedy path: {len(path)-1} steps | "
          f"Reached goal: {'YES' if reached else 'NO'}")

    print("\nSaving visualisations ...")
    plot_policy(env, agent, path, out_dir)
    plot_learning_curve(rewards, out_dir)
    plot_path_frames(env, path, out_dir)

    print("\nDone! Outputs in:", out_dir.resolve())
    print("=" * 55)


if __name__ == "__main__":
    main()
