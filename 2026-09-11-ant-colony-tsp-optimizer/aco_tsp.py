"""
Ant Colony Optimization (ACO) for the Traveling Salesman Problem (TSP)
========================================================================

A from-scratch implementation of a swarm-intelligence metaheuristic inspired
by how real ants find shortest paths between their colony and food sources
using pheromone trails.

The algorithm:
  1. A colony of virtual "ants" is placed on random cities.
  2. Each ant builds a complete tour by probabilistically choosing the next
     city, biased towards nearby cities (heuristic desirability) and cities
     connected by strong pheromone trails (learned experience).
  3. After all ants finish, pheromone evaporates a little everywhere (so old,
     bad information fades) and each ant deposits new pheromone on the edges
     of its tour, proportional to how good (short) that tour was.
  4. Over many iterations, pheromone concentrates on edges that belong to
     good tours, and the colony converges on a near-optimal route.

This script also runs a classic Nearest-Neighbor heuristic as a baseline for
comparison, then plots the best route found and the convergence curve.

No external services or API keys are required -- everything runs locally
with numpy + matplotlib.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Problem setup: generate a random set of "city" coordinates on a 2D plane
# ---------------------------------------------------------------------------
def generate_cities(n_cities: int, width: float = 100.0, height: float = 100.0) -> np.ndarray:
    """Return an (n_cities, 2) array of random (x, y) coordinates."""
    xs = np.random.uniform(0, width, n_cities)
    ys = np.random.uniform(0, height, n_cities)
    return np.stack([xs, ys], axis=1)


def build_distance_matrix(cities: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance matrix between all cities."""
    diff = cities[:, np.newaxis, :] - cities[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    # Avoid division by zero on the diagonal later by giving self-distance
    # a tiny nonzero value (it's never actually traveled).
    np.fill_diagonal(dist, 1e-10)
    return dist


def tour_length(tour: list, dist: np.ndarray) -> float:
    """Total length of a closed tour (returns to the starting city)."""
    total = 0.0
    for i in range(len(tour)):
        a, b = tour[i], tour[(i + 1) % len(tour)]
        total += dist[a, b]
    return total


# ---------------------------------------------------------------------------
# Baseline heuristic: Nearest Neighbor
# ---------------------------------------------------------------------------
def nearest_neighbor_tour(dist: np.ndarray, start: int = 0) -> list:
    """Greedy baseline: always hop to the closest unvisited city."""
    n = dist.shape[0]
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    current = start
    while unvisited:
        nxt = min(unvisited, key=lambda city: dist[current, city])
        tour.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return tour


# ---------------------------------------------------------------------------
# Ant Colony Optimization
# ---------------------------------------------------------------------------
class AntColonyOptimizer:
    """
    Parameters
    ----------
    dist        : (n, n) distance matrix
    n_ants      : number of ants per iteration (default: n_cities)
    n_iterations: number of colony iterations to run
    alpha       : pheromone influence exponent (higher = trust trails more)
    beta        : heuristic (1/distance) influence exponent (higher = greedier)
    evaporation : fraction of pheromone that evaporates each iteration (0-1)
    q           : pheromone deposit constant (total pheromone laid by a tour
                  of length L is q / L)
    elitist_weight: extra pheromone boost given to the best-so-far tour,
                  which speeds up convergence (a common ACO enhancement)
    """

    def __init__(self, dist, n_ants=None, n_iterations=200, alpha=1.0,
                 beta=3.0, evaporation=0.5, q=100.0, elitist_weight=2.0):
        self.dist = dist
        self.n = dist.shape[0]
        self.n_ants = n_ants or self.n
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.q = q
        self.elitist_weight = elitist_weight

        # Heuristic desirability eta_ij = 1 / distance_ij
        self.eta = 1.0 / dist

        # Initialize pheromone uniformly on all edges
        self.pheromone = np.ones((self.n, self.n)) * 0.1

        self.best_tour = None
        self.best_length = float("inf")
        self.history = []  # best length found so far, per iteration

    def _construct_tour(self, start_city: int) -> list:
        """One ant builds a full tour using the random-proportional rule."""
        visited = [start_city]
        unvisited = set(range(self.n)) - {start_city}
        current = start_city

        while unvisited:
            candidates = list(unvisited)
            # Attractiveness of each candidate edge: pheromone^alpha * eta^beta
            tau = self.pheromone[current, candidates] ** self.alpha
            eta = self.eta[current, candidates] ** self.beta
            weights = tau * eta
            total = weights.sum()

            if total <= 0 or not np.isfinite(total):
                # Fallback: pick uniformly at random if numerics break down
                probs = np.ones(len(candidates)) / len(candidates)
            else:
                probs = weights / total

            next_city = np.random.choice(candidates, p=probs)
            visited.append(next_city)
            unvisited.remove(next_city)
            current = next_city

        return visited

    def _update_pheromone(self, all_tours, all_lengths):
        """Evaporate old pheromone, then deposit new pheromone per ant."""
        self.pheromone *= (1.0 - self.evaporation)
        self.pheromone[self.pheromone < 1e-13] = 1e-13  # numerical floor

        for tour, length in zip(all_tours, all_lengths):
            deposit = self.q / length
            for i in range(len(tour)):
                a, b = tour[i], tour[(i + 1) % len(tour)]
                self.pheromone[a, b] += deposit
                self.pheromone[b, a] += deposit  # symmetric TSP

        # Elitist reinforcement: extra pheromone on the best tour ever found
        if self.best_tour is not None:
            deposit = self.elitist_weight * self.q / self.best_length
            for i in range(len(self.best_tour)):
                a, b = self.best_tour[i], self.best_tour[(i + 1) % len(self.best_tour)]
                self.pheromone[a, b] += deposit
                self.pheromone[b, a] += deposit

    def run(self, verbose=True) -> tuple:
        """Run the full ACO optimization loop. Returns (best_tour, best_length)."""
        for it in range(self.n_iterations):
            all_tours = []
            all_lengths = []

            for _ in range(self.n_ants):
                start = np.random.randint(self.n)
                tour = self._construct_tour(start)
                length = tour_length(tour, self.dist)
                all_tours.append(tour)
                all_lengths.append(length)

                if length < self.best_length:
                    self.best_length = length
                    self.best_tour = tour

            self._update_pheromone(all_tours, all_lengths)
            self.history.append(self.best_length)

            if verbose and (it % 20 == 0 or it == self.n_iterations - 1):
                print(f"  iteration {it:4d}/{self.n_iterations}  "
                      f"best length so far: {self.best_length:.2f}")

        return self.best_tour, self.best_length


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_route(cities, tour, title, filepath):
    fig, ax = plt.subplots(figsize=(7, 7))
    ordered = cities[tour + [tour[0]]]  # close the loop
    ax.plot(ordered[:, 0], ordered[:, 1], "o-", color="#2b6cb0", markersize=6,
            linewidth=1.4, markerfacecolor="#f6ad55", markeredgecolor="#2b6cb0")
    for idx, (x, y) in enumerate(cities):
        ax.annotate(str(idx), (x, y), fontsize=7, xytext=(3, 3),
                    textcoords="offset points", color="#4a5568")
    ax.scatter(*cities[tour[0]], color="#e53e3e", s=90, zorder=5, label="Start city")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(filepath, dpi=140)
    plt.close(fig)


def plot_convergence(history, nn_length, filepath):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history, color="#38a169", linewidth=2, label="ACO best-so-far tour length")
    ax.axhline(nn_length, color="#e53e3e", linestyle="--", linewidth=1.5,
               label=f"Nearest-Neighbor baseline ({nn_length:.1f})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Tour length")
    ax.set_title("ACO Convergence: Best Tour Length vs. Iteration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filepath, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    n_cities = 30
    print(f"Generating {n_cities} random cities (seed={SEED})...")
    cities = generate_cities(n_cities)
    dist = build_distance_matrix(cities)

    # --- Baseline ---
    print("\nRunning Nearest-Neighbor baseline...")
    nn_tour = nearest_neighbor_tour(dist, start=0)
    nn_length = tour_length(nn_tour, dist)
    print(f"  Nearest-Neighbor tour length: {nn_length:.2f}")

    # --- ACO ---
    print("\nRunning Ant Colony Optimization...")
    aco = AntColonyOptimizer(
        dist,
        n_ants=30,
        n_iterations=200,
        alpha=1.0,
        beta=3.0,
        evaporation=0.5,
        q=100.0,
        elitist_weight=2.0,
    )
    best_tour, best_length = aco.run(verbose=True)

    improvement = (nn_length - best_length) / nn_length * 100
    print("\n===== RESULTS =====")
    print(f"Nearest-Neighbor length : {nn_length:.2f}")
    print(f"ACO best length         : {best_length:.2f}")
    print(f"Improvement over NN     : {improvement:.1f}%")
    print(f"Best tour (city order)  : {best_tour}")

    # --- Save plots ---
    plot_route(cities, nn_tour, f"Nearest-Neighbor Route (length={nn_length:.1f})",
               os.path.join(OUTPUT_DIR, "nearest_neighbor_route.png"))
    plot_route(cities, best_tour, f"ACO Best Route (length={best_length:.1f})",
               os.path.join(OUTPUT_DIR, "aco_best_route.png"))
    plot_convergence(aco.history, nn_length,
                      os.path.join(OUTPUT_DIR, "convergence.png"))

    # --- Save summary ---
    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Ant Colony Optimization for TSP - Run Summary\n")
        f.write("=" * 48 + "\n")
        f.write(f"Cities: {n_cities} (seed={SEED})\n")
        f.write(f"Nearest-Neighbor length: {nn_length:.2f}\n")
        f.write(f"ACO best length: {best_length:.2f}\n")
        f.write(f"Improvement over NN: {improvement:.1f}%\n")
        f.write(f"Best tour: {best_tour}\n")

    print(f"\nPlots and summary saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
