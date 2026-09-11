# Ant Colony Optimization for the Traveling Salesman Problem

A from-scratch implementation of **Ant Colony Optimization (ACO)**, a
bio-inspired swarm-intelligence metaheuristic, applied to the classic
**Traveling Salesman Problem (TSP)**: find the shortest closed route that
visits every city exactly once.

## Why it's interesting

Real ants find the shortest path between their nest and a food source purely
through local behavior: as they walk, they deposit pheromone, and ants
statistically prefer paths with stronger pheromone. Shorter paths get
reinforced faster (because ants complete round-trips on them more often),
which creates a positive feedback loop that converges on near-optimal
paths — with no central coordination or global map.

This project simulates that process to solve TSP, an NP-hard combinatorial
optimization problem, and shows the colony's tour length improving
iteration-by-iteration until it beats a classic greedy baseline
(Nearest-Neighbor) by a wide margin.

## Tech stack and key concepts

- **Language/libraries:** Python, NumPy (vectorized distance math), Matplotlib (plots)
- **Algorithm:** Ant Colony Optimization (Ant System variant with elitist reinforcement)
  - Pheromone matrix (`tau`) updated via evaporation + deposit
  - Heuristic desirability (`eta = 1/distance`)
  - Random-proportional transition rule: `P(i->j) ∝ tau_ij^alpha * eta_ij^beta`
  - Elitist strategy: extra pheromone boost for the best-ever tour to speed convergence
- **Baseline comparison:** Nearest-Neighbor greedy heuristic
- **Concepts demonstrated:** swarm intelligence, exploration vs. exploitation,
  stochastic search, metaheuristics for NP-hard problems

## Installation

```bash
pip install -r requirements.txt
```

## How to run

```bash
python3 aco_tsp.py
```

This generates 30 random cities (fixed random seed for reproducibility),
runs the Nearest-Neighbor baseline, then runs ACO for 200 iterations with
30 ants per iteration. All plots and a text summary are written to `output/`.

### Tuning

Open `aco_tsp.py` and adjust the `AntColonyOptimizer(...)` parameters in `main()`:

| Parameter | Meaning | Default |
|---|---|---|
| `n_ants` | ants per iteration | 30 |
| `n_iterations` | colony iterations | 200 |
| `alpha` | pheromone trail influence | 1.0 |
| `beta` | distance-heuristic influence | 3.0 |
| `evaporation` | fraction of pheromone lost per iteration | 0.5 |
| `q` | pheromone deposit constant | 100.0 |
| `elitist_weight` | extra pheromone for best-ever tour | 2.0 |

## Example output

```
Generating 30 random cities (seed=42)...

Running Nearest-Neighbor baseline...
  Nearest-Neighbor tour length: 548.39

Running Ant Colony Optimization...
  iteration    0/200  best length so far: 551.77
  iteration   20/200  best length so far: 453.23
  iteration   40/200  best length so far: 452.06
  ...
  iteration  199/200  best length so far: 452.06

===== RESULTS =====
Nearest-Neighbor length : 548.39
ACO best length         : 452.06
Improvement over NN     : 17.6%
```

The colony finds a route **17.6% shorter** than the greedy Nearest-Neighbor
baseline, converging within roughly 40 iterations.

Generated files in `output/`:
- `nearest_neighbor_route.png` — the greedy baseline route
- `aco_best_route.png` — the best route ACO discovered
- `convergence.png` — best tour length vs. iteration, with the NN baseline as a reference line
- `summary.txt` — plain-text run summary

## How it works

1. **Setup** — 30 random 2D points ("cities") are generated, and the full
   pairwise Euclidean distance matrix is precomputed.
2. **Baseline** — Nearest-Neighbor builds a quick greedy tour by always
   hopping to the closest unvisited city, giving us something to beat.
3. **Pheromone initialization** — every edge between cities starts with a
   small, equal amount of pheromone.
4. **Iteration loop** (repeated `n_iterations` times):
   - Each of `n_ants` ants starts at a random city and builds a complete
     tour one step at a time. At each step, the next city is chosen
     probabilistically, weighted by `pheromone^alpha * (1/distance)^beta` —
     ants prefer nearby cities *and* cities linked by strong pheromone trails.
   - After every ant finishes, all pheromone evaporates by a fixed fraction
     (`evaporation`), so stale information fades over time.
   - Each ant then deposits pheromone on the edges of its own tour,
     proportional to `q / tour_length` — shorter tours deposit more
     pheromone, reinforcing good routes.
   - An elitist bonus adds extra pheromone to the single best tour found so
     far across the whole run, which accelerates convergence.
5. **Convergence** — over successive iterations, pheromone concentrates on
   edges belonging to good tours, and the ants' collective behavior
   converges toward a near-optimal route, visualized in `convergence.png`.

No API keys or external services are needed — the whole simulation runs
locally and deterministically (fixed random seed).
