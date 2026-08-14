"""
Evolutionary Art — Genetic Algorithm Image Approximation
==========================================================

An AI system that "paints" by evolution: a population of candidate images,
each made of a handful of semi-transparent colored polygons, competes to
best approximate a target image. Over many generations, selection,
crossover, and mutation sculpt random polygon soup into a recognizable
likeness of the target — no gradients, no backpropagation, just Darwinian
pressure applied to pixels.

This is a classic example of *evolutionary computation*, a branch of AI
that solves problems by mimicking natural selection instead of learning
from labeled data.

Usage:
    python evolve_art.py                       # evolve the built-in target
    python evolve_art.py --target path/to.png  # evolve any image you like
    python evolve_art.py --generations 800 --population 40 --polygons 60

Outputs (written to ./output/):
    target.png              the image being approximated
    best_gen_XXXX.png       snapshots of the best individual over time
    evolution_grid.png      a contact sheet of snapshots side-by-side
    fitness_curve.png       fitness (image similarity) vs. generation
    best_final.png          the final evolved artwork
"""

import argparse
import os
import random
import time

import numpy as np
from PIL import Image, ImageDraw
import matplotlib

matplotlib.use("Agg")  # headless rendering, no display needed
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Target image
# ---------------------------------------------------------------------------
def make_synthetic_target(size=(96, 96)):
    """
    Build a simple, colorful synthetic target image (no external dataset
    needed) so the project runs anywhere out of the box. Draws a sun,
    a triangle "mountain", and a circle "moon" — enough shape and color
    variety to make evolution visibly work.
    """
    img = Image.new("RGB", size, (20, 30, 60))  # night-sky blue background
    draw = ImageDraw.Draw(img)
    w, h = size

    # Glowing sun (yellow-orange circle, top-right)
    draw.ellipse([w * 0.55, h * 0.08, w * 0.85, h * 0.38], fill=(255, 200, 60))

    # Mountain (triangle, bottom half)
    draw.polygon(
        [(w * 0.05, h * 0.95), (w * 0.45, h * 0.35), (w * 0.75, h * 0.95)],
        fill=(90, 60, 110),
    )

    # Second, smaller mountain overlapping
    draw.polygon(
        [(w * 0.35, h * 0.95), (w * 0.65, h * 0.55), (w * 0.98, h * 0.95)],
        fill=(60, 40, 90),
    )

    # Moon-like accent circle (bottom-left), just for extra color variety
    draw.ellipse([w * 0.05, h * 0.60, w * 0.25, h * 0.80], fill=(230, 230, 250))

    return img


# ---------------------------------------------------------------------------
# 2. Genome representation and rendering
# ---------------------------------------------------------------------------
class Individual:
    """
    A candidate "painting": a fixed number of semi-transparent triangles.
    Each triangle is 3 (x, y) vertices + an (r, g, b, a) color, all stored
    as flat float genes in [0, 1] so crossover/mutation is genome-agnostic.
    """

    GENES_PER_POLY = 3 * 2 + 4  # 3 vertices * (x, y)  +  r, g, b, a

    def __init__(self, n_polygons, genes=None):
        self.n_polygons = n_polygons
        if genes is None:
            self.genes = np.random.rand(n_polygons * self.GENES_PER_POLY).astype(
                np.float32
            )
        else:
            self.genes = genes.copy()
        self._fitness = None  # cached fitness (lower render cost)

    def render(self, size):
        """Rasterize this genome into an RGB PIL Image of the given size."""
        w, h = size
        canvas = Image.new("RGB", size, (255, 255, 255))
        g = self.genes.reshape(self.n_polygons, self.GENES_PER_POLY)
        for row in g:
            xs = row[0:5:2] * w
            ys = row[1:6:2] * h
            r, gg, b, a = row[6:10]
            pts = list(zip(xs.tolist(), ys.tolist()))
            layer = Image.new("RGBA", size, (0, 0, 0, 0))
            ImageDraw.Draw(layer).polygon(
                pts, fill=(int(r * 255), int(gg * 255), int(b * 255), int(a * 200))
            )
            canvas = Image.alpha_composite(canvas.convert("RGBA"), layer).convert(
                "RGB"
            )
        return canvas

    def copy(self):
        clone = Individual(self.n_polygons, self.genes)
        clone._fitness = self._fitness
        return clone


# ---------------------------------------------------------------------------
# 3. Fitness
# ---------------------------------------------------------------------------
def fitness_of(individual, target_arr, size):
    """
    Fitness = similarity to target, in [0, 1], higher is better.
    Computed as 1 - normalized mean-squared-error over RGB pixels.
    """
    if individual._fitness is not None:
        return individual._fitness
    rendered = np.asarray(individual.render(size), dtype=np.float32)
    mse = np.mean((rendered - target_arr) ** 2)
    max_mse = 255.0 ** 2
    fit = 1.0 - (mse / max_mse)
    individual._fitness = fit
    return fit


# ---------------------------------------------------------------------------
# 4. Genetic operators
# ---------------------------------------------------------------------------
def tournament_select(pop, fits, k=4):
    """Pick the best of k random individuals — simple, effective selection."""
    idxs = random.sample(range(len(pop)), k)
    best = max(idxs, key=lambda i: fits[i])
    return pop[best]


def crossover(parent_a, parent_b, n_polygons):
    """
    Polygon-level crossover: each polygon (whole gene block) is inherited
    from one parent or the other, chosen independently at random. This
    keeps each polygon's genes coherent (a triangle's shape+color travel
    together) instead of mixing at the raw-float level.
    """
    ga = parent_a.genes.reshape(n_polygons, Individual.GENES_PER_POLY)
    gb = parent_b.genes.reshape(n_polygons, Individual.GENES_PER_POLY)
    mask = np.random.rand(n_polygons, 1) < 0.5
    child_genes = np.where(mask, ga, gb).reshape(-1).astype(np.float32)
    return Individual(n_polygons, child_genes)


def mutate(individual, n_polygons, rate=0.02, strength=0.25):
    """
    Two mutation modes, applied gene-by-gene with probability `rate`:
      - jitter: nudge the gene value by a random amount (local search)
      - reset:  replace the gene with a fresh random value (escape local optima)
    """
    genes = individual.genes
    mask = np.random.rand(genes.shape[0]) < rate
    n_mut = mask.sum()
    if n_mut == 0:
        return individual
    if random.random() < 0.8:
        noise = (np.random.rand(n_mut).astype(np.float32) - 0.5) * 2 * strength
        genes[mask] = np.clip(genes[mask] + noise, 0.0, 1.0)
    else:
        genes[mask] = np.random.rand(n_mut).astype(np.float32)
    individual._fitness = None
    return individual


# ---------------------------------------------------------------------------
# 5. Evolution loop
# ---------------------------------------------------------------------------
def evolve(target_img, generations, population_size, n_polygons, elite_frac, outdir):
    size = target_img.size
    target_arr = np.asarray(target_img.convert("RGB"), dtype=np.float32)

    population = [Individual(n_polygons) for _ in range(population_size)]
    n_elite = max(1, int(population_size * elite_frac))

    history = []
    snapshot_gens = sorted(
        set(
            [0]
            + [int(generations * f) for f in (0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)]
        )
    )

    start = time.time()
    for gen in range(generations + 1):
        fits = [fitness_of(ind, target_arr, size) for ind in population]
        best_idx = int(np.argmax(fits))
        best_fit = fits[best_idx]
        history.append(best_fit)

        if gen in snapshot_gens:
            population[best_idx].render(size).save(
                os.path.join(outdir, f"best_gen_{gen:04d}.png")
            )

        if gen % max(1, generations // 10) == 0 or gen == generations:
            elapsed = time.time() - start
            print(
                f"Gen {gen:4d}/{generations}  best_fitness={best_fit:.4f}  "
                f"elapsed={elapsed:5.1f}s"
            )

        if gen == generations:
            break

        # --- build next generation ---
        order = np.argsort(fits)[::-1]
        next_pop = [population[i].copy() for i in order[:n_elite]]  # elitism

        while len(next_pop) < population_size:
            parent_a = tournament_select(population, fits)
            parent_b = tournament_select(population, fits)
            child = crossover(parent_a, parent_b, n_polygons)
            child = mutate(child, n_polygons)
            next_pop.append(child)

        population = next_pop

    best_final = population[int(np.argmax([fitness_of(i, target_arr, size) for i in population]))]
    return best_final, history


# ---------------------------------------------------------------------------
# 6. Reporting: contact sheet + fitness curve
# ---------------------------------------------------------------------------
def make_evolution_grid(outdir, snapshot_gens):
    imgs = []
    labels = []
    for gen in snapshot_gens:
        path = os.path.join(outdir, f"best_gen_{gen:04d}.png")
        if os.path.exists(path):
            imgs.append(Image.open(path))
            labels.append(f"gen {gen}")
    if not imgs:
        return
    fig, axes = plt.subplots(1, len(imgs), figsize=(3 * len(imgs), 3.4))
    if len(imgs) == 1:
        axes = [axes]
    for ax, im, label in zip(axes, imgs, labels):
        ax.imshow(im)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    fig.suptitle("Evolution of the population's best individual", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "evolution_grid.png"), dpi=130)
    plt.close(fig)


def make_fitness_plot(outdir, history):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(history, color="#2b6cb0", linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness (1 - normalized MSE)")
    ax.set_title("Fitness over generations")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fitness_curve.png"), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", type=str, default=None, help="Path to a target image (optional)."
    )
    parser.add_argument("--generations", type=int, default=400)
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--polygons", type=int, default=50)
    parser.add_argument("--elite-frac", type=float, default=0.1)
    parser.add_argument("--size", type=int, default=96, help="Target image side length in px.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="output")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.outdir, exist_ok=True)

    if args.target:
        target_img = Image.open(args.target).convert("RGB").resize((args.size, args.size))
    else:
        target_img = make_synthetic_target((args.size, args.size))
    target_img.save(os.path.join(args.outdir, "target.png"))

    print(
        f"Evolving {args.polygons}-polygon paintings, population={args.population}, "
        f"generations={args.generations}, image size={args.size}x{args.size}"
    )

    best, history = evolve(
        target_img,
        generations=args.generations,
        population_size=args.population,
        n_polygons=args.polygons,
        elite_frac=args.elite_frac,
        outdir=args.outdir,
    )

    best.render(target_img.size).save(os.path.join(args.outdir, "best_final.png"))

    snapshot_gens = sorted(
        set(
            [0]
            + [
                int(args.generations * f)
                for f in (0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)
            ]
        )
    )
    make_evolution_grid(args.outdir, snapshot_gens)
    make_fitness_plot(args.outdir, history)

    print(f"\nFinal fitness: {history[-1]:.4f}")
    print(f"Outputs written to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
