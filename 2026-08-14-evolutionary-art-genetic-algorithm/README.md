# Evolutionary Art — Genetic Algorithm Image Approximation

An AI that "paints" through natural selection instead of gradient descent.
A population of candidate images — each built from ~50 semi-transparent
colored triangles — competes generation after generation to best resemble
a target image. Selection, crossover, and mutation gradually sculpt random
polygon soup into a recognizable likeness, with no neural network and no
training data required.

This is **evolutionary computation**: an AI paradigm that predates deep
learning and solves optimization problems by mimicking Darwinian evolution.
It's a fun, visual way to see search and selection pressure do the work
that gradients usually do.

## Why it's interesting

- Zero training data, zero pretrained weights — the "model" is a raw
  population of random polygons that self-organizes purely from a fitness
  signal (pixel similarity to the target).
- The evolution is visible and satisfying: you can watch a blurry color
  smear resolve into a sun, mountains, and a moon over a few hundred
  generations.
- The same code works on *any* image — swap in a photo and watch it get
  triangulated into an abstract low-poly portrait.

## Tech stack & key concepts

- **Python** with `numpy` (vectorized genomes/fitness), `Pillow` (polygon
  rasterization + alpha compositing), `matplotlib` (evolution grid and
  fitness-curve plots).
- **Genetic algorithm** building blocks, all implemented from scratch:
  - *Genome*: flat float-array encoding of N triangles (3 vertices + RGBA
    color each).
  - *Fitness*: `1 - normalized MSE` between the rendered candidate and the
    target image, computed in RGB pixel space.
  - *Selection*: tournament selection (best of 4 random individuals).
  - *Crossover*: polygon-level — each triangle is inherited wholesale from
    one parent or the other, keeping shape+color genes coherent.
  - *Mutation*: per-gene jitter (local search) or full reset (escape local
    optima), each with independent low probability.
  - *Elitism*: the top fraction of each generation survives unchanged.

## Installation

```bash
pip install -r requirements.txt
```

## How to run

Evolve the built-in synthetic target (a sun/mountain/moon scene, no
external files needed):

```bash
python evolve_art.py
```

Evolve towards your own image instead:

```bash
python evolve_art.py --target path/to/your_image.jpg --generations 500
```

Tune the algorithm:

```bash
python evolve_art.py \
  --generations 400 \
  --population 30 \
  --polygons 50 \
  --elite-frac 0.1 \
  --size 96 \
  --seed 42 \
  --outdir output
```

All outputs land in `output/`:

- `target.png` — the image being approximated
- `best_gen_XXXX.png` — snapshots of the best individual at key generations
- `evolution_grid.png` — a contact sheet of those snapshots side-by-side
- `fitness_curve.png` — best fitness vs. generation
- `best_final.png` — the final evolved artwork

## Example output

Running the defaults (`--generations 400 --population 30 --polygons 50 --size 96`)
on the built-in target produced:

```
Evolving 50-polygon paintings, population=30, generations=400, image size=96x96
Gen    0/400  best_fitness=0.7215  elapsed=  0.1s
Gen   40/400  best_fitness=0.9303  elapsed=  1.7s
Gen   80/400  best_fitness=0.9675  elapsed=  3.2s
Gen  120/400  best_fitness=0.9815  elapsed=  4.8s
Gen  160/400  best_fitness=0.9845  elapsed=  6.4s
Gen  200/400  best_fitness=0.9859  elapsed=  8.0s
Gen  240/400  best_fitness=0.9876  elapsed= 10.1s
Gen  280/400  best_fitness=0.9880  elapsed= 11.8s
Gen  320/400  best_fitness=0.9886  elapsed= 13.4s
Gen  360/400  best_fitness=0.9889  elapsed= 15.0s
Gen  400/400  best_fitness=0.9894  elapsed= 16.5s

Final fitness: 0.9894
Outputs written to: output
```

Fitness (1 minus normalized pixel MSE) climbs from ~0.72 at generation 0
(pure random polygons) to ~0.99 by generation 400 — a near-perfect
reconstruction using only 50 triangles. The whole run takes under 20
seconds on a single CPU core.

`evolution_grid.png` shows the progression: at generation 0 the canvas is
random colored noise; by generation ~100 the sun, mountains, and moon are
already clearly recognizable; by generation 400 edges have sharpened and
colors have converged closely to the target.

## How it works

1. **Initialize**: create a population of individuals, each a random
   genome of N triangles (random position, shape, and RGBA color).
2. **Render & score**: rasterize every individual into an image and compute
   its fitness as similarity (1 − normalized MSE) to the target image.
3. **Select parents**: run tournament selection — sample a handful of
   individuals at random and keep the fittest — to pick two parents,
   repeated until a new population is filled.
4. **Crossover**: build each child by taking every triangle from one parent
   or the other (coin flip per triangle), preserving whole shapes rather
   than corrupting them at the pixel/gene level.
5. **Mutate**: with low probability, nudge a gene's value slightly (fine
   local search) or replace it outright with a new random value (keeps
   diversity and helps escape local optima).
6. **Elitism**: carry the best individuals from each generation forward
   unchanged, so progress is never lost.
7. **Repeat** for the requested number of generations, snapshotting the
   best individual periodically to visualize the evolution.

No image datasets, gradients, or pretrained models are used — the target
image is the *only* supervision signal, and it only ever needs to be
compared pixel-by-pixel to a candidate, never "learned from" in the deep
learning sense.
