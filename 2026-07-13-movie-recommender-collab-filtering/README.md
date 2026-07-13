# Mini Recommendation Engine — Collaborative Filtering (Matrix Factorization)

**Category:** Machine Learning model
**Date:** 2026-07-13

A from-scratch implementation of the matrix-factorization algorithm that
powered the winning entries of the **Netflix Prize** — the same core idea
behind modern collaborative-filtering recommenders at Netflix, Spotify,
and Amazon. Given nothing but a sparse table of `(user, movie, rating)`
triples, the model learns hidden "taste" vectors for every user and every
movie, then uses them to predict how a user would rate a movie they
haven't seen yet.

No external APIs or downloads required — a realistic synthetic
MovieLens-style dataset (300 users, 120 movies, ~9,000 ratings) is
generated locally with a genuine latent structure for the model to
recover.

## Why It's Interesting

Collaborative filtering is one of the most commercially important ML
techniques ever built, yet the core algorithm is compact enough to
implement in under 200 lines of NumPy — no deep learning framework
needed. This project shows the full pipeline end to end: synthetic data
generation with a known ground-truth signal, stochastic gradient
descent optimization, a fair baseline for comparison, and real top-N
recommendations for a sample user.

## Tech Stack & Key Concepts

- **NumPy** — vectorized linear algebra for the factor matrices
- **Matrix Factorization (SVD-style)** — `r_hat(u,i) = mu + b_u + b_i + p_u · q_i`
- **Stochastic Gradient Descent** — per-rating updates with L2 regularization
- **Train/test split** — stratified per user to avoid cold-start leakage
- **RMSE evaluation** against a movie-average baseline
- **Matplotlib** (optional) — training curve visualization

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

```bash
# 1. Generate the synthetic dataset (writes data/ratings.csv, data/movies.csv)
python generate_data.py

# 2. Train the model and print recommendations
python recommender.py

# Optional flags
python recommender.py --factors 12 --epochs 40 --lr 0.01 --reg 0.05 --demo-user 7 --top-n 10
```

## Example Output

```
Loaded 9235 ratings | 300 users | 120 movies
Train: 7510 ratings | Test: 1725 ratings

Baseline (movie-average) RMSE: 1.5498

epoch   1/30  train_rmse=1.5475  val_rmse=1.5299
epoch   5/30  train_rmse=1.4571  val_rmse=1.5136
epoch  10/30  train_rmse=1.3919  val_rmse=1.4831
epoch  15/30  train_rmse=1.1677  val_rmse=1.3347
epoch  20/30  train_rmse=0.9526  val_rmse=1.1872
epoch  30/30  train_rmse=0.6688  val_rmse=0.9740

Training finished in 1.6s
Final model RMSE: 0.9740  (baseline was 1.5498)
Improvement over baseline: 37.1%

Saved training curve -> training_curve.png

--- Sample: what user 0 has already rated highly ---
  5.0/5  The Broken Compass  (Action)
  5.0/5  The Golden Voyage  (Romance)
  5.0/5  The Last Mirror  (Action)
  5.0/5  The Eternal Voyage  (Drama)
  5.0/5  The Eternal River  (Horror)

--- Top 5 recommendations for user 0 ---
  predicted 5.00/5  The Frozen Labyrinth  (Action)
  predicted 5.00/5  The Last Garden  (Drama)
  predicted 5.00/5  The Broken Symphony  (Romance)
  predicted 5.00/5  The Hidden River  (Comedy)
  predicted 5.00/5  The Forgotten Voyage  (Action)
```

## How It Works

1. **Data generation** (`generate_data.py`) assigns every user and every
   movie a hidden 6-dimensional "taste" vector plus a bias term, then
   synthesizes ratings as `global_mean + user_bias + movie_bias +
   dot(user_vector, movie_vector) + noise`, clipped to 1-5. This gives the
   recommender a real latent signal to learn — just like real rating data
   is believed to arise from unobserved taste/trait dimensions.

2. **Baseline**: predicting each movie's average training rating already
   beats "guess the global average," so it's a fair bar for the learned
   model to clear.

3. **Model training** (`recommender.py`): `MatrixFactorizationRecommender`
   learns two smaller matrices — `P` (users × factors) and `Q` (movies ×
   factors) — plus bias vectors, such that their reconstruction
   approximates the observed ratings. Each epoch shuffles the training
   ratings and applies one SGD step per rating:

   ```
   error = actual_rating - predicted_rating
   b_u  += lr * (error - reg * b_u)
   b_i  += lr * (error - reg * b_i)
   P[u] += lr * (error * Q[i] - reg * P[u])
   Q[i] += lr * (error * P[u] - reg * Q[i])
   ```

   The regularization term (`reg`) keeps factors small and prevents
   overfitting to the training ratings.

4. **Evaluation**: RMSE is tracked on a held-out test split (20% of each
   user's ratings) every epoch, so you can watch the model close the gap
   with the baseline and see the training curve in `training_curve.png`.

5. **Recommendation**: for a chosen user, the model scores every movie
   they haven't rated and returns the top-N by predicted rating (clipped
   to the valid 1-5 range, since raw dot products can drift slightly
   outside it after enough training).

## Files

- `generate_data.py` — synthetic dataset generator
- `recommender.py` — matrix factorization model, training loop, evaluation, recommendations
- `data/ratings.csv`, `data/movies.csv` — sample generated dataset (regeneratable)
- `requirements.txt` — pinned dependencies
