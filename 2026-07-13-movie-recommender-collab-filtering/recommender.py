"""
recommender.py
----------------
Mini Recommendation Engine — Collaborative Filtering via Matrix Factorization

Implements a from-scratch matrix-factorization recommender (the core idea
behind the original "Netflix Prize" algorithms and modern implicit/explicit
feedback recommenders). No scikit-learn model is used for the core
algorithm -- just NumPy and stochastic gradient descent -- so the whole
learning process is transparent.

Model
-----
For user u and movie i, the predicted rating is:

    r_hat(u, i) = mu + b_u + b_i + p_u . q_i

where:
    mu   = global average rating
    b_u  = user bias (how generous/harsh this user tends to rate)
    b_i  = movie bias (how well-loved this movie tends to be)
    p_u  = user's latent taste vector   (learned, length K)
    q_i  = movie's latent trait vector  (learned, length K)

Training minimizes regularized squared error via SGD:

    L = sum_{(u,i) in train} (r_ui - r_hat(u,i))^2
        + lambda * (||p_u||^2 + ||q_i||^2 + b_u^2 + b_i^2)

Usage
-----
    python recommender.py
    python recommender.py --factors 12 --epochs 40 --lr 0.01
"""

import argparse
import csv
import os
import time

import numpy as np


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_ratings(path):
    """Read ratings.csv into a list of (user_id, movie_id, rating) ints."""
    ratings = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ratings.append(
                (int(row["user_id"]), int(row["movie_id"]), float(row["rating"]))
            )
    return ratings


def load_movies(path):
    """Read movies.csv into {movie_id: (title, genre)}."""
    movies = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            movies[int(row["movie_id"])] = (row["title"], row["genre"])
    return movies


def train_test_split(ratings, test_frac=0.2, seed=42):
    """Simple random split, stratified per user so every user appears in
    training (otherwise cold-start users would be untrainable)."""
    rng = np.random.default_rng(seed)
    by_user = {}
    for r in ratings:
        by_user.setdefault(r[0], []).append(r)

    train, test = [], []
    for uid, rows in by_user.items():
        rng.shuffle(rows)
        n_test = max(1, int(len(rows) * test_frac)) if len(rows) > 4 else 0
        test.extend(rows[:n_test])
        train.extend(rows[n_test:])
    return train, test


# --------------------------------------------------------------------------
# Matrix Factorization model (SGD, from scratch)
# --------------------------------------------------------------------------
class MatrixFactorizationRecommender:
    def __init__(self, n_users, n_movies, n_factors=10, lr=0.01, reg=0.05, seed=42):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg

        rng = np.random.default_rng(seed)
        # Small random init keeps early predictions near the global mean.
        self.P = rng.normal(0, 0.1, size=(n_users, n_factors))   # user factors
        self.Q = rng.normal(0, 0.1, size=(n_movies, n_factors))  # movie factors
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_movies)
        self.mu = 0.0

    def predict(self, u, i):
        return self.mu + self.b_u[u] + self.b_i[i] + self.P[u] @ self.Q[i]

    def fit(self, train, val=None, epochs=30, verbose=True):
        self.mu = np.mean([r for _, _, r in train])
        history = []

        for epoch in range(1, epochs + 1):
            np.random.default_rng(epoch).shuffle(train)  # reshuffle each epoch
            sq_err_sum = 0.0

            for u, i, r in train:
                pred = self.predict(u, i)
                err = r - pred

                # --- SGD updates (gradient of the regularized loss) ---
                b_u_old, b_i_old = self.b_u[u], self.b_i[i]
                p_u_old, q_i_old = self.P[u].copy(), self.Q[i].copy()

                self.b_u[u] += self.lr * (err - self.reg * b_u_old)
                self.b_i[i] += self.lr * (err - self.reg * b_i_old)
                self.P[u] += self.lr * (err * q_i_old - self.reg * p_u_old)
                self.Q[i] += self.lr * (err * p_u_old - self.reg * q_i_old)

                sq_err_sum += err ** 2

            train_rmse = np.sqrt(sq_err_sum / len(train))
            val_rmse = self.evaluate(val) if val else None
            history.append((epoch, train_rmse, val_rmse))

            if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
                msg = f"epoch {epoch:3d}/{epochs}  train_rmse={train_rmse:.4f}"
                if val_rmse is not None:
                    msg += f"  val_rmse={val_rmse:.4f}"
                print(msg)

        return history

    def evaluate(self, data):
        if not data:
            return None
        sq_err_sum = sum((r - self.predict(u, i)) ** 2 for u, i, r in data)
        return float(np.sqrt(sq_err_sum / len(data)))

    def recommend(self, user_id, rated_movie_ids, movies, top_n=5):
        """Return the top_n unrated movies with the highest predicted rating.

        Raw dot-product predictions can drift slightly outside the valid
        1-5 rating scale (especially with more training epochs), so we
        clip for display -- same as production recommenders do.
        """
        scores = []
        for movie_id in movies:
            if movie_id in rated_movie_ids:
                continue
            raw = self.predict(user_id, movie_id)
            scores.append((movie_id, float(np.clip(raw, 1.0, 5.0))))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]


# --------------------------------------------------------------------------
# Baseline for comparison: predict the movie's global average rating
# --------------------------------------------------------------------------
def baseline_rmse(train, test):
    movie_avg = {}
    counts = {}
    global_sum, global_n = 0.0, 0
    for _, i, r in train:
        movie_avg[i] = movie_avg.get(i, 0.0) + r
        counts[i] = counts.get(i, 0) + 1
        global_sum += r
        global_n += 1
    for i in movie_avg:
        movie_avg[i] /= counts[i]
    global_mean = global_sum / global_n

    sq_err_sum = 0.0
    for _, i, r in test:
        pred = movie_avg.get(i, global_mean)
        sq_err_sum += (r - pred) ** 2
    return np.sqrt(sq_err_sum / len(test))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Mini collaborative-filtering recommender")
    parser.add_argument("--factors", type=int, default=10, help="number of latent factors")
    parser.add_argument("--epochs", type=int, default=30, help="training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--reg", type=float, default=0.05, help="L2 regularization strength")
    parser.add_argument("--demo-user", type=int, default=0, help="user id to print recommendations for")
    parser.add_argument("--top-n", type=int, default=5, help="how many recommendations to show")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    ratings_path = os.path.join(data_dir, "ratings.csv")
    movies_path = os.path.join(data_dir, "movies.csv")

    if not os.path.exists(ratings_path):
        raise SystemExit(
            "data/ratings.csv not found. Run `python generate_data.py` first."
        )

    ratings = load_ratings(ratings_path)
    movies = load_movies(movies_path)
    n_users = max(r[0] for r in ratings) + 1
    n_movies = max(r[1] for r in ratings) + 1

    print(f"Loaded {len(ratings)} ratings | {n_users} users | {n_movies} movies")

    train, test = train_test_split(ratings, test_frac=0.2)
    print(f"Train: {len(train)} ratings | Test: {len(test)} ratings\n")

    # --- Baseline ---
    bl_rmse = baseline_rmse(train, test)
    print(f"Baseline (movie-average) RMSE: {bl_rmse:.4f}\n")

    # --- Train matrix factorization model ---
    model = MatrixFactorizationRecommender(
        n_users, n_movies, n_factors=args.factors, lr=args.lr, reg=args.reg
    )
    t0 = time.time()
    history = model.fit(train, val=test, epochs=args.epochs)
    elapsed = time.time() - t0

    final_rmse = history[-1][2]
    print(f"\nTraining finished in {elapsed:.1f}s")
    print(f"Final model RMSE: {final_rmse:.4f}  (baseline was {bl_rmse:.4f})")
    improvement = (bl_rmse - final_rmse) / bl_rmse * 100
    print(f"Improvement over baseline: {improvement:.1f}%\n")

    # --- Try to plot the training curve (optional, skipped if matplotlib missing) ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs_x = [h[0] for h in history]
        train_y = [h[1] for h in history]
        val_y = [h[2] for h in history]

        plt.figure(figsize=(7, 4.5))
        plt.plot(epochs_x, train_y, label="Train RMSE", linewidth=2)
        plt.plot(epochs_x, val_y, label="Validation RMSE", linewidth=2)
        plt.axhline(bl_rmse, color="gray", linestyle="--", label="Baseline RMSE")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.title("Matrix Factorization Training Curve")
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_curve.png")
        plt.savefig(out_path, dpi=120)
        print(f"Saved training curve -> {out_path}\n")
    except ImportError:
        print("matplotlib not installed; skipping training curve plot.\n")

    # --- Show recommendations for a demo user ---
    rated_by_user = {}
    for u, i, r in ratings:
        rated_by_user.setdefault(u, set()).add(i)

    demo_user = args.demo_user
    rated_ids = rated_by_user.get(demo_user, set())
    print(f"--- Sample: what {demo_user} has already rated highly ---")
    already_rated = sorted(
        [(i, r) for u, i, r in ratings if u == demo_user],
        key=lambda x: x[1],
        reverse=True,
    )[:5]
    for movie_id, r in already_rated:
        title, genre = movies[movie_id]
        print(f"  {r}/5  {title}  ({genre})")

    print(f"\n--- Top {args.top_n} recommendations for user {demo_user} ---")
    recs = model.recommend(demo_user, rated_ids, movies, top_n=args.top_n)
    for movie_id, score in recs:
        title, genre = movies[movie_id]
        print(f"  predicted {score:.2f}/5  {title}  ({genre})")


if __name__ == "__main__":
    main()
