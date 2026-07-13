"""
generate_data.py
-----------------
Generates a synthetic MovieLens-style ratings dataset.

We simulate 300 users and 120 movies (across 8 genres). Each user and
each movie is given a hidden ("latent") preference vector. A rating is
produced from the dot product of those vectors plus user/movie bias
terms and a little Gaussian noise -- this mimics how real-world rating
data tends to behave, and it means a matrix-factorization model has a
genuine underlying signal to recover.

Run directly to regenerate data/ratings.csv, data/movies.csv:
    python generate_data.py
"""

import csv
import random

import numpy as np

RNG_SEED = 42
N_USERS = 300
N_MOVIES = 120
N_LATENT = 6              # "true" hidden taste dimensions used to generate data
RATINGS_PER_USER_RANGE = (15, 45)  # each user rates a random subset of movies

GENRES = [
    "Action", "Comedy", "Drama", "Sci-Fi",
    "Romance", "Horror", "Documentary", "Animation",
]

MOVIE_ADJECTIVES = [
    "Silent", "Last", "Hidden", "Golden", "Broken", "Distant", "Eternal",
    "Crimson", "Forgotten", "Electric", "Midnight", "Velvet", "Frozen",
    "Ancient", "Neon",
]
MOVIE_NOUNS = [
    "Horizon", "Symphony", "Voyage", "Kingdom", "Machine", "Garden",
    "River", "Signal", "Empire", "Mirror", "Storm", "Harbor", "Echo",
    "Labyrinth", "Compass",
]


def make_movie_titles(n, rng):
    titles = set()
    while len(titles) < n:
        title = f"The {rng.choice(MOVIE_ADJECTIVES)} {rng.choice(MOVIE_NOUNS)}"
        titles.add(title)
    return list(titles)


def main():
    rng = np.random.default_rng(RNG_SEED)
    py_rng = random.Random(RNG_SEED)

    # --- Latent taste vectors -------------------------------------------------
    user_latent = rng.normal(0, 1, size=(N_USERS, N_LATENT))
    movie_latent = rng.normal(0, 1, size=(N_MOVIES, N_LATENT))

    # --- Bias terms (some users rate high, some movies are just better) -------
    user_bias = rng.normal(0, 0.6, size=N_USERS)
    movie_bias = rng.normal(0, 0.6, size=N_MOVIES)
    global_mean = 3.4  # average rating on a 1-5 scale

    # --- Movie metadata ---------------------------------------------------
    titles = make_movie_titles(N_MOVIES, py_rng)
    genres = [py_rng.choice(GENRES) for _ in range(N_MOVIES)]

    movies_path = "data/movies.csv"
    with open(movies_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["movie_id", "title", "genre"])
        for mid, (title, genre) in enumerate(zip(titles, genres)):
            writer.writerow([mid, title, genre])

    # --- Ratings ------------------------------------------------------------
    ratings_path = "data/ratings.csv"
    rows = []
    for uid in range(N_USERS):
        n_rated = py_rng.randint(*RATINGS_PER_USER_RANGE)
        movie_ids = py_rng.sample(range(N_MOVIES), n_rated)
        for mid in movie_ids:
            score = (
                global_mean
                + user_bias[uid]
                + movie_bias[mid]
                + np.dot(user_latent[uid], movie_latent[mid])
            )
            score += rng.normal(0, 0.4)  # noise
            score = int(round(np.clip(score, 1, 5)))
            rows.append((uid, mid, score))

    with open(ratings_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "movie_id", "rating"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} ratings for {N_USERS} users x {N_MOVIES} movies")
    print(f"-> {ratings_path}")
    print(f"-> {movies_path}")


if __name__ == "__main__":
    main()
