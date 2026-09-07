"""
Active Learning with Uncertainty Sampling
==========================================

The core question this project answers: if labeling data is expensive
(think: a radiologist labeling scans, or a lawyer labeling contracts),
can we reach the same model accuracy with FEWER labeled examples by being
smart about which examples we ask a human to label next?

This script builds a simple "pool-based active learning" loop and compares
two labeling strategies head-to-head, using the exact same model, the exact
same starting point, and the exact same labeling budget per round:

1. RANDOM SAMPLING (the baseline) — at every round, pick the next batch of
   points to label completely at random from the unlabeled pool.
2. UNCERTAINTY SAMPLING (the active learning strategy) — at every round,
   ask the model which unlabeled points it is LEAST confident about
   (using margin sampling: the gap between the top two predicted class
   probabilities) and label those first. The intuition: points the model
   is already confident about teach it little; points near its decision
   boundary are the most informative to label next.

We run both strategies for the same number of rounds on the same dataset
and the same random seed for the initial split, then plot test-set accuracy
vs. number of labeled examples for each. If active learning is working,
its curve should sit above the random curve — i.e. it reaches a given
accuracy with fewer labels, or reaches higher accuracy with the same labels.

Tech stack: scikit-learn (RandomForestClassifier, breast cancer / digits
datasets), numpy, matplotlib. No API keys or internet access required.
"""

import argparse
import csv
import sys

import matplotlib

matplotlib.use("Agg")  # headless-safe backend, no display needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def load_dataset(name: str):
    """Load a built-in scikit-learn classification dataset by name."""
    if name == "breast_cancer":
        data = load_breast_cancer()
    elif name == "digits":
        data = load_digits()
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return data.data, data.target


def margin_uncertainty(probs: np.ndarray) -> np.ndarray:
    """
    Compute a per-sample "uncertainty score" from predicted class
    probabilities using margin sampling: the smaller the gap between the
    top two predicted probabilities, the more uncertain the model is
    about that sample. We return NEGATIVE margin so that higher score =
    more uncertain (convenient for sorting/argmax later).
    """
    sorted_probs = np.sort(probs, axis=1)  # ascending, per row
    top1 = sorted_probs[:, -1]
    top2 = sorted_probs[:, -2] if probs.shape[1] > 1 else np.zeros_like(top1)
    margin = top1 - top2
    return -margin  # higher = more uncertain


def run_active_learning(
    X_train,
    y_train,
    X_test,
    y_test,
    strategy: str,
    n_initial: int,
    n_query: int,
    n_rounds: int,
    seed: int,
):
    """
    Simulate a pool-based active learning loop.

    strategy: "random" or "uncertainty"
    n_initial: size of the small labeled "seed" set we start with
    n_query: how many new points we label per round
    n_rounds: how many rounds of querying to run after the seed round

    Returns a list of (n_labeled, test_accuracy) tuples, one per round,
    tracing out the learning curve for this strategy.
    """
    rng = np.random.RandomState(seed)

    n_samples = X_train.shape[0]
    all_indices = np.arange(n_samples)
    rng.shuffle(all_indices)

    # Start with a small randomly chosen labeled "seed" set. Both
    # strategies use the SAME seed set (same rng/seed) so the comparison
    # is fair — the only thing that differs is which points get queried
    # in later rounds.
    labeled_idx = list(all_indices[:n_initial])
    unlabeled_idx = list(all_indices[n_initial:])

    history = []

    model = RandomForestClassifier(n_estimators=200, random_state=seed)

    for round_num in range(n_rounds + 1):  # +1 to include the seed round
        # Train on everything labeled so far.
        model.fit(X_train[labeled_idx], y_train[labeled_idx])

        # Evaluate on the held-out test set.
        acc = model.score(X_test, y_test)
        history.append((len(labeled_idx), acc))

        # Stop once we've exhausted the unlabeled pool or hit the round cap.
        if round_num == n_rounds or len(unlabeled_idx) == 0:
            break

        batch_size = min(n_query, len(unlabeled_idx))

        if strategy == "random":
            chosen = rng.choice(len(unlabeled_idx), size=batch_size, replace=False)
        elif strategy == "uncertainty":
            # Ask the current model how confident it is about every
            # remaining unlabeled point, then label the most uncertain ones.
            probs = model.predict_proba(X_train[unlabeled_idx])
            scores = margin_uncertainty(probs)
            # argsort descending by uncertainty score, take top batch_size
            chosen = np.argsort(-scores)[:batch_size]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Move the chosen points from the unlabeled pool to the labeled set.
        chosen = sorted(chosen, reverse=True)  # delete from the end first
        for idx_pos in chosen:
            labeled_idx.append(unlabeled_idx.pop(idx_pos))

    return history


def main():
    parser = argparse.ArgumentParser(
        description="Compare random sampling vs. uncertainty-based active learning."
    )
    parser.add_argument(
        "--dataset",
        choices=["breast_cancer", "digits"],
        default="breast_cancer",
        help="Which built-in scikit-learn dataset to use (default: breast_cancer)",
    )
    parser.add_argument("--n-initial", type=int, default=10, help="Seed labeled set size")
    parser.add_argument("--n-query", type=int, default=5, help="Labels queried per round")
    parser.add_argument("--n-rounds", type=int, default=20, help="Number of query rounds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir", default=".", help="Where to write the plot and CSV results"
    )
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}")
    X, y = load_dataset(args.dataset)

    # Hold out a fixed test set that NEITHER strategy ever gets to label.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=args.seed, stratify=y
    )
    print(f"Train pool size: {len(X_train)}   Test set size: {len(X_test)}")
    print(
        f"Seed labels: {args.n_initial}   Query batch: {args.n_query}   "
        f"Rounds: {args.n_rounds}\n"
    )

    results = {}
    for strategy in ["random", "uncertainty"]:
        print(f"Running strategy: {strategy} ...")
        history = run_active_learning(
            X_train,
            y_train,
            X_test,
            y_test,
            strategy=strategy,
            n_initial=args.n_initial,
            n_query=args.n_query,
            n_rounds=args.n_rounds,
            seed=args.seed,
        )
        results[strategy] = history
        final_n, final_acc = history[-1]
        print(f"  -> final: {final_n} labels, test accuracy = {final_acc:.4f}")

    # --- Summarize how many labels each strategy needed to reach a target
    # accuracy, which is the real-world payoff of active learning. ---
    print("\nLabels needed to first reach various accuracy targets:")
    best_random_acc = results["random"][-1][1]
    best_uncertainty_acc = results["uncertainty"][-1][1]
    targets = sorted(
        {round(t, 2) for t in np.arange(0.80, min(best_random_acc, best_uncertainty_acc) + 0.001, 0.02)}
    )
    for target in targets:
        n_for = {}
        for strategy, history in results.items():
            n_for[strategy] = next(
                (n for n, acc in history if acc >= target), None
            )
        r, u = n_for["random"], n_for["uncertainty"]
        if r is not None and u is not None:
            print(f"  accuracy >= {target:.2f}:  random needs {r:>4} labels,  uncertainty needs {u:>4} labels")

    # --- Save results to CSV for later inspection ---
    csv_path = f"{args.output_dir}/active_learning_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "n_labeled", "test_accuracy"])
        for strategy, history in results.items():
            for n_labeled, acc in history:
                writer.writerow([strategy, n_labeled, f"{acc:.4f}"])
    print(f"\nSaved raw results to {csv_path}")

    # --- Plot the learning curves ---
    plt.figure(figsize=(8, 5))
    colors = {"random": "tab:gray", "uncertainty": "tab:red"}
    labels = {"random": "Random sampling (baseline)", "uncertainty": "Uncertainty sampling (active learning)"}
    for strategy, history in results.items():
        n_vals = [n for n, _ in history]
        acc_vals = [acc for _, acc in history]
        plt.plot(
            n_vals, acc_vals, marker="o", markersize=4,
            label=labels[strategy], color=colors[strategy],
        )
    plt.xlabel("Number of labeled training examples")
    plt.ylabel("Test set accuracy")
    plt.title(f"Active Learning vs. Random Sampling ({args.dataset})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = f"{args.output_dir}/learning_curves.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved learning curve plot to {plot_path}")


if __name__ == "__main__":
    sys.exit(main())
