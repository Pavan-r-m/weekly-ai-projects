"""
digit_classifier.py
====================
Compares four different scikit-learn classification models -- including a
real neural network (MLPClassifier) -- on the task of recognizing
handwritten digits (0-9), then digs into *why* the best model still gets
things wrong.

Why this project is interesting
--------------------------------
Rather than training a single model and reporting one accuracy number, this
script treats model selection like a real ML workflow:
  1. Train four qualitatively different algorithms (linear, margin-based,
     ensemble/tree-based, and a neural network) on the same data/split.
  2. Compare them fairly on accuracy, macro-F1, and training time.
  3. Pick the winner and analyze *its* mistakes: a confusion matrix and a
     gallery of the actual misclassified digit images.
  4. Peek inside the neural network by visualizing the learned first-layer
     weights as tiny 8x8 "feature detector" images -- a classic way to build
     intuition for what a neural net's hidden units actually respond to.

Runs entirely offline (scikit-learn's bundled `load_digits` dataset, no
downloads) in well under a minute on a laptop CPU.
"""

import time

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

SEED = 42
np.random.seed(SEED)


# ----------------------------------------------------------------------------
# 1. Load & prepare data
# ----------------------------------------------------------------------------
def load_data():
    """Load sklearn's digits dataset (8x8 grayscale, 10 classes)."""
    digits = load_digits()
    X = digits.images.reshape(len(digits.images), -1).astype(np.float64)  # (N, 64)
    y = digits.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Standardize features -- helps SVM, MLP, and logistic regression converge
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test


# ----------------------------------------------------------------------------
# 2. Define the model zoo
# ----------------------------------------------------------------------------
def build_models():
    """Return a dict of {name: (estimator, needs_scaled_input)}."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=SEED),
            True,
        ),
        "SVM (RBF kernel)": (
            SVC(kernel="rbf", C=10, gamma="scale", random_state=SEED),
            True,
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=300, max_depth=None, random_state=SEED, n_jobs=-1
            ),
            False,  # tree-based models don't need scaling
        ),
        "Neural Net (MLP)": (
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                max_iter=500,
                random_state=SEED,
            ),
            True,
        ),
    }


# ----------------------------------------------------------------------------
# 3. Train & evaluate every model
# ----------------------------------------------------------------------------
def run_comparison(models, X_train, X_test, X_train_s, X_test_s, y_train, y_test):
    results = {}
    fitted = {}

    for name, (model, needs_scaling) in models.items():
        Xtr = X_train_s if needs_scaling else X_train
        Xte = X_test_s if needs_scaling else X_test

        start = time.perf_counter()
        model.fit(Xtr, y_train)
        train_time = time.perf_counter() - start

        y_pred = model.predict(Xte)
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")

        results[name] = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "train_time_sec": train_time,
            "y_pred": y_pred,
        }
        fitted[name] = model

        print(
            f"{name:22s} | accuracy={acc:.4f} | macro_f1={macro_f1:.4f} "
            f"| train_time={train_time:.3f}s"
        )

    return results, fitted


# ----------------------------------------------------------------------------
# 4. Visualization helpers
# ----------------------------------------------------------------------------
def plot_model_comparison(results, out_path="model_comparison.png"):
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    f1s = [results[n]["macro_f1"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, accs, width, label="Accuracy")
    ax.bar(x + width / 2, f1s, width, label="Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0.8, 1.0)
    ax.set_title("Model comparison on held-out test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved model comparison chart -> {out_path}")


def plot_confusion_matrix(y_true, y_pred, model_name, out_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix -- {model_name}")
    for i in range(10):
        for j in range(10):
            ax.text(
                j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8,
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved confusion matrix -> {out_path}")


def plot_misclassified(X_test, y_true, y_pred, out_path="misclassified.png", max_examples=12):
    """Gallery of digits the best model got wrong."""
    wrong_idx = np.where(y_true != y_pred)[0]
    n = min(max_examples, len(wrong_idx))

    if n == 0:
        print("No misclassified examples -- the model got everything right!")
        return

    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.4 * rows))
    axes = np.array(axes).reshape(-1)

    for ax_i, idx in enumerate(wrong_idx[:n]):
        img = X_test[idx].reshape(8, 8)
        axes[ax_i].imshow(img, cmap="gray")
        axes[ax_i].set_title(f"true={y_true[idx]} pred={y_pred[idx]}", fontsize=9)
        axes[ax_i].axis("off")

    for ax_i in range(n, len(axes)):
        axes[ax_i].axis("off")

    fig.suptitle("Misclassified digits (true vs predicted)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved misclassified gallery -> {out_path}")


def plot_mlp_weights(mlp_model, out_path="mlp_hidden_units.png", n_units=16):
    """
    Visualize the first-layer weights of the trained MLP as 8x8 images.
    Each hidden unit learns a small "feature detector" over the input pixels;
    plotting its weight vector reshaped to 8x8 gives an intuitive picture of
    what pattern that neuron responds to.
    """
    weights = mlp_model.coefs_[0]  # shape: (64 input pixels, n_hidden_units)
    n_units = min(n_units, weights.shape[1])

    cols = 4
    rows = int(np.ceil(n_units / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(n_units):
        unit_weights = weights[:, i].reshape(8, 8)
        axes[i].imshow(unit_weights, cmap="coolwarm")
        axes[i].set_title(f"unit {i}", fontsize=8)
        axes[i].axis("off")

    for i in range(n_units, len(axes)):
        axes[i].axis("off")

    fig.suptitle("First hidden layer weights of the MLP (learned feature detectors)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved MLP hidden unit visualization -> {out_path}")


# ----------------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------------
def main():
    print("Loading data (sklearn digits: 8x8 grayscale, 10 classes)...")
    X_train, X_test, X_train_s, X_test_s, y_train, y_test = load_data()
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}\n")

    models = build_models()
    print("Training & evaluating models...\n")
    results, fitted = run_comparison(
        models, X_train, X_test, X_train_s, X_test_s, y_train, y_test
    )

    # Pick the winner by accuracy
    best_name = max(results, key=lambda n: results[n]["accuracy"])
    best_pred = results[best_name]["y_pred"]
    print(f"\nBest model: {best_name} (accuracy={results[best_name]['accuracy']:.4f})\n")

    print(f"Classification report for {best_name}:")
    print(classification_report(y_test, best_pred, digits=3))

    plot_model_comparison(results)
    plot_confusion_matrix(y_test, best_pred, best_name)

    # Misclassified gallery uses the correctly-shaped (unscaled) test images
    plot_misclassified(X_test, y_test, best_pred)

    # Bonus: peek inside the neural network's learned features
    mlp_model = fitted["Neural Net (MLP)"]
    plot_mlp_weights(mlp_model)

    print("\nDone. See the generated PNG files for visual results.")


if __name__ == "__main__":
    main()
