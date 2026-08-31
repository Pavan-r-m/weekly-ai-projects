"""
Customer Churn Prediction — Ensemble Model Comparison
========================================================

Predicts whether a telecom/subscription customer will churn (cancel their
service) using three classic ML models, then compares them head-to-head on
accuracy, precision, recall, F1, and ROC-AUC. Also extracts feature
importance so we can explain *why* the best model thinks a customer will
leave.

Models compared:
    1. Logistic Regression   (linear baseline, fast & interpretable)
    2. Random Forest         (bagged trees, handles non-linearity well)
    3. Gradient Boosting     (sequential trees, usually the strongest here)

No external dataset or API key is required — a realistic synthetic churn
dataset is generated on the fly with controllable noise and real-world-like
feature correlations (e.g. high monthly charges + short tenure + support
calls => higher churn probability).

Run:
    python churn_predictor.py
    python churn_predictor.py --n-customers 5000 --seed 7

Outputs (written next to this script):
    - churn_dataset_sample.csv   (first 200 rows of generated data)
    - model_comparison.csv       (metrics table for all 3 models)
    - model_comparison.png       (bar chart: accuracy/precision/recall/F1/AUC)
    - roc_curves.png             (ROC curves for all 3 models)
    - feature_importance.png     (top features from the best model)
"""

import argparse
import warnings

import matplotlib
matplotlib.use("Agg")  # headless-safe backend, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Synthetic data generation
# ---------------------------------------------------------------------------
def generate_churn_dataset(n_customers: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic but realistic customer-churn dataset.

    Features are drawn from distributions loosely modeled on real telecom
    churn datasets (e.g. IBM Telco Customer Churn), and the churn label is
    generated from a logistic function of a weighted combination of the
    features plus noise, so there IS a learnable signal but it's not
    trivially perfect (mirrors real-world messiness).
    """
    rng = np.random.default_rng(seed)

    tenure_months = rng.gamma(shape=2.0, scale=12.0, size=n_customers).clip(0, 72)
    monthly_charges = rng.normal(70, 30, size=n_customers).clip(15, 150)
    total_charges = tenure_months * monthly_charges * rng.uniform(0.85, 1.05, n_customers)
    num_support_calls = rng.poisson(lam=1.5, size=n_customers)
    contract_type = rng.choice(
        ["month-to-month", "one-year", "two-year"], size=n_customers, p=[0.55, 0.25, 0.20]
    )
    has_tech_support = rng.choice([0, 1], size=n_customers, p=[0.6, 0.4])
    has_online_security = rng.choice([0, 1], size=n_customers, p=[0.65, 0.35])
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "None"], size=n_customers, p=[0.35, 0.45, 0.20]
    )
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n_customers,
    )
    num_dependents = rng.poisson(lam=0.7, size=n_customers).clip(0, 5)
    satisfaction_score = rng.normal(6.5, 2.0, size=n_customers).clip(0, 10)

    # --- Build churn probability as a weighted logistic combination -------
    contract_risk = np.select(
        [contract_type == "month-to-month", contract_type == "one-year", contract_type == "two-year"],
        [1.0, 0.3, 0.05],
    )
    electronic_check_risk = (payment_method == "Electronic check").astype(float)
    fiber_risk = (internet_service == "Fiber optic").astype(float) * 0.4

    z = (
        -1.8
        + 2.4 * contract_risk
        + 0.02 * (monthly_charges - 70)
        - 0.035 * tenure_months
        + 0.30 * num_support_calls
        + 1.0 * electronic_check_risk
        - 0.9 * has_tech_support
        - 0.7 * has_online_security
        + fiber_risk
        - 0.28 * (satisfaction_score - 6.5)
        + rng.normal(0, 0.6, size=n_customers)  # irreducible noise
    )
    churn_prob = 1 / (1 + np.exp(-z))
    churned = (rng.uniform(0, 1, size=n_customers) < churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "tenure_months": tenure_months.round(1),
            "monthly_charges": monthly_charges.round(2),
            "total_charges": total_charges.round(2),
            "num_support_calls": num_support_calls,
            "contract_type": contract_type,
            "has_tech_support": has_tech_support,
            "has_online_security": has_online_security,
            "internet_service": internet_service,
            "payment_method": payment_method,
            "num_dependents": num_dependents,
            "satisfaction_score": satisfaction_score.round(2),
            "churned": churned,
        }
    )
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns, leave numeric ones as-is."""
    categorical_cols = ["contract_type", "internet_service", "payment_method"]
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True)


# ---------------------------------------------------------------------------
# 2. Model training & evaluation
# ---------------------------------------------------------------------------
def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train three models and collect comparison metrics + fitted objects."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42
        ),
    }

    results = []
    fitted = {}
    roc_data = {}

    for name, model in models.items():
        # Logistic regression benefits from scaled features; tree models don't
        # need it but it doesn't hurt, so we use scaled data for LR and raw
        # (unscaled) data for the tree ensembles to keep feature importances
        # interpretable in original units.
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            proba = model.predict_proba(X_test_scaled)[:, 1]
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="roc_auc")
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")

        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_data[name] = (fpr, tpr)

        results.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, preds),
                "precision": precision_score(y_test, preds),
                "recall": recall_score(y_test, preds),
                "f1_score": f1_score(y_test, preds),
                "roc_auc": roc_auc_score(y_test, proba),
                "cv_roc_auc_mean": cv_scores.mean(),
                "cv_roc_auc_std": cv_scores.std(),
            }
        )
        fitted[name] = model

    return pd.DataFrame(results), fitted, roc_data, scaler


# ---------------------------------------------------------------------------
# 3. Plotting
# ---------------------------------------------------------------------------
def plot_model_comparison(results_df: pd.DataFrame, out_path: str):
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (_, row) in enumerate(results_df.iterrows()):
        values = [row[m] for m in metrics]
        ax.bar(x + i * width, values, width, label=row["model"])

    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Customer Churn Prediction")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc_curves(roc_data: dict, results_df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 7))
    for name, (fpr, tpr) in roc_data.items():
        auc = results_df.loc[results_df["model"] == name, "roc_auc"].values[0]
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Churn Prediction Models")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names, model_name, out_path: str, top_n: int = 10):
    """Plot the top-N most influential features for the given fitted model.

    Tree ensembles expose `feature_importances_` directly. Logistic
    Regression doesn't, but the absolute value of its standardized
    coefficients serves the same purpose (how much each feature moves the
    predicted log-odds of churn).
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        xlabel = "Importance"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        xlabel = "|Coefficient| (standardized)"
    else:
        return

    order = np.argsort(importances)[::-1][:top_n]
    top_features = np.array(feature_names)[order]
    top_importances = importances[order]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_features[::-1], top_importances[::-1], color="steelblue")
    ax.set_xlabel(xlabel)
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Customer Churn Prediction — Ensemble Comparison")
    parser.add_argument("--n-customers", type=int, default=3000, help="Number of synthetic customers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--test-size", type=float, default=0.25, help="Fraction of data held out for testing")
    args = parser.parse_args()

    print(f"Generating synthetic churn dataset ({args.n_customers} customers, seed={args.seed})...")
    df = generate_churn_dataset(n_customers=args.n_customers, seed=args.seed)
    print(f"  Churn rate: {df['churned'].mean():.1%}")

    df.head(200).to_csv("churn_dataset_sample.csv", index=False)
    print("  Saved sample -> churn_dataset_sample.csv")

    encoded = encode_features(df)
    X = encoded.drop(columns=["churned"])
    y = encoded["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

    print("\nTraining models: Logistic Regression, Random Forest, Gradient Boosting...")
    results_df, fitted_models, roc_data, _ = train_and_evaluate(X_train, X_test, y_train, y_test)

    results_df = results_df.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    results_df.to_csv("model_comparison.csv", index=False)

    print("\n=== Model Comparison (sorted by ROC-AUC) ===")
    print(results_df.round(4).to_string(index=False))

    plot_model_comparison(results_df, "model_comparison.png")
    plot_roc_curves(roc_data, results_df, "roc_curves.png")

    best_model_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_model_name]
    plot_feature_importance(best_model, X.columns, best_model_name, "feature_importance.png")

    print(f"\nBest model: {best_model_name} (ROC-AUC = {results_df.iloc[0]['roc_auc']:.4f})")
    print("Saved: model_comparison.csv, model_comparison.png, roc_curves.png, feature_importance.png")


if __name__ == "__main__":
    main()
