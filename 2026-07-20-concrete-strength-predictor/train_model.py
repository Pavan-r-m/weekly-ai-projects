"""
train_model.py
----------------
Concrete Compressive Strength Predictor
=========================================
Trains and evaluates a Random Forest Regressor (with a Gradient Boosting
model for comparison) to predict concrete compressive strength (MPa) from
mix-design features and curing age. Produces:
    - console metrics (R^2, RMSE, MAE) for train/test + 5-fold CV
    - feature_importance.png  (which ingredients matter most)
    - predicted_vs_actual.png (model fit quality)
    - residuals.png           (error distribution / heteroscedasticity check)
    - a saved model file (model.joblib) so predict.py can reuse it

Why this project is interesting
--------------------------------
Concrete mix design is a real, high-stakes engineering problem: getting the
water-cement ratio and admixture balance wrong costs money and can affect
structural safety. This project shows the full applied-ML pipeline: data
generation -> EDA -> train/test split -> model comparison -> hyperparameter
tuning -> feature importance interpretation -> residual diagnostics.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use("Agg")  # headless-safe backend for saving PNGs without a display
import matplotlib.pyplot as plt
import joblib
import os

from generate_data import generate_dataset

FEATURE_COLS = [
    "cement", "blast_furnace_slag", "fly_ash", "water",
    "superplasticizer", "coarse_aggregate", "fine_aggregate", "age",
]
TARGET_COL = "compressive_strength"
OUTPUT_DIR = "outputs"


def load_data() -> pd.DataFrame:
    """Load dataset from data/concrete_data.csv, generating it if missing."""
    data_path = os.path.join("data", "concrete_data.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    print("No cached dataset found — generating a fresh one...")
    df = generate_dataset()
    os.makedirs("data", exist_ok=True)
    df.to_csv(data_path, index=False)
    return df


def print_metrics(name: str, y_true, y_pred) -> None:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"  [{name}] RMSE={rmse:.3f} MPa | MAE={mae:.3f} MPa | R^2={r2:.4f}")
    return rmse, mae, r2


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Concrete Compressive Strength Predictor")
    print("=" * 60)

    # ---------------------------------------------------------------
    # 1. Load / generate data
    # ---------------------------------------------------------------
    df = load_data()
    print(f"\nLoaded {len(df)} samples with {len(FEATURE_COLS)} features.")
    print(df[FEATURE_COLS + [TARGET_COL]].describe().round(2))

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

    # ---------------------------------------------------------------
    # 2. Baseline model comparison: Random Forest vs Gradient Boosting
    # ---------------------------------------------------------------
    print("\n--- Baseline model comparison (default hyperparameters) ---")
    candidates = {
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    }
    cv_scores = {}
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
        cv_scores[name] = scores.mean()
        print(f"  {name}: 5-fold CV R^2 = {scores.mean():.4f} (+/- {scores.std():.4f})")

    best_name = max(cv_scores, key=cv_scores.get)
    print(f"\nBest baseline model: {best_name}")

    # ---------------------------------------------------------------
    # 3. Hyperparameter tuning on whichever family won the baseline comparison
    # ---------------------------------------------------------------
    print(f"\n--- Hyperparameter tuning (GridSearchCV on {best_name}) ---")
    if best_name == "RandomForest":
        estimator = RandomForestRegressor(random_state=42, n_jobs=-1)
        param_grid = {
            "n_estimators": [150, 300],
            "max_depth": [None, 12, 20],
            "min_samples_leaf": [1, 2, 4],
        }
    else:
        estimator = GradientBoostingRegressor(random_state=42)
        param_grid = {
            "n_estimators": [150, 300],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.05, 0.1],
        }

    grid = GridSearchCV(
        estimator,
        param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV R^2: {grid.best_score_:.4f}")

    best_model = grid.best_estimator_

    # ---------------------------------------------------------------
    # 4. Final evaluation on held-out test set
    # ---------------------------------------------------------------
    print("\n--- Final evaluation ---")
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    print_metrics("Train", y_train, y_train_pred)
    test_rmse, test_mae, test_r2 = print_metrics("Test", y_test, y_test_pred)

    # ---------------------------------------------------------------
    # 5. Feature importance
    # ---------------------------------------------------------------
    importances = best_model.feature_importances_
    order = np.argsort(importances)[::-1]
    print("\n--- Feature importance ranking ---")
    for i in order:
        print(f"  {FEATURE_COLS[i]:<20s} {importances[i]:.4f}")

    plt.figure(figsize=(8, 5))
    plt.barh([FEATURE_COLS[i] for i in order][::-1], importances[order][::-1], color="#3B7A57")
    plt.xlabel("Importance (Gini / impurity reduction)")
    plt.title("Feature Importance — Concrete Strength Model")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=120)
    plt.close()

    # ---------------------------------------------------------------
    # 6. Predicted vs Actual plot
    # ---------------------------------------------------------------
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.5, edgecolor="k", linewidth=0.3, color="#3B7A57")
    lims = [min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())]
    plt.plot(lims, lims, "r--", label="Perfect prediction")
    plt.xlabel("Actual Compressive Strength (MPa)")
    plt.ylabel("Predicted Compressive Strength (MPa)")
    plt.title(f"Predicted vs Actual (Test R^2 = {test_r2:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "predicted_vs_actual.png"), dpi=120)
    plt.close()

    # ---------------------------------------------------------------
    # 7. Residual plot (diagnose bias / heteroscedasticity)
    # ---------------------------------------------------------------
    residuals = y_test - y_test_pred
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test_pred, residuals, alpha=0.5, edgecolor="k", linewidth=0.3, color="#B34747")
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Predicted Compressive Strength (MPa)")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "residuals.png"), dpi=120)
    plt.close()

    # ---------------------------------------------------------------
    # 8. Save the trained model for reuse in predict.py
    # ---------------------------------------------------------------
    joblib.dump(best_model, os.path.join(OUTPUT_DIR, "model.joblib"))
    print(f"\nSaved trained model -> {OUTPUT_DIR}/model.joblib")
    print(f"Saved plots -> {OUTPUT_DIR}/*.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
