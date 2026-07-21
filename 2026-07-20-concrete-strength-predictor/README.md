# Concrete Compressive Strength Predictor

Predicts the 28-day-equivalent compressive strength (in MPa) of a concrete
mix from its ingredient proportions and curing age, using an ensemble
regression model. Concrete mix design is a real engineering problem — get
the water-cement ratio or curing schedule wrong and you either waste cement
(cost) or under-build strength (safety risk). This project walks through a
complete applied-ML regression pipeline on a domain-realistic dataset.

## Why it's interesting

- Regression (not classification) on a physically-grounded dataset generated
  from a simplified water-cement ratio / curing-age model (inspired by the
  classic UCI "Concrete Compressive Strength" dataset), so results actually
  make engineering sense — e.g. strength rises with cement content and age,
  and falls with excess water.
- Compares two ensemble methods (Random Forest vs Gradient Boosting) with
  cross-validation before committing to hyperparameter tuning.
- Explains *why* the model predicts what it does via feature importances,
  not just the prediction itself.
- Includes residual diagnostics to sanity-check the model isn't systematically
  over/under-predicting at certain strength ranges.

## Tech stack & key concepts

- **scikit-learn** — `RandomForestRegressor`, `GradientBoostingRegressor`,
  `GridSearchCV`, `cross_val_score`, `train_test_split`
- **pandas / numpy** — data generation and manipulation
- **matplotlib** — feature importance, predicted-vs-actual, and residual plots
- **joblib** — model persistence
- Concepts: ensemble regression, k-fold cross-validation, grid-search
  hyperparameter tuning, feature importance, residual analysis, train/test
  generalization gap

## Installation

```bash
pip install -r requirements.txt
```

## How to run

```bash
# 1. (Optional) Generate the dataset explicitly — train_model.py will also
#    auto-generate it on first run if data/concrete_data.csv doesn't exist.
python generate_data.py

# 2. Train the model, print metrics, and save plots + the trained model
python train_model.py

# 3. Predict strength for example mixes, or your own custom mix
python predict.py
python predict.py --cement 380 --slag 60 --fly_ash 20 --water 165 \
    --superplasticizer 8 --coarse_aggregate 980 --fine_aggregate 780 --age 28
```

A small sample of the generated dataset is included at
`data/concrete_data_sample.csv` (20 rows) for quick inspection; the full
1,030-row dataset is generated on demand by `generate_data.py` /
`train_model.py` (deterministic — same seed every run).

## Example output

```
--- Baseline model comparison (default hyperparameters) ---
  RandomForest: 5-fold CV R^2 = 0.9114 (+/- 0.0196)
  GradientBoosting: 5-fold CV R^2 = 0.9496 (+/- 0.0089)

Best baseline model: GradientBoosting

--- Hyperparameter tuning (GridSearchCV on GradientBoosting) ---
  Best params: {'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 300}
  Best CV R^2: 0.9549

--- Final evaluation ---
  [Train] RMSE=2.213 MPa | MAE=1.735 MPa | R^2=0.9848
  [Test]  RMSE=3.687 MPa | MAE=3.024 MPa | R^2=0.9477

--- Feature importance ranking ---
  age                  0.6029
  cement               0.1866
  water                0.0888
  blast_furnace_slag   0.0678
  fly_ash              0.0260
  superplasticizer     0.0243
  fine_aggregate       0.0023
  coarse_aggregate     0.0014
```

`predict.py` with no arguments:

```
Low-strength mix (high water-cement ratio, 7-day cure)
  Predicted compressive strength: 14.46 MPa

Standard structural mix (28-day cure)
  Predicted compressive strength: 38.64 MPa

High-performance mix (low water, slag+fly ash, 90-day cure)
  Predicted compressive strength: 69.81 MPa
```

Generated plots (in `outputs/` after running `train_model.py`):
- `feature_importance.png` — ranked bar chart of ingredient influence
- `predicted_vs_actual.png` — scatter plot against the perfect-prediction line
- `residuals.png` — residual vs. predicted value, to check for bias/heteroscedasticity

## How it works

1. **`generate_data.py`** simulates 1,030 concrete mixes across realistic
   ingredient ranges (cement, blast furnace slag, fly ash, water,
   superplasticizer, coarse/fine aggregate, curing age). Strength is computed
   from a simplified Abrams'-law water-cement ratio term, a logarithmic
   curing-age curve, small bonuses for superplasticizer and pozzolanic
   materials (slag/fly ash), an aggregate-ratio penalty, and Gaussian noise
   to mimic lab measurement variance.
2. **`train_model.py`** loads (or generates) the data, splits 80/20
   train/test, cross-validates Random Forest vs Gradient Boosting, runs
   `GridSearchCV` on whichever family wins the baseline comparison, evaluates
   the tuned model on the held-out test set, and saves diagnostic plots plus
   the trained model (`outputs/model.joblib`).
3. **`predict.py`** loads the saved model and predicts strength for either
   three built-in example mixes or a custom mix passed via CLI flags.

No API keys or internet access are required — the dataset is fully
synthetic and generated locally with a fixed random seed for reproducibility.
