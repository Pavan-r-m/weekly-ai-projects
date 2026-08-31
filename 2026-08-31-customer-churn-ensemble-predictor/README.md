# Customer Churn Prediction — Ensemble Model Comparison

Predicts whether a subscription/telecom customer will **churn** (cancel
their service), then compares three classic ML models head-to-head so you
can see which approach — and which features — actually drive the decision.

## Why it's interesting

Churn prediction is one of the most common real-world ML problems (every
subscription business runs some version of this), but most tutorials stop
at "train one model, print accuracy." This project instead:

- Trains **three different model families** (linear, bagged trees, boosted
  trees) on the *same* data and compares them fairly across five metrics.
- Uses **5-fold cross-validated ROC-AUC**, not just a single train/test
  split, so the comparison isn't a fluke of one random split.
- Extracts **feature importance / coefficients** from whichever model wins,
  so the output is explainable, not just a number.
- Generates a **synthetic-but-realistic dataset** with actual causal
  structure (month-to-month contracts, electronic-check payment, low
  satisfaction, and support calls all genuinely increase churn risk in the
  generator), so the "signal" the models find is meaningful and reproducible
  without needing to download anything.

## Tech stack & key concepts

- **scikit-learn** — `LogisticRegression`, `RandomForestClassifier`,
  `GradientBoostingClassifier`, `StandardScaler`, `train_test_split`,
  `cross_val_score`
- **pandas / numpy** — synthetic data generation, one-hot encoding
- **matplotlib** — grouped bar chart, ROC curves, feature-importance chart
- Concepts: classification, class imbalance–aware metrics (precision/recall/
  F1), ROC-AUC, k-fold cross-validation, feature scaling, one-hot encoding,
  model interpretability (coefficients vs. `feature_importances_`)

## Installation

```bash
pip install -r requirements.txt
```

## How to run

```bash
# Default: 3,000 synthetic customers, seed 42
python churn_predictor.py

# Customize dataset size, random seed, and test split
python churn_predictor.py --n-customers 5000 --seed 7 --test-size 0.2
```

## Example output

```
Generating synthetic churn dataset (3000 customers, seed=42)...
  Churn rate: 36.7%
  Saved sample -> churn_dataset_sample.csv

Train size: 2250 | Test size: 750

Training models: Logistic Regression, Random Forest, Gradient Boosting...

=== Model Comparison (sorted by ROC-AUC) ===
              model  accuracy  precision  recall  f1_score  roc_auc  cv_roc_auc_mean  cv_roc_auc_std
Logistic Regression    0.7453     0.6680  0.6073    0.6362   0.8264           0.8392          0.0191
      Random Forest    0.7293     0.6800  0.4945    0.5726   0.8084           0.8233          0.0157
  Gradient Boosting    0.7280     0.6426  0.5818    0.6107   0.7948           0.8262          0.0134

Best model: Logistic Regression (ROC-AUC = 0.8264)
Saved: model_comparison.csv, model_comparison.png, roc_curves.png, feature_importance.png
```

Interestingly, plain Logistic Regression edges out the tree ensembles here —
a good reminder that more complex models don't always win, especially when
the underlying relationship (as generated) is close to linear in log-odds.

Generated files:
| File | Description |
|---|---|
| `churn_dataset_sample.csv` | First 200 rows of the generated dataset |
| `model_comparison.csv` | Full metrics table for all 3 models |
| `model_comparison.png` | Grouped bar chart comparing all 5 metrics |
| `roc_curves.png` | ROC curve overlay for all 3 models |
| `feature_importance.png` | Top 10 most influential features for the winning model |

## How it works

1. **Data generation** (`generate_churn_dataset`) — Builds 3,000 synthetic
   customers with features like tenure, monthly charges, contract type,
   support calls, and satisfaction score. Churn is generated from a
   logistic function of a weighted combination of these features plus
   Gaussian noise, mimicking how churn actually correlates with these
   variables in real telecom data (e.g. IBM's public Telco Churn dataset)
   without needing external downloads or an API key.

2. **Preprocessing** (`encode_features`) — One-hot encodes categorical
   columns (`contract_type`, `internet_service`, `payment_method`); numeric
   features are passed through unchanged. Logistic Regression additionally
   gets standard-scaled features via `StandardScaler` (tree models don't
   need scaling, so they train on the raw, more-interpretable values).

3. **Training & evaluation** (`train_and_evaluate`) — Fits all three models
   on an 75/25 train/test split, computes accuracy, precision, recall, F1,
   and ROC-AUC on the held-out test set, and separately runs 5-fold
   cross-validated ROC-AUC on the training set to check that performance is
   stable and not a lucky split.

4. **Explainability** (`plot_feature_importance`) — Pulls
   `feature_importances_` from tree models or `|coefficients|` from
   Logistic Regression (whichever model won) and plots the top 10 features
   driving churn predictions.

5. **Visualization** — Three charts are saved as PNGs: a grouped bar chart
   comparing all models across all metrics, an ROC curve overlay, and a
   feature-importance bar chart for the winning model.

## Notes

- No API key or internet access is required — the dataset is fully
  synthetic and generated locally with a fixed random seed for
  reproducibility.
- To use this on a real dataset (e.g. the IBM Telco Customer Churn CSV),
  swap out `generate_churn_dataset()` for `pd.read_csv(...)` with matching
  column names, then run the rest of the pipeline unchanged.
