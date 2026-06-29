# Credit Card Fraud Detection — ML Imbalance Study

A complete machine learning pipeline for detecting fraudulent transactions in a **severely imbalanced dataset** (98% legitimate, 2% fraud). This project tackles one of the most common and impactful real-world ML challenges: learning from rare events.

---

## What It Does

- Generates a realistic synthetic fraud dataset (20,000 transactions, 2% fraud rate)
- Trains and compares three classifiers: Logistic Regression, Random Forest, Gradient Boosting
- Handles class imbalance using `class_weight='balanced'` and `sample_weight`
- Evaluates with **Precision-Recall curves** (the correct metric for imbalanced data — ROC-AUC alone is misleading)
- Visualizes feature importances, confusion matrices, and a side-by-side metrics summary

---

## Why This Problem Is Hard

If you train a naive model on 98% legit / 2% fraud data, it learns to predict "legit" for everything and achieves 98% accuracy — while catching **zero** fraud. This project shows how to fix that.

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Data generation | NumPy, Pandas |
| ML models | scikit-learn (LR, RF, GBM) |
| Imbalance handling | `class_weight='balanced'`, `sample_weight` |
| Evaluation | `precision_recall_curve`, `average_precision_score` |
| Visualization | Matplotlib |

---

## Installation

```bash
pip install -r requirements.txt
```

---

## How to Run

```bash
python fraud_detector.py
```

Output:
- `fraud_detection_results.png` — 8-panel visualization
- Classification reports printed to console

---

## Key Concepts

### Class Imbalance
When one class is rare, models bias toward the majority. Solutions used here:
- **`class_weight='balanced'`** — sklearn automatically scales loss by inverse class frequency
- **`sample_weight`** — for Gradient Boosting (no native class_weight), we pass per-sample weights

### Precision vs Recall Tradeoff
- **Precision**: of transactions flagged as fraud, how many actually are?
- **Recall**: of all actual frauds, how many did we catch?
- High-stakes fraud detection typically prioritizes recall (miss fewer frauds) even at cost of more false alarms
- **Average Precision (AP)** summarizes the full PR curve into one number

### Features That Signal Fraud
The synthetic dataset encodes realistic fraud patterns:
| Feature | Legitimate | Fraudulent |
|---------|-----------|------------|
| Amount | $~$30 typical | $~$120 typical, higher variance |
| Hour | Business hours | Late night / early AM |
| Distance from home | 5 km avg | 150 km avg |
| Txns in last hour | ~1.2 | ~4.5 (velocity spike) |
| PIN used | 80% | 10% |
| Foreign transaction | 5% | 60% |

---

## Example Output

```
Generating 20,000 transactions (2% fraud)...
  Legitimate: 19,600  |  Fraudulent: 400  |  Ratio 1:49

Training: Logistic Regression (balanced)
  AP=0.721  ROC-AUC=0.963  F1=0.633

Training: Random Forest (balanced)
  AP=0.874  ROC-AUC=0.984  F1=0.732

Training: Gradient Boosting (sample_weight)
  AP=0.891  ROC-AUC=0.987  F1=0.748

-- Sample Predictions --
  [LEGIT] (p=3.21%)  — $50 lunch, home area, PIN used
  [FRAUD] (p=97.84%) — $2500 at 2am, 300km away, no PIN, foreign
```

---

## How It Works

1. **Data generation** — NumPy draws from different distributions for legit vs. fraud, encoding real behavioral differences (timing, velocity, channel)
2. **Preprocessing** — `StandardScaler` normalizes features; embedded in a sklearn `Pipeline` to prevent data leakage
3. **Training** — each model sees upweighted fraud samples; Gradient Boosting uses manual `sample_weight`
4. **Evaluation** — Precision-Recall curves reveal true performance on the minority class; confusion matrices show raw counts of TP/FP/FN/TN
5. **Feature importance** — Random Forest's Gini impurity reduction ranks which signals matter most

---

## Extensions to Try

- Add SMOTE (synthetic minority oversampling) via `imbalanced-learn`
- Tune decision threshold (default 0.5 may not be optimal for fraud)
- Try XGBoost with `scale_pos_weight` parameter
- Apply to the real Kaggle Credit Card Fraud dataset (284,807 transactions, 0.17% fraud)
