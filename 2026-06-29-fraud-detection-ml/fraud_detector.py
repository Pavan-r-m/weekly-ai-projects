"""
Credit Card Fraud Detection — Handling Class Imbalance with Machine Learning
============================================================================
Real-world fraud datasets are extremely imbalanced (often <1% fraud cases).
This project demonstrates how to:
  - Generate a realistic synthetic fraud dataset
  - Handle severe class imbalance with class_weight='balanced'
  - Compare Logistic Regression, Random Forest, and Gradient Boosting
  - Evaluate models with precision-recall curves (better than ROC for imbalance)
  - Visualize feature importances and confusion matrices

Run:  python fraud_detector.py
Output: fraud_detection_results.png + printed classification reports
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, auc,
    average_precision_score, f1_score
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. Generate Synthetic Fraud Dataset
# ─────────────────────────────────────────────

def _hour_probs():
    weights = np.zeros(24)
    weights[8:22] = 3.0
    weights[22:]  = 1.0
    weights[:8]   = 0.5
    return weights / weights.sum()

def _fraud_hour_probs():
    weights = np.zeros(24)
    weights[0:6]  = 4.0
    weights[6:8]  = 1.5
    weights[8:22] = 1.0
    weights[22:]  = 3.0
    return weights / weights.sum()

def generate_fraud_dataset(n_samples=20_000, fraud_rate=0.02, random_state=42):
    """
    Synthetic credit card transactions: ~2% fraudulent.
    Features simulate amount, timing, velocity, geography, channel signals.
    """
    print(f"Generating {n_samples:,} transactions ({fraud_rate*100:.0f}% fraud)...")
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud
    rng = np.random.RandomState(random_state)

    legit = {
        "amount":             rng.lognormal(3.5, 1.2, n_legit),
        "hour_of_day":        rng.choice(range(24), n_legit, p=_hour_probs()),
        "days_since_last":    rng.exponential(2.0, n_legit),
        "merchant_category":  rng.choice([0,1,2,3,4], n_legit),
        "distance_from_home": rng.exponential(5, n_legit),
        "n_transactions_1h":  rng.poisson(1.2, n_legit),
        "foreign_transaction":rng.binomial(1, 0.05, n_legit),
        "online_order":       rng.binomial(1, 0.30, n_legit),
        "pin_used":           rng.binomial(1, 0.80, n_legit),
        "ratio_to_median":    rng.lognormal(0, 0.5, n_legit),
    }
    fraud = {
        "amount":             rng.lognormal(4.8, 1.5, n_fraud),
        "hour_of_day":        rng.choice(range(24), n_fraud, p=_fraud_hour_probs()),
        "days_since_last":    rng.exponential(0.3, n_fraud),
        "merchant_category":  rng.choice([0,1,2,3,4], n_fraud, p=[0.05,0.05,0.6,0.2,0.1]),
        "distance_from_home": rng.exponential(150, n_fraud),
        "n_transactions_1h":  rng.poisson(4.5, n_fraud),
        "foreign_transaction":rng.binomial(1, 0.60, n_fraud),
        "online_order":       rng.binomial(1, 0.85, n_fraud),
        "pin_used":           rng.binomial(1, 0.10, n_fraud),
        "ratio_to_median":    rng.lognormal(1.2, 0.8, n_fraud),
    }
    df = pd.concat([
        pd.DataFrame(legit).assign(label=0),
        pd.DataFrame(fraud).assign(label=1),
    ], ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f"  Legitimate: {n_legit:,}  |  Fraudulent: {n_fraud:,}  |  Ratio 1:{n_legit//n_fraud}")
    return df

FEATURE_COLS = [
    "amount","hour_of_day","days_since_last","merchant_category",
    "distance_from_home","n_transactions_1h","foreign_transaction",
    "online_order","pin_used","ratio_to_median"
]
FEATURE_LABELS = [
    "Transaction Amount","Hour of Day","Days Since Last Txn",
    "Merchant Category","Distance from Home (km)","# Txns in Last Hour",
    "Foreign Transaction","Online Order","PIN Used","Amount / Median Ratio"
]

# ─────────────────────────────────────────────
# 2. Build Models
# ─────────────────────────────────────────────

def build_models():
    """
    Three classifiers. class_weight='balanced' automatically up-weights
    the minority class (fraud) so the model doesn't just predict 'legit' always.
    """
    return {
        "Logistic Regression\n(balanced)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        ]),
        "Random Forest\n(balanced)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, class_weight="balanced",
                max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)),
        ]),
        "Gradient Boosting\n(sample_weight)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.05,
                max_depth=4, subsample=0.8, random_state=42)),
        ]),
    }

# ─────────────────────────────────────────────
# 3. Train & Evaluate
# ─────────────────────────────────────────────

def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    results = {}
    fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\nClass imbalance (legit:fraud) = {fraud_ratio:.0f}:1\n")

    for name, model in models.items():
        clean = name.replace("\n", " ")
        print(f"Training: {clean}")
        if "Gradient" in name:
            sw = np.where(y_train == 1, fraud_ratio, 1.0)
            model.fit(X_train, y_train, clf__sample_weight=sw)
        else:
            model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        fpr, tpr, _  = roc_curve(y_test, y_proba)
        ap   = average_precision_score(y_test, y_proba)
        rauc = auc(fpr, tpr)
        f1   = f1_score(y_test, y_pred)

        results[name] = dict(
            model=model, y_pred=y_pred, y_proba=y_proba,
            prec=prec, rec=rec, fpr=fpr, tpr=tpr,
            ap=ap, roc_auc=rauc, f1=f1,
            cm=confusion_matrix(y_test, y_pred),
        )
        print(f"  AP={ap:.3f}  ROC-AUC={rauc:.3f}  F1={f1:.3f}")
        print(classification_report(y_test, y_pred, target_names=["Legit","Fraud"]))

    return results

# ─────────────────────────────────────────────
# 4. Visualize
# ─────────────────────────────────────────────

COLORS = ["#3B82F6","#10B981","#F59E0B"]

def plot_results(results, y_train, output_path="fraud_detection_results.png"):
    names = list(results.keys())
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Credit Card Fraud Detection — ML Model Comparison",
                 fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Class distribution
    ax0 = fig.add_subplot(gs[0, 0])
    counts = [(y_train==0).sum(), (y_train==1).sum()]
    bars = ax0.bar(["Legitimate","Fraudulent"], counts,
                   color=["#10B981","#EF4444"], edgecolor="white")
    for b, c in zip(bars, counts):
        ax0.text(b.get_x()+b.get_width()/2, b.get_height()+50,
                 f"{c:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax0.set_title("Class Distribution (Train)", fontweight="bold")
    ax0.set_yscale("log"); ax0.grid(axis="y", alpha=0.3)

    # Precision-Recall
    ax1 = fig.add_subplot(gs[0, 1])
    for (n, r), c in zip(results.items(), COLORS):
        ax1.plot(r["rec"], r["prec"], color=c, lw=2,
                 label=f"{n.split(chr(10))[0]} AP={r['ap']:.3f}")
    ax1.set(xlabel="Recall", ylabel="Precision",
            title="Precision-Recall Curves\n(key metric for imbalanced data)")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # ROC
    ax2 = fig.add_subplot(gs[0, 2])
    for (n, r), c in zip(results.items(), COLORS):
        ax2.plot(r["fpr"], r["tpr"], color=c, lw=2,
                 label=f"{n.split(chr(10))[0]} AUC={r['roc_auc']:.3f}")
    ax2.plot([0,1],[0,1],"k--",lw=1,label="Random")
    ax2.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curves")
    ax2.legend(fontsize=8, loc="lower right"); ax2.grid(alpha=0.3)

    # Confusion matrices
    for idx, ((n, r), c) in enumerate(zip(results.items(), COLORS)):
        ax = fig.add_subplot(gs[1, idx])
        cm = r["cm"]
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Pred: Legit","Pred: Fraud"], fontsize=8)
        ax.set_yticklabels(["True: Legit","True: Fraud"], fontsize=8)
        for i in range(2):
            for j in range(2):
                v = cm[i,j]
                ax.text(j, i, f"{v:,}", ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if v > cm.max()*0.6 else "black")
        ax.set_title(f"Confusion Matrix\n{n.split(chr(10))[0]}", fontweight="bold", fontsize=9)

    # Feature importances (Random Forest)
    ax6 = fig.add_subplot(gs[2, :2])
    rf_name = next(n for n in names if "Random" in n)
    imp = results[rf_name]["model"].named_steps["clf"].feature_importances_
    idx = np.argsort(imp)
    colors_bar = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(idx)))
    bars = ax6.barh([FEATURE_LABELS[i] for i in idx], imp[idx], color=colors_bar, edgecolor="white")
    for b, v in zip(bars, imp[idx]):
        ax6.text(b.get_width()+0.001, b.get_y()+b.get_height()/2,
                 f"{v:.3f}", va="center", fontsize=8)
    ax6.set(xlabel="Feature Importance (Gini reduction)",
            title="Random Forest — Feature Importances")
    ax6.grid(axis="x", alpha=0.3)

    # Metrics summary
    ax7 = fig.add_subplot(gs[2, 2])
    short_names = [n.split("\n")[0] for n in names]
    metrics = {
        "F1-Score": [results[n]["f1"]      for n in names],
        "Avg Prec": [results[n]["ap"]       for n in names],
        "ROC-AUC":  [results[n]["roc_auc"]  for n in names],
    }
    x = np.arange(len(short_names))
    w = 0.25
    for i, (mn, vals) in enumerate(metrics.items()):
        ax7.bar(x+i*w, vals, w, label=mn,
                color=["#3B82F6","#F59E0B","#10B981"][i], alpha=0.85, edgecolor="white")
    ax7.set_xticks(x+w); ax7.set_xticklabels(short_names, fontsize=8, rotation=10)
    ax7.set_ylim(0, 1.05); ax7.set_ylabel("Score")
    ax7.set_title("Model Metrics Summary", fontweight="bold")
    ax7.legend(fontsize=8); ax7.grid(axis="y", alpha=0.3)

    plt.savefig(output_path, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"\nPlot saved -> {output_path}")

# ─────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  CREDIT CARD FRAUD DETECTION")
    print("=" * 60)

    df = generate_fraud_dataset()
    X = df[FEATURE_COLS].values
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)
    print(f"\nTrain: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

    models  = build_models()
    results = train_and_evaluate(models, X_train, X_test, y_train, y_test)

    # Cross-validate best model
    best_name  = max(results, key=lambda n: results[n]["ap"])
    best_model = results[best_name]["model"]
    print(f"\n5-Fold CV ({best_name.replace(chr(10),' ')}):")
    cv = cross_val_score(
        best_model, X_train, y_train,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="average_precision"
    )
    print(f"  Avg Precision: {cv.mean():.3f} +/- {cv.std():.3f}")

    plot_results(results, y_train)

    # Quick demo
    print("\n-- Sample Predictions --")
    demo = np.array([
        [  50, 14, 2.0, 1,   3,  1, 0, 0, 1, 0.9],   # normal lunch purchase
        [2500,  2, 0.1, 2, 300,  6, 1, 1, 0, 8.5],   # suspicious: night, far, no PIN
    ])
    sc  = best_model.named_steps["scaler"]
    clf = best_model.named_steps["clf"]
    proba = clf.predict_proba(sc.transform(demo))[:, 1]
    descs = ["$50 lunch, home area, PIN used",
             "$2500 at 2am, 300km away, no PIN, foreign"]
    for p, d in zip(proba, descs):
        tag = "[FRAUD]" if p > 0.5 else "[LEGIT]"
        print(f"  {tag} (p={p:.2%}) — {d}")

    print("\nDone!")

if __name__ == "__main__":
    main()
