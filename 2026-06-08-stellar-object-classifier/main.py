"""
Stellar Object Classifier
=========================
Classifies astronomical objects (STAR, GALAXY, QSO/Quasar) from photometric
and spectroscopic features using ensemble machine learning methods.

Data is synthetically generated to mimic the statistical properties of the
Sloan Digital Sky Survey (SDSS) DR17 dataset — no download required!

Key concepts: Multi-class classification, Random Forest, Gradient Boosting,
feature importance, ROC curves, confusion matrix, train/test split.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for automated runs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ─── Random seed for reproducibility ────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ════════════════════════════════════════════════════════════════════════════
# 1.  SYNTHETIC DATA GENERATION
#     Based on real statistical properties from SDSS DR17 (photometric bands
#     u, g, r, i, z and spectroscopic redshift z_spec).
# ════════════════════════════════════════════════════════════════════════════

def generate_sdss_like_data(n_samples: int = 10_000) -> pd.DataFrame:
    """
    Generate synthetic SDSS-like photometric + spectroscopic data.

    Returns a DataFrame with columns:
        u, g, r, i, z   - AB magnitudes in five optical bands
        redshift        - spectroscopic redshift
        u_g, g_r, r_i, i_z - colour indices (differences between bands)
        label           - STAR | GALAXY | QSO
    """
    # Class proportions (roughly matching SDSS DR17)
    n_stars    = int(0.35 * n_samples)
    n_galaxies = int(0.50 * n_samples)
    n_qso      = n_samples - n_stars - n_galaxies

    records = []

    # STARS: low redshift (~0), relatively blue/white, tight magnitude scatter
    for _ in range(n_stars):
        r = np.random.normal(17.5, 2.0)
        u_g = np.random.normal(1.2,  0.3)
        g_r = np.random.normal(0.45, 0.2)
        r_i = np.random.normal(0.25, 0.15)
        i_z = np.random.normal(0.10, 0.10)
        u = r + u_g + g_r
        g = r + g_r
        i = r - r_i
        z = i - i_z
        redshift = max(0.0, np.random.normal(0.0, 0.002))
        records.append(dict(u=u, g=g, r=r, i=i, z=z, redshift=redshift, label='STAR'))

    # GALAXIES: moderate redshift (0-1), redder colours, larger scatter
    for _ in range(n_galaxies):
        r = np.random.normal(20.0, 2.5)
        u_g = np.random.normal(1.8, 0.5)
        g_r = np.random.normal(0.65, 0.3)
        r_i = np.random.normal(0.35, 0.2)
        i_z = np.random.normal(0.20, 0.15)
        u = r + u_g + g_r
        g = r + g_r
        i = r - r_i
        z = i - i_z
        redshift = max(0.0, np.random.exponential(0.2))
        records.append(dict(u=u, g=g, r=r, i=i, z=z, redshift=redshift, label='GALAXY'))

    # QUASARS (QSO): high redshift (0.5-4+), extreme colours, bright and compact
    for _ in range(n_qso):
        r = np.random.normal(18.8, 1.8)
        u_g = np.random.normal(0.3,  0.6)
        g_r = np.random.normal(0.20, 0.4)
        r_i = np.random.normal(0.15, 0.3)
        i_z = np.random.normal(0.05, 0.25)
        u = r + u_g + g_r
        g = r + g_r
        i = r - r_i
        z = i - i_z
        redshift = np.random.uniform(0.5, 4.5)
        records.append(dict(u=u, g=g, r=r, i=i, z=z, redshift=redshift, label='QSO'))

    df = pd.DataFrame(records)

    # Derived colour indices
    df['u_g'] = df['u'] - df['g']
    df['g_r'] = df['g'] - df['r']
    df['r_i'] = df['r'] - df['i']
    df['i_z'] = df['i'] - df['z']

    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return df


# ════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE ENGINEERING & SPLITTING
# ════════════════════════════════════════════════════════════════════════════

FEATURES = ['u', 'g', 'r', 'i', 'z', 'redshift', 'u_g', 'g_r', 'r_i', 'i_z']
TARGET   = 'label'


def prepare_data(df: pd.DataFrame):
    """Split into train / test and encode labels."""
    X = df[FEATURES].values
    y = df[TARGET].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=RANDOM_SEED, stratify=y_enc
    )
    return X_train, X_test, y_train, y_test, le


# ════════════════════════════════════════════════════════════════════════════
# 3.  MODEL DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════

def build_models():
    """Return a dict of named sklearn pipelines."""
    return {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_SEED,
                                       multi_class='multinomial', C=1.0))
        ]),
        'Random Forest': Pipeline([
            ('clf', RandomForestClassifier(
                n_estimators=200, max_depth=15,
                min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1
            ))
        ]),
        'Gradient Boosting': Pipeline([
            ('clf', GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.1, max_depth=5,
                subsample=0.8, random_state=RANDOM_SEED
            ))
        ]),
    }


# ════════════════════════════════════════════════════════════════════════════
# 4.  TRAINING & EVALUATION
# ════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(models, X_train, X_test, y_train, y_test, le):
    """Train every model, print metrics, return results dict."""
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    print("\n" + "=" * 65)
    print("  MODEL COMPARISON - STELLAR OBJECT CLASSIFIER")
    print("=" * 65)

    for name, pipeline in models.items():
        print(f"\n> {name}")
        cv_scores = cross_val_score(pipeline, X_train, y_train,
                                    cv=cv, scoring='accuracy', n_jobs=-1)
        print(f"  CV Accuracy : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        pipeline.fit(X_train, y_train)
        y_pred  = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)
        acc     = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
        print(f"  Test Acc    : {acc:.4f}")
        print(f"  ROC-AUC     : {roc_auc:.4f}")
        print()
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        results[name] = {
            'pipeline': pipeline, 'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(), 'acc': acc, 'roc_auc': roc_auc,
            'y_pred': y_pred, 'y_proba': y_proba,
        }

    return results


# ════════════════════════════════════════════════════════════════════════════
# 5.  VISUALISATIONS
# ════════════════════════════════════════════════════════════════════════════

def _style_ax(ax, hide_spines=True):
    """Apply dark theme to a matplotlib axis."""
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='white', labelsize=8)
    for spine in ax.spines.values():
        if hide_spines:
            spine.set_visible(False)
        else:
            spine.set_edgecolor('#444')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')


def plot_results(df, results, X_test, y_test, le, best_name):
    """Generate a 2x3 dashboard of diagnostic plots."""
    CLASS_COLORS = {'STAR': '#FFD700', 'GALAXY': '#1E90FF', 'QSO': '#FF4500'}
    class_names  = le.classes_

    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor('#0d1117')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Colour-Colour Diagram
    ax1 = fig.add_subplot(gs[0, 0])
    for cls, color in CLASS_COLORS.items():
        sub = df[df['label'] == cls].sample(min(500, len(df[df['label'] == cls])), random_state=42)
        ax1.scatter(sub['u_g'], sub['g_r'], s=4, alpha=0.5, color=color, label=cls)
    ax1.set_xlabel('u - g', color='white')
    ax1.set_ylabel('g - r', color='white')
    ax1.set_title('Colour-Colour Diagram', color='white', fontweight='bold')
    ax1.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
    _style_ax(ax1)

    # 2. Redshift Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    for cls, color in CLASS_COLORS.items():
        sub = df[df['label'] == cls]['redshift']
        ax2.hist(sub, bins=60, alpha=0.7, color=color, label=cls, density=True)
    ax2.set_xlabel('Redshift (z)', color='white')
    ax2.set_ylabel('Density', color='white')
    ax2.set_title('Redshift Distribution', color='white', fontweight='bold')
    ax2.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
    ax2.set_xlim(-0.1, 5)
    _style_ax(ax2)

    # 3. Model Accuracy Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    model_names = list(results.keys())
    accs  = [results[n]['acc']     for n in model_names]
    aucs  = [results[n]['roc_auc'] for n in model_names]
    x     = np.arange(len(model_names))
    w     = 0.35
    bars1 = ax3.bar(x - w/2, accs, w, label='Accuracy',  color='#4CAF50', alpha=0.85)
    bars2 = ax3.bar(x + w/2, aucs, w, label='ROC-AUC',   color='#2196F3', alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels([n.replace(' ', '\n') for n in model_names], color='white', fontsize=8)
    ax3.set_ylim(0.85, 1.01)
    ax3.set_title('Model Performance', color='white', fontweight='bold')
    ax3.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
    for bar in list(bars1) + list(bars2):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{bar.get_height():.3f}', ha='center', va='bottom', color='white', fontsize=7)
    _style_ax(ax3)

    # 4. Confusion Matrix
    ax4 = fig.add_subplot(gs[1, 0])
    cm   = confusion_matrix(y_test, results[best_name]['y_pred'])
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    ax4.imshow(cm_n, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax4.set_xticks(range(len(class_names)))
    ax4.set_yticks(range(len(class_names)))
    ax4.set_xticklabels(class_names, color='white', fontsize=9)
    ax4.set_yticklabels(class_names, color='white', fontsize=9)
    ax4.set_xlabel('Predicted', color='white')
    ax4.set_ylabel('Actual',    color='white')
    ax4.set_title(f'Confusion Matrix\n({best_name})', color='white', fontweight='bold')
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax4.text(j, i, f'{cm_n[i,j]:.2f}', ha='center', va='center',
                     color='black' if cm_n[i,j] > 0.5 else 'white', fontsize=9)
    _style_ax(ax4, hide_spines=False)

    # 5. Feature Importance
    ax5 = fig.add_subplot(gs[1, 1])
    best_pipe = results[best_name]['pipeline']
    clf = best_pipe.named_steps.get('clf', list(best_pipe.named_steps.values())[-1])
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        idx = np.argsort(importances)[::-1]
        ax5.barh([FEATURES[i] for i in idx[::-1]], importances[idx[::-1]], color='#FF8C00', alpha=0.85)
        ax5.set_xlabel('Importance', color='white')
        ax5.set_title(f'Feature Importance\n({best_name})', color='white', fontweight='bold')
    else:
        coefs = np.abs(clf.coef_).mean(axis=0)
        idx   = np.argsort(coefs)
        ax5.barh([FEATURES[i] for i in idx], coefs[idx], color='#FF8C00', alpha=0.85)
        ax5.set_xlabel('|Coefficient| avg', color='white')
        ax5.set_title('Feature Importance\n(Logistic Regression)', color='white', fontweight='bold')
    _style_ax(ax5)

    # 6. ROC Curves
    ax6 = fig.add_subplot(gs[1, 2])
    y_proba = results[best_name]['y_proba']
    ROC_COLORS = ['#FFD700', '#1E90FF', '#FF4500']
    for i, (cls, col) in enumerate(zip(class_names, ROC_COLORS)):
        fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_proba[:, i])
        auc_val = roc_auc_score((y_test == i).astype(int), y_proba[:, i])
        ax6.plot(fpr, tpr, color=col, lw=2, label=f'{cls} (AUC={auc_val:.3f})')
    ax6.plot([0, 1], [0, 1], 'w--', lw=1, alpha=0.4)
    ax6.set_xlabel('False Positive Rate', color='white')
    ax6.set_ylabel('True Positive Rate',  color='white')
    ax6.set_title(f'ROC Curves (OvR)\n({best_name})', color='white', fontweight='bold')
    ax6.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
    _style_ax(ax6)

    fig.suptitle('Stellar Object Classifier - SDSS-like Photometric Survey',
                 fontsize=15, fontweight='bold', color='white', y=0.97)
    plt.savefig('results.png', dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    print("\n  [OK] Dashboard saved -> results.png")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# 6.  INFERENCE DEMO
# ════════════════════════════════════════════════════════════════════════════

def demo_inference(best_pipeline, le):
    """Show predictions on hand-crafted example objects."""
    print("\n" + "-" * 55)
    print("  INFERENCE DEMO - classify new objects")
    print("-" * 55)

    # Columns: u,   g,    r,    i,    z,  redshift, u_g, g_r, r_i, i_z
    demo_objects = [
        [18.1, 17.4, 17.1, 17.0, 16.9, 0.0001, 0.7, 0.3, 0.1, 0.1],  # STAR
        [22.5, 21.1, 20.5, 20.1, 19.9, 0.32,   1.4, 0.6, 0.4, 0.2],  # GALAXY
        [18.9, 19.0, 19.1, 18.9, 18.7, 2.10,  -0.1,-0.1, 0.2, 0.2],  # QSO
        [21.0, 20.3, 19.9, 19.7, 19.5, 0.08,   0.7, 0.4, 0.2, 0.2],  # GALAXY
    ]
    labels_demo = ['True STAR', 'True GALAXY', 'True QSO', 'True GALAXY']

    X_demo = np.array(demo_objects)
    preds  = best_pipeline.predict(X_demo)
    probas = best_pipeline.predict_proba(X_demo)

    for idx, (true_lbl, pred, prob) in enumerate(zip(labels_demo, preds, probas), start=1):
        pred_name = le.inverse_transform([pred])[0]
        conf      = prob.max() * 100
        correct   = "OK" if pred_name in true_lbl else "WRONG"
        print(f"  [{idx}] {true_lbl:<14} -> predicted: {pred_name:<7}  "
              f"confidence: {conf:.1f}%  [{correct}]")


# ════════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  STELLAR OBJECT CLASSIFIER")
    print("  Classifying STARS, GALAXIES, and QUASARS from photometry")
    print("=" * 65)

    print("\n[1/5] Generating synthetic SDSS-like photometric survey data ...")
    df = generate_sdss_like_data(n_samples=10_000)
    print(f"  Total objects : {len(df):,}")
    print(f"  Class counts  :\n{df['label'].value_counts().to_string()}")

    print("\n[2/5] Preparing features and train/test split ...")
    X_train, X_test, y_train, y_test, le = prepare_data(df)
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"  Features: {FEATURES}")

    print("\n[3/5] Training and evaluating three classifiers ...")
    models  = build_models()
    results = train_and_evaluate(models, X_train, X_test, y_train, y_test, le)

    best_name = max(results, key=lambda n: results[n]['roc_auc'])
    print(f"\n  Best model: {best_name} (ROC-AUC = {results[best_name]['roc_auc']:.4f})")

    print("\n[4/5] Generating visualisation dashboard ...")
    plot_results(df, results, X_test, y_test, le, best_name)

    print("\n[5/5] Running inference demo ...")
    demo_inference(results[best_name]['pipeline'], le)

    print("\n" + "=" * 65)
    print("  Done! Check results.png for the full diagnostic dashboard.")
    print("=" * 65)


if __name__ == '__main__':
    main()
