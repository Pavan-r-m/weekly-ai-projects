# 🌌 Stellar Object Classifier

A machine learning classifier that distinguishes **stars**, **galaxies**, and **quasars (QSOs)** from photometric survey data — inspired by the Sloan Digital Sky Survey (SDSS).

No external data download required: the script generates 10,000 synthetic objects whose statistical properties closely mirror real SDSS DR17 observations, so you can run everything offline and learn the full ML pipeline end-to-end.

---

## What It Does

Astronomical surveys photograph hundreds of millions of objects. Sorting them into *stars* (point sources in the Milky Way), *galaxies* (extended extragalactic objects), and *quasars* (extremely luminous active galactic nuclei at cosmological distances) is one of astronomy's classic classification problems.

This project:

1. **Generates** a realistic synthetic photometric catalogue (10 000 objects) with five optical-band magnitudes (u, g, r, i, z) and spectroscopic redshift — the same features used in real SDSS pipelines.
2. **Trains and compares** three classifiers: Logistic Regression, Random Forest, and Gradient Boosting.
3. **Evaluates** each model with 5-fold cross-validation, test-set accuracy, macro ROC-AUC, and a per-class classification report.
4. **Visualises** a six-panel diagnostic dashboard saved as `results.png`.
5. **Demonstrates** inference on four hand-crafted example objects.

---

## Tech Stack & Key Concepts

| Concept | Library |
|---|---|
| Synthetic data generation | NumPy |
| Data wrangling | pandas |
| Multi-class classification | scikit-learn |
| Ensemble methods | `RandomForestClassifier`, `GradientBoostingClassifier` |
| Model evaluation | ROC-AUC (OvR), confusion matrix, cross-validation |
| Visualisation | matplotlib |

**Why ensemble methods?** Photometric classification involves overlapping feature distributions (e.g. faint stars look like faint galaxies). Ensemble methods handle this gracefully through variance reduction (Random Forest) and sequential error correction (Gradient Boosting).

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.9+ recommended.

---

## How to Run

```bash
python main.py
```

Output:
- Console: cross-validation scores, test accuracy, ROC-AUC, full classification report, inference demo
- File: `results.png` — six-panel dark-theme diagnostic dashboard

---

## Example Output

```
=================================================================
  STELLAR OBJECT CLASSIFIER
  Classifying STARS, GALAXIES, and QUASARS from photometry
=================================================================

[1/5] Generating synthetic SDSS-like photometric survey data ...
  Total objects : 10,000
  Class counts  :
  GALAXY    5000
  STAR      3500
  QSO       1500

[3/5] Training and evaluating three classifiers ...

> Gradient Boosting
  CV Accuracy : 0.9821 +/- 0.0018
  Test Acc    : 0.9830
  ROC-AUC     : 0.9985

              precision    recall  f1-score
  GALAXY         0.98      0.99      0.98
  QSO            0.97      0.96      0.97
  STAR           0.99      0.98      0.99

[5/5] Running inference demo ...
  [1] True STAR      -> predicted: STAR     confidence: 99.8%  [OK]
  [2] True GALAXY    -> predicted: GALAXY   confidence: 97.4%  [OK]
  [3] True QSO       -> predicted: QSO      confidence: 95.1%  [OK]
  [4] True GALAXY    -> predicted: GALAXY   confidence: 93.2%  [OK]
```

---

## How It Works

### Feature Space

Each object is described by **10 features**:

- **u, g, r, i, z** — AB magnitudes in five SDSS optical filters (ultraviolet → near-infrared). Brighter objects have *smaller* magnitudes.
- **redshift** — the Doppler stretch of spectral lines. Stars have z ≈ 0; galaxies z ≈ 0–1; quasars z ≈ 0.5–4+.
- **u_g, g_r, r_i, i_z** — *colour indices* (band differences). These encode the shape of the spectral energy distribution and are highly discriminating.

### Why Classes Are Separable

| Class | Redshift | Colours | Notes |
|---|---|---|---|
| STAR | ~0 | Intermediate | Galactic; point source |
| GALAXY | 0–1 | Red (old stellar populations) | Extended morphology |
| QSO | 0.5–4.5 | Blue/UV excess | Non-thermal power-law SED |

Redshift alone is nearly perfect for separating stars from extragalactic objects, but colour indices are essential for galaxy/QSO disambiguation at similar redshifts.

### Model Pipeline

```
Raw features → (StandardScaler for LR) → Classifier → Predicted class + probability
```

The best model (usually Gradient Boosting) achieves >98% accuracy and >0.998 macro ROC-AUC on the test set.

---

## Want Real SDSS Data?

Download the actual SDSS DR17 star/galaxy/QSO catalogue from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17) and replace the `generate_sdss_like_data()` call with `pd.read_csv('star_classification.csv')`. The feature names and label column (`class`) map directly.

---

## Project Structure

```
2026-06-08-stellar-object-classifier/
├── main.py           # Full ML pipeline (data gen, training, eval, viz)
├── requirements.txt  # pip dependencies
├── README.md         # This file
└── results.png       # Generated after running main.py
```
