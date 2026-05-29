# Fake News Detector — TF-IDF + Logistic Regression

A classic but effective NLP pipeline that classifies news headlines as **REAL** or **FAKE** using TF-IDF vectorisation and Logistic Regression. No internet connection or API key required — a synthetic dataset of 120 labelled headlines is bundled directly in the script.

---

## What it does & why it's interesting

Fake news detection is one of the most practically relevant NLP problems today. This project shows that you don't need deep learning to get strong results: a sparse TF-IDF feature matrix combined with a simple linear classifier can learn the stylistic fingerprints that separate sensational misinformation (ALL CAPS, conspiratorial phrases, absolute claims) from measured factual reporting.

Key concepts demonstrated:
- **TF-IDF vectorisation** with bigrams and sublinear TF scaling
- **Logistic Regression** with balanced class weights
- **Scikit-learn Pipelines** for clean train/predict workflows
- **Feature interpretability** — which words push toward FAKE vs REAL
- **Evaluation**: confusion matrix, classification report, ROC-AUC curve

---

## Tech stack

| Tool | Purpose |
|------|---------|
| `scikit-learn` | TF-IDF, Logistic Regression, pipeline, metrics |
| `pandas` | Dataset construction and manipulation |
| `numpy` | Array operations for feature analysis |
| `matplotlib` | Confusion matrix and ROC curve visualisation |

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.9+ recommended.

---

## How to run

**Full training + evaluation + demo:**
```bash
python fake_news_detector.py
```

**Classify a single custom headline:**
```bash
python fake_news_detector.py --text "Scientists discover cure for common cold in new trial"
python fake_news_detector.py --text "EXPOSED: Government mind control via 5G towers CONFIRMED"
```

---

## Example output

```
=======================================================
 FAKE NEWS DETECTOR — TF-IDF + Logistic Regression
=======================================================

Dataset : 120 headlines  |  REAL=60  FAKE=60
Train   : 96   Test : 24

5-Fold CV Accuracy : 0.9688 ± 0.0228

=======================================================
CLASSIFICATION REPORT
=======================================================
              precision    recall  f1-score   support

        FAKE       0.92      1.00      0.96        12
        REAL       1.00      0.92      0.96        12

    accuracy                           0.96        24

ROC-AUC Score : 0.9931

Plots saved → evaluation_plots.png

=======================================================
TOP 12 FEATURES → FAKE
=======================================================
  confirmed                          coef=-1.842
  exposed                            coef=-1.721
  secret                             coef=-1.534
  ...

=======================================================
DEMO PREDICTIONS
=======================================================
  Text   : Scientists discover potential link between gut bacteria...
  Verdict: REAL ✓  (FAKE=3.41%, REAL=96.59%)

  Text   : EXPOSED Secret government mind control programme uses 5G...
  Verdict: FAKE ✗  (FAKE=98.72%, REAL=1.28%)
```

Saved artefacts:
- `evaluation_plots.png` — confusion matrix + ROC curve
- `fake_news_model.pkl` — serialised trained pipeline (ready for inference)

---

## How it works

1. **Dataset** — 60 "real" headlines use factual, measured language (percentages, institution names, hedging verbs like "linked to", "suggests"). 60 "fake" headlines use hallmarks of misinformation: ALL CAPS words, conspiratorial nouns ("chemtrails", "reptilian", "shadow government"), absolute certainty language, and phrases like "EXPOSED" or "CONFIRMED".

2. **TF-IDF** converts each headline into a sparse numerical vector. `ngram_range=(1,2)` captures two-word phrases like "mind control" or "secret government". `sublinear_tf=True` applies log scaling to reduce the dominance of frequently repeated terms.

3. **Logistic Regression** learns a linear decision boundary in the high-dimensional TF-IDF space. The resulting coefficients are directly interpretable — words with the most negative coefficients are strong FAKE indicators; most positive are REAL indicators.

4. **Why it works so well on this dataset** — the linguistic style gap between sensational misinformation and factual reporting is large and consistent. Real-world performance degrades on more subtle cases (sophisticated disinfo, satire, clickbait) where deeper contextual models are needed.

---

## Limitations & next steps

- The bundled dataset is synthetic and stylistically exaggerated — real-world accuracy will vary
- Try replacing Logistic Regression with a fine-tuned BERT model for harder cases
- Add a proper labelled dataset (e.g. LIAR, FakeNewsNet) for production-grade training
- Extend to full articles rather than just headlines
