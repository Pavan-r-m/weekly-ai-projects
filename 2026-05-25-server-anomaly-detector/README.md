# Server Metrics Anomaly Detector 🔍

**Detect unusual behaviour in server telemetry using Isolation Forest — an unsupervised ML algorithm that requires no labelled training data.**

---

## What it does

Simulates 24 hours of server monitoring data (CPU, memory, latency, error rate) with realistic diurnal patterns, then injects 11 distinct anomaly events across four failure types:

| Anomaly Type        | What it simulates                    |
|---------------------|--------------------------------------|
| CPU spike           | Runaway process / crypto-mining      |
| Memory burst        | Memory leak / uncontrolled cache     |
| Latency surge       | Slow DB query / network saturation   |
| Error rate surge    | Downstream service failure           |

The model detects these events **without ever seeing the labels** — purely from the statistical structure of the data.

---

## Why Isolation Forest?

Traditional anomaly detection (e.g., threshold alerts) requires hand-tuned rules per metric. Isolation Forest instead:

1. Randomly selects a feature and a split value
2. Repeats recursively until each point is isolated
3. **Anomalies are isolated in fewer splits** (they're rare and different)
4. Average path length becomes the anomaly score

This makes it:
- Unsupervised (no labels needed)
- Fast even on high-dimensional data
- Robust to the curse of dimensionality

---

## Tech stack

- **scikit-learn** — `IsolationForest` implementation
- **pandas / numpy** — data generation & feature engineering
- **matplotlib** — 5-panel visualisation with anomaly overlays
- **StandardScaler** — feature normalisation before model training

Key ML concepts demonstrated:
- Unsupervised anomaly detection
- Feature engineering with rolling windows & z-scores
- Cyclic time encoding (sine/cosine for hour-of-day)
- Contamination parameter tuning
- Binary classification evaluation (precision/recall/F1)

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+. No API keys needed — everything runs locally on synthetic data.

---

## How to run

```bash
python anomaly_detector.py
```

**Expected output:**
```
┌─────────────────────────────────────────────────┐
│   Server Metrics Anomaly Detector               │
│   Algorithm : Isolation Forest (scikit-learn)   │
│   Data      : Synthetic 24-hour server telemetry│
└─────────────────────────────────────────────────┘

[1/5] Generating 24 h of synthetic server metrics …
      1,440 data points  |  156 anomaly points injected
[2/5] Engineering features (rolling stats + cyclic time) …
      Feature set size: 16 columns
[3/5] Training Isolation Forest (200 estimators, contamination=5%) …
[4/5] Evaluating …

====================================================
   ANOMALY DETECTION — EVALUATION REPORT
====================================================
  True anomalies     : 156  (10.8%)
  Predicted anomalies: 72   (5.0%)
  True positives     : 60

              precision  recall  f1-score  support
  Normal         0.96    0.98      0.97     1284
  Anomaly        0.83    0.72      0.77      156

[5/5] Plotting results …
[✓] Plot saved → anomaly_detection_results.png
[✓] Results saved → anomaly_results.csv
```

Two output files are created in the working directory:
- `anomaly_detection_results.png` — 5-panel chart (see below)
- `anomaly_results.csv` — per-minute scores and labels

---

## Example output chart

The saved PNG shows:
- **Blue** — CPU usage trace with orange predicted-anomaly dots
- **Purple** — Memory usage
- **Orange** — Latency
- **Red** — Error rate
- **Pink fill** — Normalised anomaly score (panel 5)
- **Red shading** — Ground-truth anomaly windows (for comparison)

---

## How it works (step by step)

### Step 1 — Data generation
`generate_server_metrics()` creates 1,440 one-minute samples with realistic diurnal patterns and injects 11 anomaly windows of varying width and severity.

### Step 2 — Feature engineering
`engineer_features()` adds:
- 5-min and 15-min rolling averages per metric
- Rolling z-scores (deviation from local mean)
- Sine/cosine encoding of the hour-of-day

This gives the model **temporal context** — it can see *how much* a value diverged from its recent baseline, not just its absolute level.

### Step 3 — Model training
`train_model()` scales the features with `StandardScaler`, then fits `IsolationForest(n_estimators=200, contamination=0.05)`.

### Step 4 — Scoring
`score_samples()` returns each point's average path length across all 200 trees. Short path = anomalous. Scores are normalised to [0, 1] and thresholded.

### Step 5 — Evaluation & visualisation
Predicted labels are compared against the injected ground truth to compute precision, recall, and F1. The 5-panel chart makes the detections visually interpretable.

---

## Adapting to real data

To run on real server metrics, replace `generate_server_metrics()` with a loader for your monitoring export (Prometheus, Datadog CSV, CloudWatch, etc.), keeping the same four column names: `cpu_pct`, `memory_pct`, `latency_ms`, `error_rate_pct`.

---

*Part of the [Weekly AI Projects](https://github.com/Pavan-r-m/weekly-ai-projects) series — one new AI project every weekday.*
