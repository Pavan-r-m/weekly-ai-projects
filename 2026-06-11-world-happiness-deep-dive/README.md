# 🌍 World Happiness Report Deep Dive (2023)

A comprehensive data analysis and multi-panel visualization of the **2023 World Happiness Report** — exploring what really drives happiness across 80+ countries.

---

## What It Does

This project answers three core questions:
1. **Which countries are happiest — and why?**
2. **What factors predict national happiness most strongly?**
3. **How do different world regions compare?**

It produces six publication-quality charts covering country rankings, correlation heatmaps, scatter regression plots, regional distributions, feature importance, and a global score histogram.

---

## Tech Stack & Key Concepts

| Tool | Purpose |
|---|---|
| **pandas** | Data loading, groupby aggregations, descriptive stats |
| **seaborn** | Violin plots, heatmaps, scatter overlays |
| **matplotlib** | Multi-panel figures, bar charts, annotations |
| **scipy** | Pearson r, p-values for regression lines |
| **scikit-learn** | Standardised OLS regression for feature importance |

Key concepts: correlation analysis, ordinary least squares regression, standardised coefficients, feature importance, regional aggregation, distribution analysis.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## How to Run

```bash
python analyze.py
```

Charts are saved to `output_charts/`.

---

## Example Output

```
==============================
  World Happiness Report 2023
==============================
Loaded 81 countries across 9 regions.

── Correlation with Happiness Score ──────────
  GDP per Capita               +0.790  ████████████████
  Healthy Life Expectancy      +0.746  ███████████████
  Social Support               +0.742  ██████████████
  Freedom                      +0.651  █████████████
  Corruption Perception        +0.558  ███████████
  Generosity                   +0.024  ▌

── Regional Averages ─────────────────────────
  Western Europe          7.32 (avg)
  North America and ANZ   7.02
  ...
  Sub-Saharan Africa      4.28

Key Findings:
  Happiest country  : Finland (7.804)
  Least happy       : Afghanistan (1.859)
  Strongest predictor: GDP per Capita (r = +0.790)
  Happiest region   : Western Europe (avg 7.320)
```

### Generated Charts

| File | Description |
|---|---|
| `01_country_rankings.png` | Top & bottom 15 countries side-by-side |
| `02_correlation_heatmap.png` | Full feature correlation matrix |
| `03_top_predictors_scatter.png` | Scatter + regression for top 3 predictors |
| `04_regional_distribution.png` | Violin + strip plot by world region |
| `05_feature_importance.png` | Standardised OLS coefficients |
| `06_score_distribution.png` | Global score histogram with KDE |

---

## How It Works

1. **Load & clean** — reads the bundled `happiness_data.csv` (80+ countries, 2023 data)
2. **EDA** — descriptive stats, top/bottom rankings, regional means
3. **Correlation** — Pearson r between each factor and happiness score, printed as an ASCII bar chart
4. **Regression** — fits a standardised OLS model; coefficients reveal the relative "weight" of each factor after controlling for scale differences
5. **Visualisation** — six matplotlib/seaborn figures covering rankings, distributions, and relationships

---

## Dataset

`happiness_data.csv` is a curated sample derived from the [World Happiness Report 2023](https://worldhappiness.report/). Columns:

| Column | Description |
|---|---|
| `Score` | Cantril ladder score (0–10) |
| `GDP_per_capita` | Log GDP contribution |
| `Social_support` | Avg "someone to count on" response |
| `Healthy_life_expectancy` | Healthy life years at birth |
| `Freedom` | Freedom to make life choices |
| `Generosity` | Donation behaviour |
| `Corruption_perception` | Low corruption = high trust |
| `Region` | World region grouping |
