# 🎮 Video Game Sales & Genre Evolution Analyzer

A data analysis and visualization project that digs into video game industry trends from 1980 to 2023 — exploring how genres rose and fell, which regions dominated, which publishers won, and how platform eras changed the landscape.

---

## What It Does

- **Genre evolution heatmap** — see which genres ruled each decade
- **Publisher leaderboard** — who moved the most units, ever
- **Regional market breakdown** — NA vs EU vs JP vs Rest of World, per genre
- **Platform era area chart** — Nintendo vs PlayStation vs Xbox vs PC across eras
- **Sales bubble chart** — every notable title plotted by year, sized by NA sales
- **Genre trajectory lines** — how each genre's total sales climbed or fell across eras
- **Regional correlation scatter** — do NA hits also sell in Europe? In Japan?

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| **pandas** | Data loading, grouping, pivoting |
| **numpy** | Normalization, linear regression for trend lines |
| **matplotlib** | All chart rendering |
| **seaborn** | Heatmap and aesthetic styling |

Key concepts: time-series grouping, percentage normalization, stacked area charts, correlation analysis, multi-panel dashboards.

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

Outputs three PNG charts in the `outputs/` folder:

| File | What it shows |
|------|--------------|
| `vg_sales_dashboard.png` | 6-panel overview dashboard |
| `genre_trajectory.png` | Genre sales across eras (line chart) |
| `regional_correlations.png` | NA vs EU vs JP scatter plots with trend lines |

---

## Example Output

```
============================================================
  Video Game Sales & Genre Evolution Analyzer
============================================================
✓ Loaded 120 game records spanning 1980–2023
  Genres: ['Action', 'Action-Adventure', 'Fighting', 'Misc', ...]
  Platforms: 24 unique platforms
  Publishers: 28 unique publishers

📊 Top 5 Publishers:
     Publisher  Total_Sales_M
      Nintendo         1023.4
Take-Two Interactive    352.1
  Electronic Arts       247.8
      Activision        193.5
         Capcom          78.2

🌍 Regional Sales Share:
  North America        42.3%
  Europe               31.5%
  Japan                16.4%
  Rest of World         9.8%

📈 Most Dominant Genre per Era:
  1980s   →  Platform             (139.9M)
  1990s   →  Platform             (187.3M)
  2000s   →  Sports               (312.4M)
  2010s   →  Action               (284.1M)
  2020s   →  Action-Adventure     (136.2M)
```

---

## Dataset

`data/vgsales_sample.csv` — 120 curated titles from 1980–2023, hand-selected to represent each era, genre, and platform. Fields:

- `Rank`, `Name`, `Platform`, `Year`, `Genre`, `Publisher`
- `NA_Sales`, `EU_Sales`, `JP_Sales`, `Other_Sales`, `Global_Sales` (all in millions)

The original full dataset (16,000+ titles) is available at [Kaggle — Video Game Sales](https://www.kaggle.com/datasets/gregorut/videogamesales). Just replace the CSV path to use it.

---

## How It Works

1. **Load & clean** — read CSV, drop nulls, bin years into era labels (1980s–2020s)
2. **Aggregate** — `groupby` on genre×era, publisher, platform family, region
3. **Normalize** — convert raw sales to percentage shares for fair cross-era comparison
4. **Visualize** — matplotlib multi-panel figure with seaborn heatmap overlay
5. **Annotate** — landmark titles labeled on the scatter, Pearson r on correlation plots

### Interesting findings

- **Japan is an outlier**: Role-Playing games sell ~3× more in JP relative to their global share, while Shooter games barely register
- **The 2000s were the Sports era**: FIFA, Madden, and Wii Fit drove enormous volumes
- **NA and EU correlate strongly** (r ≈ 0.85+), but JP correlation with both is weaker — Japan has its own distinct tastes
- **Action-Adventure surged in the 2020s** as cinematic single-player games became the prestige format

---

## Project Info

- **Category:** Data Analysis & Visualization (Thursday)
- **Part of:** [Weekly AI Projects](https://github.com/Pavan-r-m/weekly-ai-projects)
- **Date:** 2026-06-18
