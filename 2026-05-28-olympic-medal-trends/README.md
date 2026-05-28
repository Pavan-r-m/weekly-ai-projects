# Olympic Games Medal Trends Visualizer

A rich data analysis and visualization project exploring **126 years of Olympic Games history** (1896–2022). Built with pandas and matplotlib, it generates a four-panel dark-mode dashboard covering medal counts, composition, trends over time, and a decade-by-decade heatmap.

---

## What It Does & Why It's Interesting

The Olympics is one of humanity's richest long-running datasets — spanning two World Wars, the Cold War rivalry between the USA and USSR, China's emergence as a sporting superpower, and Norway's quiet dominance of the Winter Games. This project turns that history into four compelling visualizations in a single command.

Key insights the dashboard reveals:
- How dramatically the USA's medal haul jumped in 1984 (LA Olympics, Soviet boycott)
- The USSR/Russia trajectory from 1952 to present
- China's explosive rise from zero medals in 1980 to 100+ in 2008
- Norway's unrivalled Winter Games dominance across 10 decades

---

## Tech Stack & Key Concepts

- **pandas** — data wrangling, groupby aggregations, pivot tables
- **matplotlib** — multi-panel figure layout with `GridSpec`, horizontal bar charts, line plots
- **seaborn** — annotated heatmap with custom colormap
- **numpy** — array operations for stacked bar positioning
- All data is **embedded directly** in the script — no internet access or API keys needed

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.8+ required.

---

## How to Run

```bash
python olympic_medals.py
```

The script prints a summary table and fun-facts to the terminal, then saves the dashboard as:

```
olympic_medal_analysis.png
```

---

## Example Output (Terminal)

```
Dataset Overview
==================================================
Total records        : 211
Countries covered    : 10
Years covered        : 1896 - 2022
Total medals in data : 14,782

Top 10 Nations by Total Medals (All-Time)
--------------------------------------------------
              gold  silver  bronze  total
country
USA           1173    1041     919   3133
USSR/Russia    736     618     589   1943
Germany        468     487     486   1441
...

── Fun Facts ──────────────────────────────────────────────
Most gold medals all-time : USA (1,173)
Best Winter Olympics nation: Norway (386 medals)
Most medals in a single Games: USA at 1904 Summer (239 medals)

Highest Gold-Medal Efficiency (nations with >=50 total medals):
  China: 42.3% gold rate
  USSR/Russia: 37.9% gold rate
  USA: 37.5% gold rate
```

---

## Dashboard Panels

| Panel | Description |
|-------|-------------|
| **Top 10 All-Time** | Horizontal bar chart — total medals, gold-colored for #1 nation |
| **Medal Composition** | Stacked bar showing gold/silver/bronze split per nation |
| **Trends Over Time** | Line chart tracking the top 5 nations across every Games |
| **Decade Heatmap** | Color-coded grid: intensity = medals won in that decade |

---

## How It Works

1. **Data** — 211 hand-curated records covering 10 nations across all Summer and Winter Games. Each row stores `(country, year, game_type, gold, silver, bronze)`.

2. **Aggregation** — pandas `groupby` sums medals by country for rankings, by `(country, year)` for trend lines, and by `(country, decade)` for the heatmap pivot.

3. **Layout** — `matplotlib.gridspec.GridSpec` arranges four axes in a 2×2 grid on a single figure. Each panel shares a dark-mode theme (`#0d1117` background) for visual cohesion.

4. **Heatmap** — `seaborn.heatmap` with the `YlOrRd` colormap and inline annotations shows medal intensity at a glance; warmer colors = more medals that decade.

---

## Extending This Project

- Add all ~140 countries using the full IOC dataset from [kaggle.com](https://www.kaggle.com/datasets/the-guardian/olympic-games)
- Replace the static data with a live scrape of [olympedia.org](https://www.olympedia.org)
- Add an interactive Plotly version for hover tooltips
- Filter by sport or event type for deeper analysis
