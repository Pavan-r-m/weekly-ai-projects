# 🚀 Orbital Launch / Space Race Analyzer

A pandas + matplotlib/seaborn/plotly data-visualization project that
traces 68 years (1957–2024) of global orbital rocket launch activity,
broken down by country/agency — from Sputnik 1 through the Cold War
Space Race, the post-Soviet lull, China's rise, and the 2015–2024
"New Space" boom driven by reusable rockets and mega-constellations.

## Why it's interesting

Orbital launch cadence is one of the cleanest proxies we have for the
geopolitics and economics of spaceflight over time. A single dataset —
"who launched what, when" — tells the story of the US–USSR Space Race,
the 1990s post-Soviet contraction, China's steady 2000s–2010s buildup,
and the dramatic 2018–2024 surge caused almost entirely by reusable
Falcon 9 boosters and Starlink batch launches. Visualizing it surfaces
patterns that are hard to see in a raw table: eras, inflection points,
and shifting leadership.

## Tech stack & key concepts

- **pandas** — building and reshaping a 68-year × 7-country time series,
  decade aggregation via integer-division grouping
- **numpy** — piecewise-linear interpolation (`np.interp`) between known
  historical "anchor" totals, seeded random noise for realism
- **matplotlib** — stacked area chart, annotated timeline with milestone
  markers, horizontal bar chart
- **seaborn** — annotated heatmap (country × decade)
- **plotly** — interactive stacked-area chart exported as standalone HTML
- Concepts: time-series decomposition by era, CAGR calculation, market
  share analysis, data storytelling through layered visualization

## Data source note

This project runs in a sandboxed environment without general internet
access, so `launches_by_country_1957_2024.csv` is a **hand-compiled,
best-effort approximation** built from well-documented public
spaceflight history (the kind of annual orbital-launch counts widely
reported in sources like *Jonathan's Space Report* and Wikipedia's
"List of orbital launches by year"). Per-country splits are modeled
from known historical patterns (e.g. Soviet dominance through the
1980s, the Falcon 9/Starlink-driven US surge after 2018) and anchored
to real, well-known worldwide totals for landmark years (1957, 1969,
1991, 2015, 2018, 2022, 2024, etc.). Treat the figures as
**illustrative, order-of-magnitude accurate** rather than an
authoritative launch log — the script that builds them
(`build_dataset()` in `space_launch_analysis.py`) is fully transparent
about its assumptions and easy to swap for a real downloaded dataset
(e.g. from a launch-log API) if you have internet access.

## Installation

```bash
pip install -r requirements.txt
```

## How to run

```bash
python space_launch_analysis.py
```

This will:
1. Build the 1957–2024 launch dataset and save it to `output/launches_by_country_1957_2024.csv`
2. Print summary insights to the console
3. Generate five charts into `output/`:
   - `01_stacked_area_by_country.png` — launches by country over time
   - `02_global_launches_timeline.png` — global total with era milestones
   - `03_heatmap_country_decade.png` — country × decade heatmap
   - `04_top_countries_alltime.png` — all-time leaderboard
   - `05_interactive_stacked_area.html` — interactive Plotly version (open in a browser)

## Example output

```
======================================================================
ORBITAL LAUNCH DATA — SUMMARY INSIGHTS
======================================================================

Years covered: 1957–2024 (68 years)
Total launches (all-time, all countries): 6,662

All-time launches by country/agency:
  USSR_Russia     2,895
  USA             2,478
  China             584
  Europe            411
  Japan             145
  India              83
  Other              66

Busiest year on record: 2024 (259 launches)

Launches by era:
  Cold War Space Race (1957-1991): 3,429
  Post-Soviet lull   (1992-2014): 1,797
  New Space boom     (2015-2024): 1,436

US launch-rate CAGR, 2015→2024: 16.7% per year (driven largely by reusable Falcon 9 / Starlink)
China's share of global launches in 2024: 20.5%
======================================================================
```

## How it works

1. **Dataset construction** (`build_dataset`) — worldwide yearly totals
   are interpolated from a handful of well-known historical anchor
   points using `np.interp`. Each country's *share* of that total is
   modeled per era with a hand-tuned weight function
   (`_country_growth_curve`) reflecting real historical shifts (e.g.
   Soviet dominance 1957–1991, China's launch-rate growth after 2010,
   the US surge after 2018). Light multiplicative noise is added for
   realism, then each year's row is rescaled to sum exactly to the
   interpolated global total.
2. **Analysis** (`summarize`) — computes all-time totals per country,
   the busiest year, launch volume by "era" (Cold War / post-Soviet /
   New Space), and CAGR for the US 2015→2024 ramp-up.
3. **Visualization** — four static matplotlib/seaborn charts plus one
   interactive Plotly HTML chart, saved to `output/`.

Everything is self-contained — no API keys or network access required.
