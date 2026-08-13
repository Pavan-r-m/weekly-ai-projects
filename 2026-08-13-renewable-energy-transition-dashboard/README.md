# Global Renewable Energy Transition Dashboard

A pandas + matplotlib/seaborn data analysis project exploring how renewable
energy adoption has evolved across 21 countries and 5 regions from 2000 to
2023 — and how it relates to GDP per capita, population, and total energy
consumption.

## Why it's interesting

The global energy transition is one of the defining economic stories of the
last two decades, but the pattern is far from uniform: hydro/geothermal-rich
nations like Iceland and Norway started near the top, fossil-fuel-heavy
economies like Germany and the UK saw explosive *growth* in renewable share
thanks to policy-driven wind/solar buildout, and fast-industrializing giants
like China and India tell a more complicated story where total renewable
*capacity* is growing fast even as renewable *share* stays roughly flat
because overall energy demand is growing even faster. This project turns
that story into six charts and a set of summary statistics.

## Tech stack and key concepts

- **pandas** — data loading, group-by aggregation, weighted averages, wide/long reshaping (`unstack`)
- **matplotlib** — line charts, stacked area charts, horizontal bar charts, log-scale scatter plots
- **seaborn** — themed styling, correlation heatmap, bubble scatter plot with hue/size encoding
- **numpy** — synthetic data generation (S-curve and exponential trajectory modeling)

## Installation

```bash
pip install -r requirements.txt
```

## How to run

```bash
# 1. Generate the sample dataset (already included in data/, but you can regenerate it)
python generate_data.py

# 2. Run the full analysis and produce all charts + summary stats
python analyze.py
```

Both scripts run fully offline — no API keys or network access required.

## Dataset

`data/renewable_energy_sample.csv` contains 504 rows: 21 countries x 24 years
(2000-2023), with columns:

| column | description |
|---|---|
| `country` | Country name |
| `region` | Europe / Asia / Middle East / North America / South America / Africa / Oceania |
| `year` | 2000-2023 |
| `renewable_share_pct` | Renewable energy as % of total energy consumption |
| `gdp_per_capita_usd` | Approximate GDP per capita (USD) |
| `total_energy_twh` | Approximate total primary energy consumption (TWh) |
| `population_millions` | Population (millions) |

**Note:** this is a *synthetic* dataset calibrated to roughly track known
real-world patterns (Iceland/Norway near-100% hydro/geothermal, China's
post-2010 renewable buildout, Gulf states starting near 0%, etc.) using
S-curve and exponential trajectory models with random noise. It's built for
demonstrating the analysis pipeline, not for citation as official energy
statistics — swap in a real dataset (e.g. Our World in Data's energy dataset)
by matching the same column names if you want live figures.

## Example output

```
Loaded 504 rows covering 21 countries, 2000-2023.

Top 10 countries by renewable share (latest year):
       country  renewable_share_pct
United Kingdom                42.26
         Chile                44.04
       Germany                46.23
        Brazil                48.31
       Denmark                64.68
       Nigeria                65.89
        Canada                68.24
         Kenya                88.71
       Iceland                91.91
        Norway                97.13

Summary stats:
                                           metric  value
                Global avg renewable share (2000)  26.67
                Global avg renewable share (2023)  40.84
                        Top country (latest year) Norway
                  Top country renewable share (%)  97.13
   Correlation: renewable share vs GDP per capita  0.158
Countries with >50% renewable share (latest year)      6
```

Generated charts (in `output/`):

1. `01_global_trend.png` — population-weighted global average renewable share, 2000-2023
2. `02_top10_2023.png` — top 10 countries by renewable share in the latest year
3. `03_leader_trajectories.png` — trajectories for Germany, China, US, UK, Australia, India
4. `04_correlation_heatmap.png` — correlation between renewable share, GDP/capita, energy use, population
5. `05_regional_stacked_area.png` — average renewable share by region over time
6. `06_gdp_vs_renewable_scatter.png` — GDP per capita vs. renewable share, sized by population, colored by region

## How it works

1. **`generate_data.py`** assigns each country a start/end renewable share, a
   growth "curve" shape (`linear`, `accelerating` S-curve, or `flat`), and
   start/end GDP and energy-consumption values. It interpolates yearly values
   along that curve and adds small Gaussian noise so the series look organic
   rather than perfectly smooth, then writes the long-format CSV.

2. **`analyze.py`** loads the CSV and runs six independent plotting
   functions, each isolating one question: What's the global trend? Who
   leads today? How did today's leaders/laggards get there? Are renewable
   share and wealth related? How do regions compare? Does wealth predict
   adoption? A `write_summary_stats` step distills the headline numbers into
   `output/summary_stats.csv` for quick reference without re-running the
   whole script.
