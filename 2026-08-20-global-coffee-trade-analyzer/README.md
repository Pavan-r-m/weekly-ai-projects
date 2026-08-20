# Global Coffee Trade & Consumption Analyzer

A data analysis and visualization project exploring how global coffee
production, exports, and domestic consumption have shifted across the
world's top 10 producing nations from 2000-2023.

## Why it's interesting

Coffee is one of the most heavily traded agricultural commodities in the
world, and the balance of power among producers has shifted dramatically
over the past two decades — Vietnam's robusta boom, Brazil's continued
dominance, and the emergence of African producers like Uganda and Ethiopia.
This project turns that story into a set of clear, exploratory
visualizations and a lightweight forecast, using only pandas + matplotlib/
seaborn + scikit-learn.

## Tech stack & key concepts

- **pandas** — data wrangling, pivoting, groupby aggregation
- **matplotlib / seaborn** — line charts, stacked area charts, regression
  plots, bubble scatter plots, pie charts
- **scikit-learn** — simple linear regression for a 5-year production
  forecast (basic time-series extrapolation)
- **numpy** — seeded synthetic data generation (random walks + cyclical
  price shocks)

## Dataset

`generate_data.py` produces `coffee_trade_data.csv`: a synthetic-but-realistic
sample dataset (240 rows: 10 countries × 24 years) anchored to
approximate, publicly reported magnitudes broadly consistent with
International Coffee Organization (ICO) country profiles — e.g. Brazil at
~60M 60kg bags/year, Vietnam at ~28M, etc. Production trends, price cycles
(including the well-known 2011 and 2021-2022 price spikes), and per-capita
consumption are simulated with realistic noise and growth rates so the
analysis reflects genuine industry dynamics without requiring a live API
call or dataset download.

Columns: `year, country, region, production_million_bags,
export_value_million_usd, domestic_consumption_kg_per_capita,
world_price_index`

## Installation

```bash
pip install -r requirements.txt
```

## How to run

```bash
# 1. Generate the sample dataset
python generate_data.py

# 2. Run the full analysis + generate charts
python coffee_trade_analysis.py
```

Output: 6 PNG charts and a `summary.txt` written to `output/`.

## Example output

```
============================================================
GLOBAL COFFEE TRADE & CONSUMPTION ANALYSIS -- SUMMARY
============================================================
Dataset: 10 countries, 2000-2023

Top producer (2023): Brazil (63.0M bags)
Highest domestic per-capita consumption (2023): Brazil (7.18 kg/person)
Fastest-growing producer since 2000: Vietnam (+156.8% change in output)
Brazil price-vs-export-revenue correlation: r = 0.74 (strong positive relationship)

Vietnam production forecast (linear trend):
  2024: 26.1M bags
  2025: 26.8M bags
  2026: 27.5M bags
  2027: 28.2M bags
  2028: 28.9M bags
============================================================
```

Generated charts:

1. `01_production_trends.png` — line chart of production by country over time
2. `02_market_share.png` — stacked area chart of global market share evolution
3. `03_price_vs_exports.png` — regression plot: world price index vs. Brazil's export revenue
4. `04_consumption_vs_production.png` — bubble scatter: who drinks what they grow, sized/colored by export revenue
5. `05_regional_breakdown.png` — pie chart of 2023 production by region
6. `06_vietnam_forecast.png` — linear trend forecast of Vietnam's production through 2028

## How it works

1. **Data generation** (`generate_data.py`): Starts from approximate 2023
   production anchors for each country, then walks backward to 2000 using
   country-specific long-run growth/decline rates plus random
   year-to-year noise (simulating weather and harvest variability). A
   world price index is generated with a sinusoidal cycle plus scripted
   shocks for known real-world price spike years (2011, 2021-2022) and
   the 2020 pandemic demand dip. Export value is derived from
   `production × price × noise`, and domestic consumption grows slowly
   over time with its own noise term.

2. **Analysis** (`coffee_trade_analysis.py`):
   - Pivots the long-format data into wide (year × country) tables for
     time-series plotting.
   - Computes each country's share of global production per year to show
     market-share shifts.
   - Fits a `seaborn.regplot` with Pearson correlation between world price
     and Brazil's export revenue to quantify how tightly export earnings
     track global prices.
   - Builds a bubble scatter comparing production volume against
     domestic per-capita consumption, sized/colored by export value, to
     highlight countries that consume much of what they grow (e.g.
     Brazil, Ethiopia) vs. those that export nearly everything (e.g.
     Uganda, Honduras).
   - Aggregates production by region for a pie chart snapshot.
   - Fits a `sklearn.linear_model.LinearRegression` on Vietnam's
     year-vs-production history and extrapolates 5 years forward as a
     simple trend forecast.
   - Prints and saves a text summary highlighting the top producer, top
     per-capita consumer, fastest-growing producer, and forecasted values.

No API key is required — everything runs locally and offline.
