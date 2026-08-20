"""
generate_data.py
-----------------
Generates a realistic sample dataset of global coffee production, exports,
and domestic consumption for the top 10 coffee-producing countries from
2000-2023.

The values are synthesized using a seeded random walk anchored to
approximate, publicly-known magnitudes (e.g. Brazil producing roughly
60 million 60kg bags/year, Vietnam ~28 million, etc. -- figures broadly
consistent with International Coffee Organization (ICO) reporting).
This keeps the dataset realistic and internally consistent (production
trends, price cycles, growth patterns) without requiring a live API call,
so the project runs fully offline.

Run this script first to produce `coffee_trade_data.csv`, which is then
consumed by `coffee_trade_analysis.py`.
"""

import numpy as np
import pandas as pd

# Reproducible randomness so the "sample" data is stable across runs
rng = np.random.default_rng(seed=42)

YEARS = list(range(2000, 2024))  # 2000-2023 inclusive

# Approximate 2023 production (million 60kg bags), long-run annual growth
# rate, region, and typical per-capita domestic consumption (kg/year).
# These anchors are broadly consistent with ICO country profiles.
COUNTRIES = {
    "Brazil":      {"base_2023": 63.0, "growth": 0.018, "region": "South America",   "consumption_kg": 6.2},
    "Vietnam":     {"base_2023": 27.5, "growth": 0.035, "region": "Southeast Asia",  "consumption_kg": 2.6},
    "Colombia":    {"base_2023": 11.5, "growth": 0.006, "region": "South America",   "consumption_kg": 2.9},
    "Indonesia":   {"base_2023": 10.0, "growth": 0.012, "region": "Southeast Asia",  "consumption_kg": 1.2},
    "Ethiopia":    {"base_2023": 7.8,  "growth": 0.022, "region": "Africa",          "consumption_kg": 3.2},
    "Honduras":    {"base_2023": 5.9,  "growth": 0.020, "region": "Central America", "consumption_kg": 1.8},
    "India":       {"base_2023": 5.8,  "growth": 0.014, "region": "South Asia",      "consumption_kg": 0.9},
    "Uganda":      {"base_2023": 5.7,  "growth": 0.028, "region": "Africa",          "consumption_kg": 0.6},
    "Mexico":      {"base_2023": 4.0,  "growth": -0.008,"region": "Central America", "consumption_kg": 1.6},
    "Guatemala":   {"base_2023": 3.5,  "growth": -0.004,"region": "Central America", "consumption_kg": 2.1},
}

# World coffee price cycle (approximate relative price index, loosely in
# USD cents/lb terms). Coffee prices spiked around 2011 and 2021-2022 due
# to weather/supply shocks; that cycle is reproduced below for realism.
def price_curve(year):
    base = 130  # roughly baseline arabica cents/lb-ish index
    cycle = 35 * np.sin((year - 2000) / 4.2) + 0.6 * (year - 2000)
    shock = 0
    if year in (2011, 2021, 2022):
        shock = 45
    if year == 2020:
        shock = -10  # pandemic demand dip
    noise = rng.normal(0, 4)
    return max(60, base + cycle + shock + noise)


def build_dataset():
    rows = []
    for country, cfg in COUNTRIES.items():
        base = cfg["base_2023"]
        growth = cfg["growth"]
        # Walk backwards from 2023 to 2000 using the long-run growth rate,
        # with year-to-year noise for realism (weather, disease, policy).
        production_by_year = {}
        prod = base
        for year in reversed(YEARS):
            production_by_year[year] = prod
            # undo one year of growth to step backward
            prod = prod / (1 + growth)
            prod *= rng.normal(1.0, 0.03)  # weather/harvest noise

        for year in YEARS:
            production = max(0.5, production_by_year[year])
            price_idx = price_curve(year)
            # Export value roughly scales with bags * price, with a unit
            # conversion fudge factor to land in plausible USD millions.
            export_value_musd = production * price_idx * 0.62 * rng.normal(1.0, 0.05)
            consumption_kg = max(
                0.1,
                cfg["consumption_kg"] * (1 + 0.01 * (year - 2000)) * rng.normal(1.0, 0.04),
            )
            rows.append(
                {
                    "year": year,
                    "country": country,
                    "region": cfg["region"],
                    "production_million_bags": round(production, 2),
                    "export_value_million_usd": round(max(0, export_value_musd), 1),
                    "domestic_consumption_kg_per_capita": round(consumption_kg, 2),
                    "world_price_index": round(price_idx, 1),
                }
            )

    df = pd.DataFrame(rows).sort_values(["country", "year"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = build_dataset()
    out_path = "coffee_trade_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows for {df['country'].nunique()} countries "
          f"({df['year'].min()}-{df['year'].max()}) to {out_path}")
