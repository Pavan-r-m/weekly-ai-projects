"""
generate_data.py
-----------------
Generates a synthetic-but-realistic sample dataset of national renewable
energy adoption for the years 2000-2023.

Why synthetic? Real global energy datasets (e.g. Our World in Data / IEA)
require a network download that may not always be available in a sandboxed
environment. To keep this project fully self-contained and runnable offline,
we generate a dataset whose trends are calibrated to roughly match publicly
known real-world patterns (e.g. Iceland/Norway near-100% renewable due to
hydro/geothermal, China's rapid solar/wind buildout since ~2010, Gulf states
starting from a very low base, etc.). The numbers are illustrative, not
official statistics.

Run:
    python generate_data.py
Produces:
    data/renewable_energy_sample.csv
"""

import numpy as np
import pandas as pd

# Reproducible randomness so results are consistent across runs
RNG = np.random.default_rng(seed=42)

YEARS = list(range(2000, 2024))  # 2000-2023 inclusive

# Each country is defined by:
#   region            - used for regional aggregation charts
#   start_share        - renewable % of total energy in year 2000
#   end_share          - renewable % of total energy in year 2023 (target)
#   curve              - "linear", "accelerating" (S-curve-like), or "flat"
#   start_gdp / end_gdp - approx GDP per capita (USD) in 2000 / 2023
#   start_pop / growth  - population (millions) in 2000 and annual growth rate
#   start_energy/end_energy - total primary energy consumption (TWh)
COUNTRIES = {
    "Iceland":        dict(region="Europe",        start_share=85, end_share=99, curve="flat",
                            start_gdp=30000,  end_gdp=75000,  start_pop=0.28, growth=0.008,
                            start_energy=18,   end_energy=22),
    "Norway":         dict(region="Europe",        start_share=68, end_share=98, curve="linear",
                            start_gdp=38000,  end_gdp=95000,  start_pop=4.5,  growth=0.008,
                            start_energy=210,  end_energy=230),
    "Germany":        dict(region="Europe",        start_share=6,  end_share=46, curve="accelerating",
                            start_gdp=24000,  end_gdp=52000,  start_pop=82.0, growth=0.001,
                            start_energy=3800, end_energy=3300),
    "Denmark":        dict(region="Europe",        start_share=17, end_share=65, curve="accelerating",
                            start_gdp=31000,  end_gdp=68000,  start_pop=5.3,  growth=0.003,
                            start_energy=200,  end_energy=190),
    "United Kingdom": dict(region="Europe",        start_share=2,  end_share=42, curve="accelerating",
                            start_gdp=28000,  end_gdp=49000,  start_pop=58.9, growth=0.005,
                            start_energy=2400, end_energy=1900),
    "France":         dict(region="Europe",        start_share=15, end_share=25, curve="linear",
                            start_gdp=23000,  end_gdp=44000,  start_pop=59.0, growth=0.004,
                            start_energy=2700, end_energy=2500),
    "China":          dict(region="Asia",          start_share=17, end_share=31, curve="accelerating",
                            start_gdp=1000,   end_gdp=13000,  start_pop=1270, growth=0.003,
                            start_energy=13000, end_energy=45000),
    "India":          dict(region="Asia",          start_share=30, end_share=24, curve="flat",
                            start_gdp=450,    end_gdp=2600,   start_pop=1056, growth=0.011,
                            start_energy=5300, end_energy=15000),
    "Japan":          dict(region="Asia",          start_share=5,  end_share=22, curve="linear",
                            start_gdp=39000,  end_gdp=34000,  start_pop=127.0, growth=-0.001,
                            start_energy=5000, end_energy=4300),
    "South Korea":    dict(region="Asia",          start_share=1,  end_share=9,  curve="linear",
                            start_gdp=12000,  end_gdp=33000,  start_pop=47.0, growth=0.003,
                            start_energy=1900, end_energy=2900),
    "Saudi Arabia":   dict(region="Middle East",   start_share=0,  end_share=1,  curve="flat",
                            start_gdp=9000,   end_gdp=29000,  start_pop=20.0, growth=0.025,
                            start_energy=1100, end_energy=3000),
    "United Arab Emirates": dict(region="Middle East", start_share=0, end_share=5, curve="linear",
                            start_gdp=34000,  end_gdp=44000,  start_pop=3.2,  growth=0.03,
                            start_energy=350,  end_energy=1100),
    "United States":  dict(region="North America", start_share=6,  end_share=21, curve="linear",
                            start_gdp=36000,  end_gdp=76000,  start_pop=282.0, growth=0.007,
                            start_energy=23000, end_energy=25000),
    "Canada":         dict(region="North America", start_share=60, end_share=68, curve="linear",
                            start_gdp=24000,  end_gdp=53000,  start_pop=30.7, growth=0.009,
                            start_energy=2900, end_energy=3200),
    "Mexico":         dict(region="North America", start_share=20, end_share=17, curve="flat",
                            start_gdp=7000,   end_gdp=11000,  start_pop=100.0, growth=0.012,
                            start_energy=1600, end_energy=2200),
    "Brazil":         dict(region="South America",  start_share=42, end_share=48, curve="linear",
                            start_gdp=3700,   end_gdp=10000,  start_pop=175.0, growth=0.009,
                            start_energy=2200, end_energy=3300),
    "Chile":          dict(region="South America",  start_share=30, end_share=45, curve="accelerating",
                            start_gdp=5000,   end_gdp=16000,  start_pop=15.4, growth=0.008,
                            start_energy=250,  end_energy=450),
    "South Africa":   dict(region="Africa",        start_share=6,  end_share=12, curve="linear",
                            start_gdp=3000,   end_gdp=6600,   start_pop=45.0, growth=0.014,
                            start_energy=1000, end_energy=1300),
    "Kenya":          dict(region="Africa",        start_share=70, end_share=88, curve="linear",
                            start_gdp=430,    end_gdp=2100,   start_pop=31.0, growth=0.026,
                            start_energy=90,   end_energy=200),
    "Nigeria":        dict(region="Africa",        start_share=75, end_share=55, curve="flat",
                            start_gdp=760,    end_gdp=2200,   start_pop=123.0, growth=0.026,
                            start_energy=700,  end_energy=1500),
    "Australia":      dict(region="Oceania",       start_share=8,  end_share=32, curve="accelerating",
                            start_gdp=22000,  end_gdp=65000,  start_pop=19.0, growth=0.013,
                            start_energy=1100, end_energy=1400),
}


def share_curve(start, end, curve, n_years, rng):
    """Generate a plausible yearly renewable-share trajectory between two endpoints."""
    t = np.linspace(0, 1, n_years)
    if curve == "linear":
        base = start + (end - start) * t
    elif curve == "accelerating":
        # S-curve: slow start, fast middle growth, tapering near the end
        s = 1 / (1 + np.exp(-10 * (t - 0.6)))
        s = (s - s.min()) / (s.max() - s.min())
        base = start + (end - start) * s
    else:  # "flat" - stays near start with mild drift toward end
        base = start + (end - start) * (t ** 0.5) * 0.5
    noise = rng.normal(0, 0.6, n_years)
    return np.clip(base + noise, 0, 100)


def smooth_series(start, end, n_years, rng, noise_scale=0.02):
    """Generate a smooth exponential-ish series between two endpoints with mild noise."""
    t = np.linspace(0, 1, n_years)
    growth = (end / start) ** t if start > 0 else np.linspace(0, end, n_years)
    base = start * growth if start > 0 else growth
    noise = rng.normal(1.0, noise_scale, n_years)
    return base * noise


rows = []
for country, cfg in COUNTRIES.items():
    n = len(YEARS)
    shares = share_curve(cfg["start_share"], cfg["end_share"], cfg["curve"], n, RNG)
    gdp = smooth_series(cfg["start_gdp"], cfg["end_gdp"], n, RNG, noise_scale=0.015)
    energy = smooth_series(cfg["start_energy"], cfg["end_energy"], n, RNG, noise_scale=0.01)

    pop = cfg["start_pop"]
    for i, year in enumerate(YEARS):
        rows.append({
            "country": country,
            "region": cfg["region"],
            "year": year,
            "renewable_share_pct": round(float(shares[i]), 2),
            "gdp_per_capita_usd": round(float(gdp[i]), 2),
            "total_energy_twh": round(float(energy[i]), 2),
            "population_millions": round(pop, 3),
        })
        pop *= (1 + cfg["growth"])

df = pd.DataFrame(rows)
df = df.sort_values(["country", "year"]).reset_index(drop=True)

out_path = "data/renewable_energy_sample.csv"
df.to_csv(out_path, index=False)
print(f"Wrote {len(df)} rows for {df['country'].nunique()} countries to {out_path}")
print(df.head(10).to_string(index=False))
