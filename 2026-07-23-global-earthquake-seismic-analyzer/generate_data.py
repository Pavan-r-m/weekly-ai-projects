"""
generate_data.py
----------------
Generates a realistic SYNTHETIC global earthquake catalog.

Why synthetic instead of a live download? This project runs in a sandboxed
environment without general internet access (only an allowlisted set of
domains, which does not include seismological data providers like the
USGS FDSN web service). Rather than silently failing, we build a catalog
that follows the real statistical and geographic laws seismologists use:

1. Gutenberg-Richter law: log10(N) = a - b*M, i.e. the number of quakes
   above magnitude M falls off exponentially with a b-value ~1.0.
2. Real tectonic belts: ~90% of the world's seismic energy is released
   along a handful of well known belts (Pacific "Ring of Fire", the
   Alpide belt from the Mediterranean to the Himalayas, and the
   Mid-Atlantic Ridge). We sample epicenters from Gaussian clusters
   centered on real coordinates along these belts.
3. Depth distribution differs by tectonic setting: subduction zones
   (Ring of Fire) produce many intermediate/deep quakes, while ridges
   and transform faults are almost all shallow (<70 km).

The result is a CSV that "looks and behaves" like a real USGS catalog
export, making it a fair stand-in for teaching/demoing the analysis
pipeline in analyze.py.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(seed=42)

# ---------------------------------------------------------------------------
# 1. Define real tectonic belts: (region name, center lat, center lon,
#    spatial spread in degrees, share of global events, depth profile)
# depth profile is (mean_km, std_km, max_km) roughly matching real settings
# ---------------------------------------------------------------------------
BELTS = [
    dict(name="Japan-Kuril Subduction Zone",     lat=38.0,  lon=142.0, spread=4.5, weight=0.14, depth=(60, 70, 600)),
    dict(name="Indonesia-Philippines Arc",       lat=-2.0,  lon=118.0, spread=8.0, weight=0.16, depth=(80, 90, 650)),
    dict(name="Andes Subduction Zone (Chile/Peru)", lat=-20.0, lon=-70.0, spread=8.0, weight=0.13, depth=(50, 60, 600)),
    dict(name="Aleutian-Alaska Arc",             lat=57.0,  lon=-155.0, spread=6.0, weight=0.09, depth=(40, 50, 300)),
    dict(name="Mexico-Central America Trench",   lat=16.0,  lon=-97.0, spread=4.0, weight=0.08, depth=(30, 40, 250)),
    dict(name="Himalaya-Alpide Belt",            lat=30.0,  lon=80.0,  spread=10.0, weight=0.14, depth=(20, 20, 200)),
    dict(name="Mediterranean-Aegean Zone",       lat=37.0,  lon=25.0,  spread=5.0, weight=0.07, depth=(15, 15, 100)),
    dict(name="California-Cascadia Fault System",lat=37.5,  lon=-121.5, spread=3.5, weight=0.06, depth=(10, 8, 40)),
    dict(name="Mid-Atlantic Ridge",              lat=0.0,   lon=-25.0, spread=25.0, weight=0.08, depth=(10, 5, 30)),
    dict(name="East African Rift",               lat=-2.0,  lon=35.0,  spread=6.0, weight=0.03, depth=(15, 10, 60)),
    dict(name="New Zealand-Tonga Arc",           lat=-25.0, lon=180.0, spread=8.0, weight=0.02, depth=(100, 120, 650)),
]
weights = np.array([b["weight"] for b in BELTS])
weights = weights / weights.sum()

N_EVENTS = 6000
START = datetime(2015, 1, 1)
END = datetime(2024, 12, 31)
total_seconds = int((END - START).total_seconds())

# ---------------------------------------------------------------------------
# 2. Sample which belt each event belongs to
# ---------------------------------------------------------------------------
belt_idx = rng.choice(len(BELTS), size=N_EVENTS, p=weights)

records = []
for i in range(N_EVENTS):
    belt = BELTS[belt_idx[i]]

    # Location: Gaussian jitter around the belt's center, wrapped to valid ranges
    lat = np.clip(rng.normal(belt["lat"], belt["spread"]), -85, 85)
    lon = rng.normal(belt["lon"], belt["spread"] * 1.4)
    lon = ((lon + 180) % 360) - 180  # wrap to [-180, 180]

    # Magnitude via inverse-transform sampling of the Gutenberg-Richter law
    # P(M >= m) ~ 10^(-b*m); we sample M in [4.0, 9.0] with b=1.0
    b_value = 1.0
    m_min, m_max = 4.0, 9.0
    u = rng.uniform()
    # CDF inversion for truncated exponential-like decay
    mag = m_min - np.log10(1 - u * (1 - 10 ** (-b_value * (m_max - m_min)))) / b_value
    mag = float(np.clip(mag, m_min, m_max))

    # Depth: gamma-like distribution parameterized per belt, capped at max
    mean_d, std_d, max_d = belt["depth"]
    depth = float(np.clip(rng.normal(mean_d, std_d), 2, max_d))

    # Time: uniform over the 10-year window, with larger quakes rarer (handled above)
    ts = START + timedelta(seconds=int(rng.uniform(0, total_seconds)))

    # Tsunami risk heuristic: shallow + large + oceanic belt
    oceanic = belt["name"] not in ("Himalaya-Alpide Belt", "East African Rift", "California-Cascadia Fault System")
    tsunami_risk = int(oceanic and depth < 70 and mag >= 7.0)

    records.append(dict(
        time=ts,
        latitude=round(lat, 4),
        longitude=round(lon, 4),
        depth_km=round(depth, 1),
        magnitude=round(mag, 2),
        region=belt["name"],
        tsunami_risk=tsunami_risk,
    ))

df = pd.DataFrame(records).sort_values("time").reset_index(drop=True)

# Seismic energy release (Joules), from the standard Gutenberg-Richter energy relation:
# log10(E) = 1.5*M + 4.8
df["energy_joules"] = 10 ** (1.5 * df["magnitude"] + 4.8)

out_path = "data/earthquake_catalog.csv"
df.to_csv(out_path, index=False)
print(f"Generated {len(df)} synthetic earthquake events -> {out_path}")
print(df["region"].value_counts())
