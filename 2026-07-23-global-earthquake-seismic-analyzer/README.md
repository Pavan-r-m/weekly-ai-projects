# Global Earthquake Seismic Activity Analyzer

A data analysis & visualization project that builds a realistic earthquake
catalog and analyzes it the way seismologists analyze real event data:
frequency-magnitude scaling laws, depth/tectonic-setting patterns, geographic
clustering along fault belts, temporal trends, and a composite regional risk
score.

## Why this is interesting

Earthquakes aren't random — they follow well-known statistical and physical
laws:

- **Gutenberg-Richter law**: the number of earthquakes above magnitude `M`
  decays exponentially as `M` increases, with a characteristic "b-value"
  (usually ~0.8–1.2 for real tectonic regions).
- **Tectonic clustering**: ~90% of global seismic energy is released along a
  handful of belts — the Pacific "Ring of Fire," the Alpide belt (Mediterranean
  → Himalayas), and the mid-ocean ridges.
- **Depth patterns**: subduction zones (e.g., Japan, Indonesia) produce quakes
  from the surface down to 600+ km, while ridges and transform faults (e.g.,
  California, Mid-Atlantic Ridge) are almost entirely shallow (<70 km).

This project generates a synthetic-but-physically-grounded catalog that
reproduces these laws, then runs an analysis pipeline that would work
identically on a real USGS earthquake export.

**Note on data**: this sandbox environment doesn't have general internet
access (seismological APIs like USGS FDSN aren't reachable), so
`generate_data.py` synthesizes a 10-year, 6,000-event global catalog by
sampling from real fault-belt coordinates, a genuine Gutenberg-Richter
magnitude distribution, and belt-appropriate depth profiles. The analysis
script recovers a b-value of **≈0.99** from the synthetic data — right in
the realistic 0.8–1.2 range — which validates that the catalog behaves like
real seismic data. Swapping in a real USGS CSV export (same column names)
requires no code changes.

## Tech stack & key concepts

- **pandas** — data wrangling, resampling, groupby aggregation
- **NumPy** — inverse-transform sampling, log-linear regression (`polyfit`)
- **Matplotlib / Seaborn** — multi-panel dashboard, geographic scatter,
  log-scale bar charts
- **Concepts**: Gutenberg-Richter frequency-magnitude law, b-value
  estimation, seismic energy-magnitude relation (`log10(E) = 1.5M + 4.8`),
  composite weighted risk scoring, time-series resampling with rolling
  averages

## Installation

```bash
pip install -r requirements.txt
```

## How to run

```bash
# Step 1: generate the synthetic earthquake catalog (data/earthquake_catalog.csv)
python3 generate_data.py

# Step 2: run the full analysis and produce charts
python3 analyze.py
```

Outputs land in `output/`:
- `earthquake_dashboard.png` — 5-panel dashboard (Gutenberg-Richter fit,
  depth vs. magnitude, world epicenter map, monthly frequency trend, energy
  by region)
- `regional_risk_ranking.png` — bar chart of the composite risk score per
  region
- `regional_risk_ranking.csv` — full numeric table behind the ranking

## Example output

```
Loaded 6000 earthquake events spanning 2015-01-01 to 2024-12-29

Top 5 regions by composite risk score:
                                    events  ...  risk_score
region                                      ...
Mexico-Central America Trench          475  ...       0.828
Indonesia-Philippines Arc             1009  ...       0.527
Mediterranean-Aegean Zone              423  ...       0.450
Andes Subduction Zone (Chile/Peru)     757  ...       0.320
Mid-Atlantic Ridge                     462  ...       0.298

Done. b-value estimate: 0.986 (real-world b-values are typically ~0.8-1.2,
so this confirms the synthetic catalog is realistic)
```

## How it works

1. **`generate_data.py`** defines 11 real tectonic belts (name + approximate
   center lat/lon + spatial spread + share of global seismicity + depth
   profile). For each of 6,000 events it: picks a belt (weighted by real
   relative seismic activity), jitters a lat/lon around that belt's center,
   samples a magnitude via inverse-transform sampling of the Gutenberg-Richter
   CDF (so large quakes are correctly rare), samples a depth from a
   belt-appropriate distribution, and flags tsunami risk for shallow, large,
   oceanic events.

2. **`analyze.py`** loads the catalog and:
   - Fits `log10(N events ≥ M)` vs. `M` with linear regression to recover the
     b-value (Panel A).
   - Plots depth vs. magnitude colored by region to show how subduction zones
     differ from ridges/faults (Panel B).
   - Renders a world scatter map of epicenters, sized by magnitude and colored
     by depth — no basemap library needed, just lat/lon as x/y (Panel C).
   - Resamples events to monthly counts and overlays a 12-month rolling
     average to reveal any temporal trend (Panel D).
   - Aggregates total seismic energy per region using the energy-magnitude
     relation and plots it on a log scale, since energy release is dominated
     by the rare largest events (Panel E).
   - Computes a composite **risk score** per region: 40% total energy + 30%
     max magnitude + 20% event count + 10% tsunami-risk event count (each
     min-max normalized), then ranks regions and exports the table to CSV.
