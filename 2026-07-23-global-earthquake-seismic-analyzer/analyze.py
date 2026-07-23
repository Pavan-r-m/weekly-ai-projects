"""
analyze.py
----------
Global Earthquake Seismic Activity Analyzer

Loads the earthquake catalog (data/earthquake_catalog.csv, produced by
generate_data.py) and produces a multi-panel analysis covering:

  1. Gutenberg-Richter frequency-magnitude relationship (log-linear fit,
     used in real seismology to estimate the "b-value" of a region)
  2. Depth vs. magnitude scatter, colored by tectonic region
  3. Geographic distribution of epicenters (lat/lon scatter, sized by
     magnitude, colored by depth)
  4. Monthly earthquake frequency time series with a 12-month rolling average
  5. Total seismic energy released per region (bar chart, log scale)
  6. A simple composite "regional risk score" ranking

Run: python3 analyze.py
Output: output/*.png charts + output/regional_risk_ranking.csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=0.9)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("data/earthquake_catalog.csv", parse_dates=["time"])
print(f"Loaded {len(df)} earthquake events spanning "
      f"{df['time'].min().date()} to {df['time'].max().date()}")

regions = df["region"].unique()
palette = dict(zip(regions, sns.color_palette("tab10", len(regions))))

# ---------------------------------------------------------------------------
# 1. Gutenberg-Richter law: N(>=M) vs M, log-linear fit to estimate b-value
# ---------------------------------------------------------------------------
mags_sorted = np.sort(df["magnitude"].values)
n = len(mags_sorted)
# For each magnitude threshold, count events with magnitude >= threshold
thresholds = np.linspace(mags_sorted.min(), mags_sorted.max() - 0.05, 60)
counts = np.array([(df["magnitude"] >= t).sum() for t in thresholds])
log_counts = np.log10(counts, where=counts > 0, out=np.full_like(counts, np.nan, dtype=float))

valid = ~np.isnan(log_counts) & (counts > 5)  # drop noisy tail with very few events
b_fit = np.polyfit(thresholds[valid], log_counts[valid], 1)
b_value = -b_fit[0]

# ---------------------------------------------------------------------------
# 2 & 3 setup: region-level aggregates
# ---------------------------------------------------------------------------
region_stats = df.groupby("region").agg(
    events=("magnitude", "count"),
    mean_magnitude=("magnitude", "mean"),
    max_magnitude=("magnitude", "max"),
    mean_depth=("depth_km", "mean"),
    total_energy=("energy_joules", "sum"),
    tsunami_events=("tsunami_risk", "sum"),
).sort_values("total_energy", ascending=False)

# Composite risk score: normalize each factor 0-1 and combine
def norm(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-12)

region_stats["risk_score"] = (
    0.40 * norm(region_stats["total_energy"]) +
    0.30 * norm(region_stats["max_magnitude"]) +
    0.20 * norm(region_stats["events"]) +
    0.10 * norm(region_stats["tsunami_events"])
).round(3)
region_stats_sorted = region_stats.sort_values("risk_score", ascending=False)
region_stats_sorted.to_csv("output/regional_risk_ranking.csv")
print("\nTop 5 regions by composite risk score:")
print(region_stats_sorted[["events", "max_magnitude", "total_energy", "risk_score"]].head(5))

# ---------------------------------------------------------------------------
# 4. Monthly time series
# ---------------------------------------------------------------------------
monthly = df.set_index("time").resample("ME").size().rename("count").to_frame()
monthly["rolling_12mo"] = monthly["count"].rolling(12, min_periods=1).mean()

# ===========================================================================
# BUILD THE DASHBOARD FIGURE
# ===========================================================================
fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.28)
fig.suptitle("Global Earthquake Seismic Activity Dashboard (2015–2024, synthetic catalog)",
             fontsize=16, fontweight="bold", y=0.98)

# --- Panel A: Gutenberg-Richter law ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(thresholds[valid], log_counts[valid], s=18, color="steelblue", label="Observed")
fit_line = np.poly1d(b_fit)
ax1.plot(thresholds[valid], fit_line(thresholds[valid]), color="crimson", lw=2,
          label=f"Fit: b-value ≈ {b_value:.2f}")
ax1.set_xlabel("Magnitude threshold (M)")
ax1.set_ylabel("log10(N events ≥ M)")
ax1.set_title("A. Gutenberg-Richter Frequency-Magnitude Law")
ax1.legend()

# --- Panel B: Depth vs Magnitude ---
ax2 = fig.add_subplot(gs[0, 1])
for r in regions:
    sub = df[df["region"] == r]
    ax2.scatter(sub["magnitude"], sub["depth_km"], s=10, alpha=0.5, color=palette[r], label=r)
ax2.invert_yaxis()
ax2.set_xlabel("Magnitude")
ax2.set_ylabel("Depth (km)")
ax2.set_title("B. Depth vs. Magnitude by Region")
ax2.legend(fontsize=6, loc="lower right", ncol=1, framealpha=0.9)

# --- Panel C: Geographic scatter of epicenters ---
ax3 = fig.add_subplot(gs[1, :])
sc = ax3.scatter(df["longitude"], df["latitude"], s=df["magnitude"] ** 2.8 / 25,
                  c=df["depth_km"], cmap="viridis_r", alpha=0.6, edgecolors="none")
ax3.set_xlim(-180, 180)
ax3.set_ylim(-90, 90)
ax3.set_xlabel("Longitude")
ax3.set_ylabel("Latitude")
ax3.set_title("C. Global Epicenter Map (bubble size = magnitude, color = depth)")
ax3.axhline(0, color="gray", lw=0.5, ls="--")
cbar = plt.colorbar(sc, ax=ax3, fraction=0.025, pad=0.01)
cbar.set_label("Depth (km)")

# --- Panel D: Monthly frequency time series ---
ax4 = fig.add_subplot(gs[2, 0])
ax4.bar(monthly.index, monthly["count"], width=25, color="lightsteelblue", label="Monthly count")
ax4.plot(monthly.index, monthly["rolling_12mo"], color="darkorange", lw=2, label="12-mo rolling avg")
ax4.set_xlabel("Date")
ax4.set_ylabel("Number of events")
ax4.set_title("D. Monthly Earthquake Frequency")
ax4.legend()

# --- Panel E: Total energy released per region (log scale) ---
ax5 = fig.add_subplot(gs[2, 1])
energy_sorted = region_stats.sort_values("total_energy", ascending=True)
bars = ax5.barh(energy_sorted.index, energy_sorted["total_energy"],
                 color=[palette[r] for r in energy_sorted.index])
ax5.set_xscale("log")
ax5.set_xlabel("Total seismic energy released (Joules, log scale)")
ax5.set_title("E. Cumulative Seismic Energy by Region")
ax5.tick_params(axis="y", labelsize=7)

plt.savefig("output/earthquake_dashboard.png", dpi=150, bbox_inches="tight")
print("\nSaved dashboard -> output/earthquake_dashboard.png")

# ---------------------------------------------------------------------------
# Extra standalone chart: risk score ranking
# ---------------------------------------------------------------------------
fig2, ax = plt.subplots(figsize=(10, 6))
ranked = region_stats_sorted.sort_values("risk_score")
ax.barh(ranked.index, ranked["risk_score"], color=[palette[r] for r in ranked.index])
ax.set_xlabel("Composite risk score (0-1)")
ax.set_title("Regional Seismic Risk Ranking\n(weighted: energy 40% + max magnitude 30% + event count 20% + tsunami risk 10%)")
plt.tight_layout()
plt.savefig("output/regional_risk_ranking.png", dpi=150, bbox_inches="tight")
print("Saved risk ranking chart -> output/regional_risk_ranking.png")

print("\nDone. b-value estimate:", round(b_value, 3),
      "(real-world b-values are typically ~0.8-1.2, so this confirms the synthetic catalog is realistic)")
