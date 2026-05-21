"""
Global Air Quality Index (AQI) Analyzer
========================================
Generates a realistic synthetic daily AQI dataset for 10 world cities (2023),
then performs multi-dimensional analysis and produces a 5-panel visualization:

  1. Monthly AQI trends (line chart)
  2. AQI distribution per city (box plot)
  3. Pollutant correlation matrix (heatmap)
  4. AQI category breakdown per city (stacked bar)
  5. Average AQI by day of week (bar chart)

Run:
  python aqi_analyzer.py

Outputs:
  aqi_data.csv        -- raw generated dataset (3 650 rows)
  aqi_analysis.png    -- composite 5-panel figure
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
# 1. CITY CONFIGURATION
# Each city has baseline PM2.5, seasonal amplitude, and noise level (all µg/m³).
# These approximate real-world annual averages from WHO / IQAir reports.
# ─────────────────────────────────────────────────────────────────────────────

CITIES = {
    "Delhi":     {"base_pm25": 110, "seasonal_amp": 60, "noise": 30},
    "Beijing":   {"base_pm25":  85, "seasonal_amp": 50, "noise": 25},
    "Cairo":     {"base_pm25":  70, "seasonal_amp": 15, "noise": 20},
    "Lagos":     {"base_pm25":  55, "seasonal_amp": 20, "noise": 15},
    "Sao Paulo": {"base_pm25":  25, "seasonal_amp": 10, "noise":  8},
    "Paris":     {"base_pm25":  14, "seasonal_amp":  6, "noise":  5},
    "London":    {"base_pm25":  18, "seasonal_amp":  8, "noise":  6},
    "New York":  {"base_pm25":  12, "seasonal_amp":  5, "noise":  4},
    "Tokyo":     {"base_pm25":  15, "seasonal_amp":  6, "noise":  4},
    "Sydney":    {"base_pm25":   8, "seasonal_amp":  4, "noise":  3},
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def pm25_to_aqi(pm25: float) -> int:
    """
    Simplified EPA PM2.5 -> AQI conversion using linear interpolation
    between official concentration / AQI breakpoints.
    """
    breakpoints = [
        (0.0,   12.0,   0,  50),
        (12.1,  35.4,  51, 100),
        (35.5,  55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ]
    for lo_c, hi_c, lo_a, hi_a in breakpoints:
        if lo_c <= pm25 <= hi_c:
            return round(lo_a + (pm25 - lo_c) / (hi_c - lo_c) * (hi_a - lo_a))
    return 500  # Hazardous cap


def aqi_category(aqi: int) -> str:
    """Map AQI integer to EPA health category label."""
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Moderate"
    if aqi <= 150:  return "Unhealthy for Sensitive"
    if aqi <= 200:  return "Unhealthy"
    if aqi <= 300:  return "Very Unhealthy"
    return "Hazardous"


def generate_aqi_data(start_date: str = "2023-01-01", days: int = 365) -> "pd.DataFrame":
    """
    Generate a daily AQI / pollutant dataset for all cities.

    Modelled effects:
      - Seasonal cycle   : winter peaks in Northern Hemisphere cities
      - Weekly cycle     : ~10% higher on weekdays (traffic / industry)
      - Long-term trend  : slow -5% annual improvement
      - Gaussian noise   : day-to-day variability
      - Correlated pollutants: PM10, NO2, CO follow PM2.5; O3 is inversely related
    """
    np.random.seed(42)
    dates = pd.date_range(start=start_date, periods=days, freq="D")
    records = []

    for city, params in CITIES.items():
        base      = params["base_pm25"]
        amp       = params["seasonal_amp"]
        noise_std = params["noise"]

        for i, date in enumerate(dates):
            # Seasonal pattern: cosine peaking ~Jan 15 (day-of-year 15)
            doy      = date.day_of_year
            seasonal = amp * np.cos(2 * np.pi * (doy - 15) / 365)

            # Weekly cycle
            weekday_factor = 1.10 if date.dayofweek < 5 else 0.85

            # Slow improving trend over the year
            trend = -0.05 * (i / 365) * base

            # Daily PM2.5
            pm25 = max(1.0,
                       (base + seasonal + trend) * weekday_factor
                       + np.random.normal(0, noise_std))

            # Derive correlated pollutants
            pm10 = max(1.0, pm25 * np.random.uniform(1.5, 2.0) + np.random.normal(0, 5))
            no2  = max(1.0, pm25 * np.random.uniform(0.4, 0.7) + np.random.normal(0, 3))
            o3   = max(5.0, 40 - pm25 * 0.15 + np.random.normal(0, 8))  # inverse relationship
            co   = max(0.1, pm25 * np.random.uniform(0.05, 0.10) + np.random.normal(0, 0.5))

            aqi = pm25_to_aqi(pm25)

            records.append({
                "date": date,
                "city": city,
                "pm25": round(pm25, 1),
                "pm10": round(pm10, 1),
                "no2":  round(no2, 1),
                "o3":   round(o3, 1),
                "co":   round(co, 2),
                "aqi":  aqi,
            })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 3. VISUALISATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

CAT_ORDER  = ["Good", "Moderate", "Unhealthy for Sensitive",
              "Unhealthy", "Very Unhealthy", "Hazardous"]
CAT_COLORS = ["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#8f3f97", "#7e0023"]


def plot_monthly_trends(df, ax):
    """Panel 1 -- Monthly average AQI per city (full-width line chart)."""
    monthly = (df.groupby(["city", df["date"].dt.month])["aqi"]
                 .mean()
                 .reset_index())
    monthly.columns = ["city", "month", "aqi"]

    palette = sns.color_palette("tab10", n_colors=len(CITIES))
    for idx, (city, grp) in enumerate(monthly.groupby("city")):
        ax.plot(grp["month"], grp["aqi"],
                marker="o", markersize=3, linewidth=1.8,
                label=city, color=palette[idx])

    ax.axhline(100, color="orange", linestyle="--", linewidth=1.0,
               alpha=0.7, label="Moderate threshold (AQI 100)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS, fontsize=9)
    ax.set_ylabel("AQI")
    ax.set_title("Monthly Average AQI by City (2023)", fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(alpha=0.3)


def plot_aqi_distribution(df, ax):
    """Panel 2 -- Box plot of AQI distribution, cities ordered by median."""
    city_order = (df.groupby("city")["aqi"]
                    .median()
                    .sort_values(ascending=False)
                    .index.tolist())

    colors = sns.color_palette("RdYlGn_r", n_colors=len(city_order))
    bp = ax.boxplot(
        [df.loc[df["city"] == c, "aqi"].values for c in city_order],
        labels=city_order,
        patch_artist=True,
        notch=False,
        medianprops=dict(color="black", linewidth=1.5),
        flierprops=dict(marker=".", markersize=2, alpha=0.4),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.axhline(100, color="orange", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_title("AQI Distribution (ordered by median)", fontweight="bold")
    ax.set_ylabel("AQI")
    ax.set_xticklabels(city_order, rotation=30, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_pollutant_correlation(df, ax):
    """Panel 3 -- Heatmap of Pearson correlations between pollutants."""
    pollutants = ["pm25", "pm10", "no2", "o3", "co"]
    corr = df[pollutants].corr()

    sns.heatmap(
        corr, ax=ax,
        annot=True, fmt=".2f",
        cmap="coolwarm", center=0,
        square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.75},
        annot_kws={"size": 9},
    )
    labels = ["PM2.5", "PM10", "NO2", "O3", "CO"]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9, rotation=0)
    ax.set_title("Pollutant Correlation Matrix", fontweight="bold")


def plot_category_breakdown(df, ax):
    """Panel 4 -- Stacked horizontal bar of AQI category share per city."""
    df = df.copy()
    df["category"] = df["aqi"].apply(aqi_category)

    city_order = (df.groupby("city")["aqi"]
                    .median()
                    .sort_values()
                    .index.tolist())

    present_cats = [c for c in CAT_ORDER if c in df["category"].unique()]
    pivot = (df.groupby(["city", "category"])
               .size()
               .unstack(fill_value=0)
               .reindex(columns=present_cats, fill_value=0))
    pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pivot = pivot.loc[city_order]

    bottom = np.zeros(len(pivot))
    for cat, color in zip(CAT_ORDER, CAT_COLORS):
        if cat not in pivot.columns:
            continue
        vals = pivot[cat].values
        ax.barh(pivot.index, vals, left=bottom,
                color=color, label=cat, alpha=0.88)
        bottom += vals

    ax.set_title("AQI Category Breakdown by City (%)", fontweight="bold")
    ax.set_xlabel("% of Days in Category")
    ax.set_xlim(0, 100)
    ax.legend(fontsize=6.5, loc="lower right")
    ax.grid(axis="x", alpha=0.3)


def plot_weekday_pattern(df, ax):
    """Panel 5 -- Average AQI by day of week across all cities."""
    df = df.copy()
    df["dow"] = df["date"].dt.day_name()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
    avg = df.groupby("dow")["aqi"].mean().reindex(day_order)

    colors = ["#e74c3c" if d in ("Saturday", "Sunday") else "#3498db"
              for d in day_order]
    bars = ax.bar(range(7), avg.values, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.8)

    ax.set_xticks(range(7))
    ax.set_xticklabels([d[:3] for d in day_order], fontsize=9)
    ax.set_title("Average AQI by Day of Week (All Cities)", fontweight="bold")
    ax.set_ylabel("Mean AQI")
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, avg.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.0f}",
                ha="center", va="bottom", fontsize=8)


# ─────────────────────────────────────────────────────────────────────────────
# 4. CONSOLE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df):
    """Print a concise text summary of key findings to stdout."""
    sep = "=" * 62
    print(f"\n{sep}")
    print("   GLOBAL AIR QUALITY INDEX -- ANALYSIS SUMMARY (2023)")
    print(sep)

    stats = (df.groupby("city")["aqi"]
               .agg(["mean", "median", "max", "min"])
               .round(1)
               .rename(columns={"mean": "Mean", "median": "Median",
                                "max": "Max", "min": "Min"})
               .sort_values("Mean", ascending=False))

    print("\n  City AQI Statistics:\n")
    print(stats.to_string())

    best  = stats["Mean"].idxmin()
    worst = stats["Mean"].idxmax()

    df2 = df.copy()
    df2["cat"] = df2["aqi"].apply(aqi_category)
    good_pct      = (df2["cat"] == "Good").mean() * 100
    hazardous_pct = (df2["cat"] == "Hazardous").mean() * 100

    print(f"\n  Best air quality :  {best}  (mean AQI = {stats.loc[best,'Mean']})")
    print(f"  Worst air quality:  {worst}  (mean AQI = {stats.loc[worst,'Mean']})")
    print(f"  City-days with 'Good' AQI:      {good_pct:.1f}%")
    print(f"  City-days with 'Hazardous' AQI: {hazardous_pct:.1f}%")
    print(f"\n{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Global AQI Analyzer -- starting up...")

    # Generate data
    print("  Generating synthetic AQI dataset (365 days x 10 cities)...")
    df = generate_aqi_data(start_date="2023-01-01", days=365)
    print(f"  -> {len(df):,} records generated")

    df.to_csv("aqi_data.csv", index=False)
    print("  -> Saved raw dataset to aqi_data.csv")

    # Console summary
    print_summary(df)

    # Build composite figure
    print("  Rendering visualisations...")
    sns.set_style("whitegrid")
    sns.set_context("notebook")

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Global Air Quality Index -- Comprehensive Analysis (2023)",
        fontsize=15, fontweight="bold", y=0.99,
    )

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, :])   # full-width top row
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])

    plot_monthly_trends(df, ax1)
    plot_aqi_distribution(df, ax2)
    plot_pollutant_correlation(df, ax3)
    plot_category_breakdown(df, ax4)
    plot_weekday_pattern(df, ax5)

    output_path = "aqi_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  -> Figure saved to {output_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
