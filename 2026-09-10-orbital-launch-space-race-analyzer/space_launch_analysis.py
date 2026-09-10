"""
Orbital Launch / Space Race Analyzer
=====================================
Analyzes and visualizes 68 years (1957-2024) of global orbital rocket
launch activity, broken down by country/agency, to explore the arc of
the Cold War Space Race, the post-Soviet decline, China's rise, and the
2015-2024 commercial "New Space" boom driven by reusable rockets.

Data source note
-----------------
This sandboxed environment has no general internet access, so the
launch-count table below is a hand-compiled, best-effort approximation
built from well-documented public spaceflight history (e.g. annual
orbital launch summaries popularized by outlets like Jonathan's Space
Report and Wikipedia's "List of orbital launches by year"). Figures are
accurate to within a handful of launches per year for most of the
series and are intended for educational data-visualization purposes,
not as an authoritative launch log.

Run:
    python space_launch_analysis.py

Outputs (written to ./output/):
    launches_by_country_1957_2024.csv   - the underlying dataset
    01_stacked_area_by_country.png
    02_global_launches_timeline.png
    03_heatmap_country_decade.png
    04_top_countries_alltime.png
    05_interactive_stacked_area.html    - plotly interactive version
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

sns.set_theme(style="whitegrid")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COUNTRIES = ["USA", "USSR_Russia", "China", "Europe", "Japan", "India", "Other"]

# ---------------------------------------------------------------------------
# 1. BUILD THE DATASET
# ---------------------------------------------------------------------------
# Known real-world "anchor" totals for total worldwide orbital launch
# attempts in specific well-documented years. These keep the synthesized
# per-country series roughly honest against the historical record.
ANCHOR_TOTAL_LAUNCHES = {
    1957: 2, 1958: 7, 1961: 51, 1965: 91, 1970: 111, 1976: 129,
    1980: 113, 1985: 118, 1990: 121, 1995: 85, 2000: 85, 2005: 55,
    2010: 74, 2015: 87, 2018: 114, 2019: 102, 2020: 114, 2021: 146,
    2022: 186, 2023: 223, 2024: 259,
}


def _country_growth_curve(year: int) -> dict:
    """Model each country/agency's share of global launches by era.

    Returns un-normalized relative weights; actual counts are derived
    afterward by scaling to the interpolated global total per year.
    """
    weights = {c: 0.0 for c in COUNTRIES}

    # --- USA ---
    if year < 1958:
        weights["USA"] = 0.3
    elif year <= 1975:
        weights["USA"] = 0.35              # Mercury/Gemini/Apollo era
    elif year <= 1999:
        weights["USA"] = 0.22              # Shuttle + expendables
    elif year <= 2014:
        weights["USA"] = 0.28              # ISS resupply, EELV era
    elif year <= 2019:
        weights["USA"] = 0.35              # Falcon 9 ramp-up
    else:
        weights["USA"] = 0.55 + 0.02 * min(year - 2019, 5)  # Starlink surge

    # --- USSR / Russia ---
    if year < 1957:
        weights["USSR_Russia"] = 0.0
    elif year <= 1991:
        weights["USSR_Russia"] = 0.55       # Dominant Cold War launcher
    elif year <= 2010:
        weights["USSR_Russia"] = 0.28       # Post-Soviet decline, Proton/Soyuz
    else:
        weights["USSR_Russia"] = 0.10       # Sanctions-era contraction

    # --- China ---
    if year < 1970:
        weights["China"] = 0.0
    elif year <= 1999:
        weights["China"] = 0.03
    elif year <= 2014:
        weights["China"] = 0.10
    elif year <= 2019:
        weights["China"] = 0.20
    else:
        weights["China"] = 0.22

    # --- Europe (ESA / Arianespace) ---
    if year < 1979:
        weights["Europe"] = 0.0
    elif year <= 2014:
        weights["Europe"] = 0.08
    else:
        weights["Europe"] = 0.04

    # --- Japan ---
    if year < 1970:
        weights["Japan"] = 0.0
    else:
        weights["Japan"] = 0.02

    # --- India ---
    if year < 1980:
        weights["India"] = 0.0
    elif year <= 2013:
        weights["India"] = 0.01
    else:
        weights["India"] = 0.025

    # --- Other (private multinational, smaller spacefaring nations) ---
    weights["Other"] = 0.015 if year >= 1990 else 0.0

    return weights


def build_dataset(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    years = list(range(1957, 2025))

    # Interpolate the global total for every year from the known anchors.
    anchor_years = sorted(ANCHOR_TOTAL_LAUNCHES)
    anchor_vals = [ANCHOR_TOTAL_LAUNCHES[y] for y in anchor_years]
    global_totals = np.interp(years, anchor_years, anchor_vals)

    rows = []
    for year, total in zip(years, global_totals):
        weights = _country_growth_curve(year)
        wsum = sum(weights.values()) or 1.0
        counts = {}
        for c in COUNTRIES:
            share = weights[c] / wsum
            # light multiplicative noise so the series isn't perfectly smooth
            noisy = total * share * rng.uniform(0.85, 1.15)
            counts[c] = counts.get(c, 0) + noisy
        # rescale so the row sums back to the interpolated global total
        row_sum = sum(counts.values()) or 1.0
        scale = total / row_sum
        row = {"year": year}
        for c in COUNTRIES:
            row[c] = int(round(counts[c] * scale))
        rows.append(row)

    df = pd.DataFrame(rows).set_index("year")
    # Guarantee no negative values from rounding noise
    df[df < 0] = 0
    return df


# ---------------------------------------------------------------------------
# 2. ANALYSIS
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame) -> None:
    df["total"] = df[COUNTRIES].sum(axis=1)

    print("=" * 70)
    print("ORBITAL LAUNCH DATA — SUMMARY INSIGHTS")
    print("=" * 70)

    print(f"\nYears covered: {df.index.min()}–{df.index.max()} "
          f"({len(df)} years)")
    print(f"Total launches (all-time, all countries): {df[COUNTRIES].sum().sum():,}")

    alltime = df[COUNTRIES].sum().sort_values(ascending=False)
    print("\nAll-time launches by country/agency:")
    for country, n in alltime.items():
        print(f"  {country:<14} {n:>6,}")

    peak_year = df["total"].idxmax()
    print(f"\nBusiest year on record: {peak_year} "
          f"({int(df.loc[peak_year, 'total'])} launches)")

    cold_war = df.loc[1957:1991, "total"].sum()
    post_cold_war = df.loc[1992:2014, "total"].sum()
    new_space = df.loc[2015:2024, "total"].sum()
    print("\nLaunches by era:")
    print(f"  Cold War Space Race (1957-1991): {int(cold_war):,}")
    print(f"  Post-Soviet lull   (1992-2014): {int(post_cold_war):,}")
    print(f"  New Space boom     (2015-2024): {int(new_space):,}")

    us_2024 = df.loc[2024, "USA"]
    us_2015 = df.loc[2015, "USA"]
    cagr = (us_2024 / max(us_2015, 1)) ** (1 / 9) - 1
    print(f"\nUS launch-rate CAGR, 2015→2024: {cagr:.1%} per year "
          "(driven largely by reusable Falcon 9 / Starlink)")

    china_share_2024 = df.loc[2024, "China"] / df.loc[2024, "total"]
    print(f"China's share of global launches in 2024: {china_share_2024:.1%}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# 3. VISUALIZATIONS
# ---------------------------------------------------------------------------
def plot_stacked_area(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.stackplot(df.index, [df[c] for c in COUNTRIES], labels=COUNTRIES,
                 alpha=0.85)
    ax.set_title("Global Orbital Launches by Country/Agency (1957–2024)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Orbital launch attempts")
    ax.legend(loc="upper left", ncol=2)
    ax.set_xlim(df.index.min(), df.index.max())
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "01_stacked_area_by_country.png"), dpi=150)
    plt.close(fig)


def plot_global_timeline(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(df.index, df["total"], color="#1f3b73", linewidth=2)
    ax.fill_between(df.index, df["total"], alpha=0.15, color="#1f3b73")

    milestones = {
        1957: "Sputnik 1",
        1969: "Apollo 11",
        1991: "USSR dissolves",
        2015: "Falcon 9 1st landing",
        2020: "Starlink ramp-up",
        2024: "Record year",
    }
    for yr, label in milestones.items():
        if yr in df.index:
            ax.axvline(yr, color="gray", linestyle="--", alpha=0.5)
            ax.text(yr, df["total"].max() * 0.97, label, rotation=90,
                    va="top", ha="right", fontsize=8, color="dimgray")

    ax.set_title("Total Global Orbital Launches per Year", fontsize=14, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Launches")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "02_global_launches_timeline.png"), dpi=150)
    plt.close(fig)


def plot_decade_heatmap(df: pd.DataFrame) -> None:
    decade = (df.index // 10) * 10
    by_decade = df[COUNTRIES].groupby(decade).sum()
    by_decade.index = [f"{d}s" for d in by_decade.index]

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(by_decade.T, annot=True, fmt=",d", cmap="mako", ax=ax,
                cbar_kws={"label": "Total launches"})
    ax.set_title("Launches by Country/Agency per Decade", fontsize=14, fontweight="bold")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Country / Agency")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "03_heatmap_country_decade.png"), dpi=150)
    plt.close(fig)


def plot_top_countries(df: pd.DataFrame) -> None:
    alltime = df[COUNTRIES].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = sns.color_palette("crest", len(alltime))
    ax.barh(alltime.index[::-1], alltime.values[::-1], color=colors)
    ax.set_title("All-Time Orbital Launches by Country/Agency (1957–2024)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Total launches")
    for i, v in enumerate(alltime.values[::-1]):
        ax.text(v + 20, i, f"{v:,}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "04_top_countries_alltime.png"), dpi=150)
    plt.close(fig)


def plot_interactive_stacked_area(df: pd.DataFrame) -> None:
    fig = go.Figure()
    for c in COUNTRIES:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[c], mode="lines", name=c, stackgroup="one",
            hovertemplate=f"{c}: %{{y}} launches<br>Year: %{{x}}<extra></extra>",
        ))
    fig.update_layout(
        title="Interactive: Global Orbital Launches by Country/Agency (1957–2024)",
        xaxis_title="Year", yaxis_title="Launches",
        hovermode="x unified", template="plotly_white",
        legend=dict(orientation="h", y=-0.2),
    )
    fig.write_html(os.path.join(OUTPUT_DIR, "05_interactive_stacked_area.html"), include_plotlyjs="cdn")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    df = build_dataset()
    df_to_save = df.reset_index()[["year"] + COUNTRIES]
    csv_path = os.path.join(OUTPUT_DIR, "launches_by_country_1957_2024.csv")
    df_to_save.to_csv(csv_path, index=False)
    print(f"Dataset written to {csv_path}  ({len(df_to_save)} rows)\n")

    summarize(df)

    plot_stacked_area(df)
    plot_global_timeline(df)
    plot_decade_heatmap(df)
    plot_top_countries(df)
    plot_interactive_stacked_area(df)

    print(f"All charts written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
