"""
analyze.py
----------
Data analysis & visualization of global renewable energy adoption
(2000-2023) using pandas, matplotlib, and seaborn.

Run:
    python analyze.py

Reads:
    data/renewable_energy_sample.csv   (generate it first with generate_data.py)

Writes (into output/):
    01_global_trend.png
    02_top10_2023.png
    03_leader_trajectories.png
    04_correlation_heatmap.png
    05_regional_stacked_area.png
    06_gdp_vs_renewable_scatter.png
    summary_stats.csv
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless rendering, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="viridis")
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(path="data/renewable_energy_sample.csv"):
    df = pd.read_csv(path)
    return df


def plot_global_trend(df):
    """Line chart: population-weighted global average renewable share by year."""
    tmp = df.assign(weighted=df["renewable_share_pct"] * df["population_millions"])
    grouped = tmp.groupby("year")[["weighted", "population_millions"]].sum()
    yearly = (grouped["weighted"] / grouped["population_millions"]).rename(
        "global_avg_renewable_pct").reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(yearly["year"], yearly["global_avg_renewable_pct"], marker="o",
            color="#2E8B57", linewidth=2.5)
    ax.fill_between(yearly["year"], yearly["global_avg_renewable_pct"], alpha=0.15, color="#2E8B57")
    ax.set_title("Population-Weighted Global Average Renewable Energy Share (2000-2023)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Renewable Share of Total Energy (%)")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/01_global_trend.png", dpi=150)
    plt.close(fig)
    return yearly


def plot_top10_2023(df):
    """Horizontal bar chart of the top 10 countries by renewable share in 2023."""
    latest = df[df["year"] == df["year"].max()]
    top10 = latest.nlargest(10, "renewable_share_pct").sort_values("renewable_share_pct")

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(top10["country"], top10["renewable_share_pct"], color=sns.color_palette("viridis", 10))
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_title(f"Top 10 Countries by Renewable Energy Share ({df['year'].max()})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Renewable Share of Total Energy (%)")
    ax.set_xlim(0, 105)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/02_top10_2023.png", dpi=150)
    plt.close(fig)
    return top10


def plot_leader_trajectories(df):
    """Multi-line chart tracking a handful of notable countries over time."""
    countries = ["Germany", "China", "United States", "United Kingdom", "Australia", "India"]
    subset = df[df["country"].isin(countries)]

    fig, ax = plt.subplots(figsize=(10, 6))
    for country, grp in subset.groupby("country"):
        grp = grp.sort_values("year")
        ax.plot(grp["year"], grp["renewable_share_pct"], marker=".", linewidth=2, label=country)
    ax.set_title("Renewable Energy Share Trajectories, Selected Countries (2000-2023)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Renewable Share (%)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/03_leader_trajectories.png", dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df):
    """Correlation heatmap between renewable share, GDP/capita, energy use, population."""
    cols = ["renewable_share_pct", "gdp_per_capita_usd", "total_energy_twh", "population_millions"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                cbar_kws={"label": "Pearson correlation"})
    ax.set_title("Correlation Between Renewable Share and Economic/Energy Indicators",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/04_correlation_heatmap.png", dpi=150)
    plt.close(fig)
    return corr


def plot_regional_stacked_area(df):
    """Stacked area chart of average renewable share by region over time."""
    regional = df.groupby(["year", "region"])["renewable_share_pct"].mean().unstack()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stackplot(regional.index, regional.T.values, labels=regional.columns,
                 colors=sns.color_palette("viridis", len(regional.columns)), alpha=0.85)
    ax.set_title("Average Renewable Energy Share by Region (2000-2023)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Sum of Regional Average Renewable Share (%)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/05_regional_stacked_area.png", dpi=150)
    plt.close(fig)


def plot_gdp_vs_renewable(df):
    """Scatter plot: GDP per capita vs renewable share in the latest year, colored by region."""
    latest = df[df["year"] == df["year"].max()]

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=latest, x="gdp_per_capita_usd", y="renewable_share_pct",
        hue="region", size="population_millions", sizes=(50, 500),
        alpha=0.75, ax=ax, palette="Set2",
    )
    for _, row in latest.iterrows():
        ax.annotate(row["country"], (row["gdp_per_capita_usd"], row["renewable_share_pct"]),
                    fontsize=7, alpha=0.7, xytext=(4, 2), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_title(f"GDP per Capita vs. Renewable Energy Share ({df['year'].max()})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("GDP per Capita (USD, log scale)")
    ax.set_ylabel("Renewable Share (%)")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/06_gdp_vs_renewable_scatter.png", dpi=150)
    plt.close(fig)


def write_summary_stats(df, corr):
    """Write a small CSV of headline numbers used in the README."""
    latest_year = df["year"].max()
    first_year = df["year"].min()
    latest = df[df["year"] == latest_year]
    first = df[df["year"] == first_year]

    summary = pd.DataFrame({
        "metric": [
            f"Global avg renewable share ({first_year})",
            f"Global avg renewable share ({latest_year})",
            "Top country (latest year)",
            "Top country renewable share (%)",
            "Correlation: renewable share vs GDP per capita",
            "Countries with >50% renewable share (latest year)",
        ],
        "value": [
            round(first["renewable_share_pct"].mean(), 2),
            round(latest["renewable_share_pct"].mean(), 2),
            latest.loc[latest["renewable_share_pct"].idxmax(), "country"],
            round(latest["renewable_share_pct"].max(), 2),
            round(corr.loc["renewable_share_pct", "gdp_per_capita_usd"], 3),
            int((latest["renewable_share_pct"] > 50).sum()),
        ],
    })
    summary.to_csv(f"{OUTPUT_DIR}/summary_stats.csv", index=False)
    return summary


def main():
    df = load_data()
    print(f"Loaded {len(df)} rows covering {df['country'].nunique()} countries, "
          f"{df['year'].min()}-{df['year'].max()}.\n")

    plot_global_trend(df)
    top10 = plot_top10_2023(df)
    plot_leader_trajectories(df)
    corr = plot_correlation_heatmap(df)
    plot_regional_stacked_area(df)
    plot_gdp_vs_renewable(df)
    summary = write_summary_stats(df, corr)

    print("Top 10 countries by renewable share (latest year):")
    print(top10[["country", "renewable_share_pct"]].to_string(index=False))
    print("\nSummary stats:")
    print(summary.to_string(index=False))
    print(f"\nAll charts and summary_stats.csv written to '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
