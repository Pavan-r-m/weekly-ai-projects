"""
analyze_names.py
-----------------
Data analysis & visualization of a century of U.S. baby-name popularity
(1900-2024). Explores three real, well-documented phenomena in naming data:

1. "Name lifecycles" -- most names rise, peak for a decade or two, then fade
   (e.g. Linda, Jennifer, Jessica each briefly dominated, then declined).
2. "Gender-crossover" names -- some names (Leslie, Ashley, Jordan, Avery,
   Riley, Casey, Morgan) shift their male/female split dramatically over
   the decades.
3. "Naming diversity" -- the pool of names parents choose from has become
   dramatically more diverse since 1900 (measured here with Shannon entropy),
   meaning any single name today captures a much smaller share of babies
   than a top name did in 1900-1960.

Run `generate_dataset.py` first (or just run this script, which will call it
automatically if the CSV is missing) to produce `babynames_sample.csv`.

Outputs (written to ./output/):
    01_name_trajectories_female.png
    02_name_trajectories_male.png
    03_gender_crossover_names.png
    04_naming_diversity_index.png
    05_decade_top3_heatmap.png
    decade_top_names.csv
"""

import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")
OUTPUT_DIR = "output"
DATA_FILE = "babynames_sample.csv"


def ensure_dataset():
    """Generate the dataset if it hasn't been built yet."""
    if not os.path.exists(DATA_FILE):
        print(f"{DATA_FILE} not found -- generating it now via generate_dataset.py ...")
        subprocess.run([sys.executable, "generate_dataset.py"], check=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    return df


def plot_trajectories(df: pd.DataFrame, names: list, gender: str, title: str, filename: str):
    """Line plot showing each name's yearly count across the century."""
    subset = df[(df["name"].isin(names)) & (df["gender"] == gender)]
    pivot = subset.pivot_table(index="year", columns="name", values="count", aggfunc="sum").fillna(0)

    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(ax=ax, linewidth=2.2)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Babies named per year (modeled)")
    ax.legend(title="Name", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close(fig)

    # Print each name's peak year/count -- a quick numeric summary alongside the chart
    for name in names:
        name_df = subset[subset["name"] == name]
        if name_df.empty:
            continue
        peak_row = name_df.loc[name_df["count"].idxmax()]
        print(f"  {name:<10} peaked in {int(peak_row['year'])} "
              f"with ~{int(peak_row['count']):,} babies/year")


def plot_gender_crossover(df: pd.DataFrame, names: list, filename: str):
    """For unisex/crossover names, plot % of babies who were female, over time."""
    subset = df[df["name"].isin(names)]
    pivot_total = subset.pivot_table(index="year", columns="name", values="count", aggfunc="sum")
    pivot_female = (
        subset[subset["gender"] == "F"]
        .pivot_table(index="year", columns="name", values="count", aggfunc="sum")
    )
    pct_female = (pivot_female.reindex(columns=pivot_total.columns) / pivot_total * 100).interpolate()

    fig, ax = plt.subplots(figsize=(11, 6))
    pct_female.plot(ax=ax, linewidth=2.2)
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, label="50 / 50 split")
    ax.set_title("Gender Crossover: % of Babies Who Were Female, Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("% female")
    ax.set_ylim(0, 100)
    ax.legend(title="Name", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close(fig)


def shannon_entropy(counts: np.ndarray) -> float:
    """Shannon entropy (in bits) of a distribution of name counts for one year."""
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def plot_diversity_index(df: pd.DataFrame, filename: str):
    """
    Shannon entropy of the name distribution per year: higher entropy means
    babies are spread across many different names (more diverse naming pool),
    lower entropy means a few names dominate (less diverse).
    """
    yearly_entropy = (
        df.groupby("year")["count"]
        .apply(lambda counts: shannon_entropy(counts.to_numpy()))
        .rename("entropy_bits")
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    yearly_entropy.plot(ax=ax, linewidth=2.5, color="darkorange")
    ax.set_title("Naming Diversity Over Time (Shannon Entropy)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Entropy (bits) -- higher = more diverse name choices")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close(fig)

    print(f"  Naming diversity (entropy) in 1900: {yearly_entropy.iloc[0]:.2f} bits")
    print(f"  Naming diversity (entropy) in 2024: {yearly_entropy.iloc[-1]:.2f} bits")
    print("  -> Rising entropy confirms parents draw from a far wider pool of names today.")


def decade_top3(df: pd.DataFrame) -> pd.DataFrame:
    """For each decade, find the top-3 names by total babies named (both genders combined)."""
    df = df.copy()
    df["decade"] = (df["year"] // 10) * 10
    grouped = df.groupby(["decade", "name"])["count"].sum().reset_index()

    top3_rows = []
    for decade, group in grouped.groupby("decade"):
        top3 = group.nlargest(3, "count")
        for rank, (_, row) in enumerate(top3.iterrows(), start=1):
            top3_rows.append({"decade": decade, "rank": rank, "name": row["name"], "total_count": row["count"]})
    result = pd.DataFrame(top3_rows)
    result.to_csv(os.path.join(OUTPUT_DIR, "decade_top_names.csv"), index=False)
    return result


def plot_decade_heatmap(top3_df: pd.DataFrame, filename: str):
    """Heatmap: which names were #1/#2/#3 in each decade."""
    pivot = top3_df.pivot_table(index="name", columns="decade", values="total_count", aggfunc="sum").fillna(0)
    # Keep only names that appear in at least one decade's top-3, ordered by first appearance
    pivot = pivot.loc[(pivot > 0).any(axis=1)]

    fig, ax = plt.subplots(figsize=(13, 8))
    sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Total babies named (that decade)"}, ax=ax)
    ax.set_title("Top-3 Names Per Decade -- Heatmap of Popularity")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Name")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ensure_dataset()
    df = load_data()

    print(f"Loaded {len(df):,} rows spanning {df['year'].min()}-{df['year'].max()}, "
          f"{df['name'].nunique()} names.\n")

    print("=== Female name trajectories ===")
    female_names = ["Mary", "Linda", "Jennifer", "Jessica", "Emily", "Emma", "Olivia", "Isabella", "Sophia", "Madison"]
    plot_trajectories(df, female_names, "F",
                       "A Century of Girls' Names: Rise and Fall",
                       "01_name_trajectories_female.png")

    print("\n=== Male name trajectories ===")
    male_names = ["John", "Michael", "Robert", "James", "William", "Jacob", "Noah", "Liam", "Ethan"]
    plot_trajectories(df, male_names, "M",
                       "A Century of Boys' Names: Rise and Fall",
                       "02_name_trajectories_male.png")

    print("\n=== Gender-crossover names ===")
    crossover_names = ["Leslie", "Ashley", "Jordan", "Avery", "Riley", "Casey", "Morgan"]
    plot_gender_crossover(df, crossover_names, "03_gender_crossover_names.png")
    print(f"  Chart saved: {crossover_names}")

    print("\n=== Naming diversity index ===")
    plot_diversity_index(df, "04_naming_diversity_index.png")

    print("\n=== Decade top-3 names ===")
    top3 = decade_top3(df)
    plot_decade_heatmap(top3, "05_decade_top3_heatmap.png")
    for decade, group in top3.groupby("decade"):
        names_str = ", ".join(f"{r['rank']}.{r['name']}" for _, r in group.iterrows())
        print(f"  {int(decade)}s: {names_str}")

    print(f"\nAll charts and decade_top_names.csv written to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
