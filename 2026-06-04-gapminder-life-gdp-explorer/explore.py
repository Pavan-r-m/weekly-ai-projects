"""
Gapminder Life Expectancy & GDP Correlation Explorer
=====================================================
Analyzes the famous Gapminder dataset to uncover relationships between
GDP per capita, life expectancy, and population across countries and decades.

Features:
  - Animated bubble chart (GDP vs life expectancy over time)
  - Regional heatmap of average life expectancy by decade
  - Correlation scatter matrix
  - K-Means clustering of countries by development indicators
  - Top/bottom country ranking by life expectancy gains
  - Static PNG exports + interactive HTML chart

Run:
    python explore.py

Outputs (in ./outputs/):
  - bubble_chart.html       — interactive animated Plotly chart
  - regional_heatmap.png    — seaborn heatmap by continent & decade
  - scatter_matrix.png      — pairplot of key variables
  - clusters.png            — PCA-reduced cluster visualization
  - top_gainers.png         — countries with biggest LE improvement
  - summary_stats.csv       — per-continent summary table
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ── Output directory ─────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Seaborn style ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load data
# ═══════════════════════════════════════════════════════════════════════════════

def load_gapminder() -> pd.DataFrame:
    """
    Load Gapminder data straight from Plotly's built-in dataset.
    Columns: country, continent, year, lifeExp, pop, gdpPercap, iso_alpha, iso_num
    """
    print("📥  Loading Gapminder dataset from plotly.express …")
    df = px.data.gapminder()
    # Rename for clarity
    df = df.rename(columns={
        "lifeExp": "life_expectancy",
        "gdpPercap": "gdp_per_capita",
        "pop": "population",
    })
    print(f"    {len(df):,} rows  |  {df['country'].nunique()} countries  |  "
          f"years {df['year'].min()}–{df['year'].max()}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Animated bubble chart  (Plotly → HTML)
# ═══════════════════════════════════════════════════════════════════════════════

def animated_bubble_chart(df: pd.DataFrame) -> None:
    print("\n🫧   Building animated bubble chart …")
    fig = px.scatter(
        df,
        x="gdp_per_capita",
        y="life_expectancy",
        animation_frame="year",
        animation_group="country",
        size="population",
        color="continent",
        hover_name="country",
        log_x=True,
        size_max=55,
        range_x=[200, 120_000],
        range_y=[25, 90],
        title="GDP per Capita vs Life Expectancy (1952 – 2007)",
        labels={
            "gdp_per_capita": "GDP per Capita (log scale, USD)",
            "life_expectancy": "Life Expectancy (years)",
        },
    )
    fig.update_layout(
        font_family="Arial",
        title_font_size=18,
        legend_title_text="Continent",
    )
    out = os.path.join(OUTPUT_DIR, "bubble_chart.html")
    fig.write_html(out)
    print(f"    Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Regional heatmap — avg life expectancy by continent × decade
# ═══════════════════════════════════════════════════════════════════════════════

def regional_heatmap(df: pd.DataFrame) -> None:
    print("\n🗺️   Building regional heatmap …")
    df = df.copy()
    df["decade"] = (df["year"] // 10 * 10).astype(str) + "s"

    pivot = (
        df.groupby(["continent", "decade"])["life_expectancy"]
        .mean()
        .round(1)
        .unstack("decade")
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Avg Life Expectancy (yrs)"},
    )
    ax.set_title("Average Life Expectancy by Continent and Decade", fontsize=15, pad=12)
    ax.set_xlabel("Decade")
    ax.set_ylabel("Continent")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "regional_heatmap.png")
    plt.savefig(out)
    plt.close()
    print(f"    Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Scatter / pairplot matrix
# ═══════════════════════════════════════════════════════════════════════════════

def scatter_matrix(df: pd.DataFrame) -> None:
    print("\n📊  Building scatter matrix …")
    # Use only the latest year for clarity
    latest = df[df["year"] == df["year"].max()].copy()
    latest["log_gdp"] = np.log10(latest["gdp_per_capita"])
    latest["log_pop"] = np.log10(latest["population"])

    g = sns.pairplot(
        latest[["life_expectancy", "log_gdp", "log_pop", "continent"]],
        hue="continent",
        diag_kind="kde",
        plot_kws={"alpha": 0.7, "s": 60},
        corner=True,
    )
    g.fig.suptitle(f"Scatter Matrix — {df['year'].max()} (log-scaled GDP & Pop)", y=1.02, fontsize=14)

    out = os.path.join(OUTPUT_DIR, "scatter_matrix.png")
    g.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"    Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. K-Means clustering of countries (latest year)
# ═══════════════════════════════════════════════════════════════════════════════

CLUSTER_LABELS = {0: "Developing", 1: "Emerging", 2: "Advanced", 3: "Frontier"}

def cluster_countries(df: pd.DataFrame) -> pd.DataFrame:
    print("\n🔬  Clustering countries by development indicators …")
    latest = df[df["year"] == df["year"].max()].copy()

    features = ["life_expectancy", "gdp_per_capita", "population"]
    X = latest[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow method — pick k=4 for clear groups
    k = 4
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    latest["cluster"] = km.fit_predict(X_scaled)

    # Sort clusters by average life expectancy so labels are stable
    order = (
        latest.groupby("cluster")["life_expectancy"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    remap = {old: new for new, old in enumerate(order)}
    latest["cluster"] = latest["cluster"].map(remap)
    latest["cluster_label"] = latest["cluster"].map(CLUSTER_LABELS)

    # PCA → 2-D for visualization
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    latest["pca1"] = coords[:, 0]
    latest["pca2"] = coords[:, 1]

    # Plot
    palette = {"Developing": "#e74c3c", "Emerging": "#f39c12",
               "Advanced": "#27ae60", "Frontier": "#2980b9"}
    fig, ax = plt.subplots(figsize=(10, 7))
    for label, grp in latest.groupby("cluster_label"):
        ax.scatter(grp["pca1"], grp["pca2"],
                   label=label, alpha=0.75, s=80, color=palette[label])
        # Annotate a few notable countries
        for _, row in grp.nlargest(2, "population").iterrows():
            ax.annotate(row["country"], (row["pca1"], row["pca2"]),
                        fontsize=7, alpha=0.8,
                        xytext=(4, 4), textcoords="offset points")

    var_exp = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% variance)", fontsize=11)
    ax.set_title(f"K-Means Clusters (k=4) — Country Development Groups ({df['year'].max()})",
                 fontsize=14, pad=10)
    ax.legend(title="Cluster", fontsize=10)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "clusters.png")
    plt.savefig(out)
    plt.close()
    print(f"    Saved → {out}")

    # Print cluster summary
    summary = (
        latest.groupby("cluster_label")[["life_expectancy", "gdp_per_capita"]]
        .mean()
        .round(1)
    )
    print("\n    Cluster summary (averages):")
    print(summary.to_string())
    return latest


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Top life expectancy gainers 1952 → 2007
# ═══════════════════════════════════════════════════════════════════════════════

def top_gainers(df: pd.DataFrame, n: int = 15) -> None:
    print(f"\n🏆  Finding top {n} life-expectancy gainers …")
    first_year = df["year"].min()
    last_year  = df["year"].max()

    start = df[df["year"] == first_year][["country", "continent", "life_expectancy"]].rename(
        columns={"life_expectancy": "le_start"})
    end   = df[df["year"] == last_year][["country", "life_expectancy"]].rename(
        columns={"life_expectancy": "le_end"})

    merged = start.merge(end, on="country")
    merged["gain"] = merged["le_end"] - merged["le_start"]
    top = merged.nlargest(n, "gain").sort_values("gain")

    # Color by continent
    cont_colors = {
        "Africa": "#e67e22", "Americas": "#3498db",
        "Asia": "#e74c3c", "Europe": "#2ecc71", "Oceania": "#9b59b6",
    }
    colors = [cont_colors.get(c, "gray") for c in top["continent"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(top["country"], top["gain"], color=colors)
    ax.bar_label(bars, fmt="%.1f yrs", padding=4, fontsize=9)
    ax.set_xlabel("Life Expectancy Gain (years)", fontsize=11)
    ax.set_title(f"Top {n} Countries by Life Expectancy Gain\n({first_year} → {last_year})",
                 fontsize=14, pad=10)
    ax.set_xlim(0, top["gain"].max() + 8)

    # Legend for continents
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=v, label=k) for k, v in cont_colors.items()
                      if k in top["continent"].values]
    ax.legend(handles=legend_patches, title="Continent", fontsize=9, loc="lower right")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "top_gainers.png")
    plt.savefig(out)
    plt.close()
    print(f"    Saved → {out}")
    print(f"\n    #1 gainer: {top.iloc[-1]['country']} "
          f"(+{top.iloc[-1]['gain']:.1f} years)")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Summary statistics CSV
# ═══════════════════════════════════════════════════════════════════════════════

def export_summary(df: pd.DataFrame) -> None:
    print("\n📋  Exporting summary statistics …")
    latest = df[df["year"] == df["year"].max()]
    summary = (
        latest.groupby("continent")
        .agg(
            countries=("country", "count"),
            avg_life_expectancy=("life_expectancy", "mean"),
            median_gdp=("gdp_per_capita", "median"),
            total_population=("population", "sum"),
        )
        .round({"avg_life_expectancy": 1, "median_gdp": 0})
        .reset_index()
    )
    summary["total_population"] = summary["total_population"].apply(
        lambda x: f"{x/1e9:.2f}B" if x >= 1e9 else f"{x/1e6:.0f}M"
    )
    out = os.path.join(OUTPUT_DIR, "summary_stats.csv")
    summary.to_csv(out, index=False)
    print(f"    Saved → {out}")
    print(f"\n    {summary.to_string(index=False)}")


# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Gapminder Life Expectancy & GDP Correlation Explorer")
    print("=" * 60)

    df = load_gapminder()

    animated_bubble_chart(df)
    regional_heatmap(df)
    scatter_matrix(df)
    cluster_countries(df)
    top_gainers(df)
    export_summary(df)

    print("\n✅  All outputs saved to ./outputs/")
    print("    Open bubble_chart.html in a browser for the animated chart.")


if __name__ == "__main__":
    main()
