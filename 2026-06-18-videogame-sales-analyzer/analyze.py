"""
Video Game Sales & Genre Evolution Analyzer
============================================
A comprehensive data analysis and visualization project that explores
video game industry trends from 1980 to 2023 using sales data.

Topics covered:
- Genre popularity over time (how genres rose and fell)
- Regional market differences (NA vs EU vs JP)
- Publisher dominance analysis
- Platform lifecycle visualization
- Decade-by-decade market evolution
- Correlation between regions

Dataset: Curated sample of ~120 notable titles (data/vgsales_sample.csv)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


# ─── Configuration ────────────────────────────────────────────────────────────

DATA_PATH = Path(__file__).parent / "data" / "vgsales_sample.csv"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Consistent color palette across the project
PALETTE = sns.color_palette("tab10", 10)
GENRE_COLORS = {
    "Action": "#e74c3c",
    "Action-Adventure": "#e67e22",
    "Platform": "#f1c40f",
    "Role-Playing": "#2ecc71",
    "Shooter": "#3498db",
    "Sports": "#9b59b6",
    "Racing": "#1abc9c",
    "Fighting": "#e91e63",
    "Simulation": "#607d8b",
    "Misc": "#95a5a6",
    "Puzzle": "#00bcd4",
    "Strategy": "#795548",
}

# Era buckets for decade analysis
ERA_BINS = [1979, 1989, 1999, 2009, 2019, 2025]
ERA_LABELS = ["1980s", "1990s", "2000s", "2010s", "2020s"]


# ─── Data Loading & Cleaning ──────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load and preprocess the video game sales CSV."""
    df = pd.read_csv(DATA_PATH)

    # Drop rows missing critical fields
    df = df.dropna(subset=["Year", "Genre", "Global_Sales"])

    # Cast year to int (some datasets have floats due to missing values)
    df["Year"] = df["Year"].astype(int)

    # Add era column
    df["Era"] = pd.cut(df["Year"], bins=ERA_BINS, labels=ERA_LABELS, right=True)

    # Normalize genre: collapse sub-genres for cleaner visuals
    df["Genre_Clean"] = df["Genre"].replace({"Action-Adventure": "Action-Adventure"})

    print(f"✓ Loaded {len(df)} game records spanning {df['Year'].min()}–{df['Year'].max()}")
    print(f"  Genres: {sorted(df['Genre_Clean'].unique())}")
    print(f"  Platforms: {df['Platform'].nunique()} unique platforms")
    print(f"  Publishers: {df['Publisher'].nunique()} unique publishers")
    return df


# ─── Analysis Functions ───────────────────────────────────────────────────────

def top_publishers(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return top N publishers by total global sales."""
    pub = (
        df.groupby("Publisher")["Global_Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    pub.columns = ["Publisher", "Total_Sales_M"]
    return pub


def genre_sales_by_era(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot: rows = genre, columns = era, values = total global sales."""
    return (
        df.groupby(["Genre_Clean", "Era"])["Global_Sales"]
        .sum()
        .unstack(fill_value=0)
    )


def regional_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate regional sales percentages."""
    cols = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
    totals = df[cols].sum()
    shares = (totals / totals.sum() * 100).round(1)
    shares.index = ["North America", "Europe", "Japan", "Rest of World"]
    return shares


def genre_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """For each genre, show how much of global sales comes from each region."""
    regions = {
        "NA": "NA_Sales",
        "EU": "EU_Sales",
        "JP": "JP_Sales",
        "Other": "Other_Sales",
    }
    result = {}
    for label, col in regions.items():
        result[label] = df.groupby("Genre_Clean")[col].sum()
    pivot = pd.DataFrame(result)
    pivot["Total"] = pivot.sum(axis=1)
    # Normalize to percentages
    for col in ["NA", "EU", "JP", "Other"]:
        pivot[col + "_pct"] = pivot[col] / pivot["Total"] * 100
    return pivot.sort_values("Total", ascending=False)


def platform_era_share(df: pd.DataFrame) -> pd.DataFrame:
    """Sales share per platform family per era (collapsed by manufacturer)."""
    # Group platforms into families
    family_map = {
        "NES": "Nintendo Home", "SNES": "Nintendo Home", "N64": "Nintendo Home",
        "GC": "Nintendo Home", "Wii": "Nintendo Home", "WiiU": "Nintendo Home",
        "NS": "Nintendo Switch",
        "GB": "Nintendo Portable", "GBA": "Nintendo Portable", "DS": "Nintendo Portable",
        "3DS": "Nintendo Portable",
        "PS": "PlayStation", "PS2": "PlayStation", "PS3": "PlayStation",
        "PS4": "PlayStation", "PS5": "PlayStation",
        "XB": "Xbox", "X360": "Xbox", "XOne": "Xbox", "XSX": "Xbox",
        "PC": "PC", "2600": "Atari", "GEN": "Sega", "SAT": "Sega",
    }
    df2 = df.copy()
    df2["Family"] = df2["Platform"].map(family_map).fillna("Other")
    return (
        df2.groupby(["Era", "Family"])["Global_Sales"]
        .sum()
        .unstack(fill_value=0)
    )


# ─── Plotting Functions ────────────────────────────────────────────────────────

def plot_top_publishers(pub_df: pd.DataFrame, ax: plt.Axes) -> None:
    """Horizontal bar chart of top publishers."""
    colors = [GENRE_COLORS.get("Action", "#3498db")] * len(pub_df)
    # Highlight Nintendo
    bar_colors = ["#e74c3c" if "Nintendo" in p else "#3498db" for p in pub_df["Publisher"]]

    bars = ax.barh(
        pub_df["Publisher"][::-1],
        pub_df["Total_Sales_M"][::-1],
        color=bar_colors[::-1],
        edgecolor="white",
        linewidth=0.5,
    )
    # Value labels
    for bar, val in zip(bars, pub_df["Total_Sales_M"][::-1]):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}M",
            va="center",
            fontsize=8,
            color="#333333",
        )
    ax.set_xlabel("Total Global Sales (millions)", fontsize=9)
    ax.set_title("Top 10 Publishers by Global Sales", fontweight="bold", fontsize=11)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, pub_df["Total_Sales_M"].max() * 1.18)


def plot_genre_heatmap(era_df: pd.DataFrame, ax: plt.Axes) -> None:
    """Heatmap showing genre sales strength across eras."""
    # Normalize each era column so we compare share, not absolute volume
    normed = era_df.div(era_df.sum(axis=0), axis=1) * 100
    normed = normed.dropna(how="all")

    sns.heatmap(
        normed,
        ax=ax,
        cmap="YlOrRd",
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={"label": "% of era sales"},
        annot_kws={"size": 7},
    )
    ax.set_title("Genre Share (%) by Era", fontweight="bold", fontsize=11)
    ax.set_xlabel("Era", fontsize=9)
    ax.set_ylabel("Genre", fontsize=9)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)


def plot_regional_pie(shares: pd.Series, ax: plt.Axes) -> None:
    """Pie chart of regional sales share."""
    region_colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]
    wedges, texts, autotexts = ax.pie(
        shares,
        labels=shares.index,
        autopct="%1.1f%%",
        colors=region_colors,
        startangle=140,
        pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for t in autotexts:
        t.set_fontsize(8)
    for t in texts:
        t.set_fontsize(8)
    ax.set_title("Regional Sales Distribution\n(All Titles Combined)", fontweight="bold", fontsize=11)


def plot_genre_region_bars(genre_region: pd.DataFrame, ax: plt.Axes) -> None:
    """Stacked bar chart: for each genre, what % comes from each region."""
    top_genres = genre_region.head(10)
    regions = ["NA_pct", "EU_pct", "JP_pct", "Other_pct"]
    region_labels = ["North America", "Europe", "Japan", "Other"]
    region_colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]

    bottom = np.zeros(len(top_genres))
    for col, label, color in zip(regions, region_labels, region_colors):
        ax.bar(
            range(len(top_genres)),
            top_genres[col],
            bottom=bottom,
            label=label,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += top_genres[col].values

    ax.set_xticks(range(len(top_genres)))
    ax.set_xticklabels(top_genres.index, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("% of Genre Sales", fontsize=9)
    ax.set_title("Regional Breakdown by Genre\n(Top 10 Genres by Total Sales)", fontweight="bold", fontsize=11)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def plot_platform_stacked_area(era_platform: pd.DataFrame, ax: plt.Axes) -> None:
    """Stacked area chart of platform family sales per era."""
    # Keep only the biggest families for readability
    top_families = era_platform.sum(axis=0).nlargest(6).index
    data = era_platform[top_families]

    family_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
    x = range(len(data.index))

    ax.stackplot(
        x,
        [data[col] for col in data.columns],
        labels=list(data.columns),
        colors=family_colors[: len(data.columns)],
        alpha=0.8,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(data.index, fontsize=9)
    ax.set_ylabel("Total Sales (millions)", fontsize=9)
    ax.set_title("Platform Family Sales by Era", fontweight="bold", fontsize=11)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def plot_top_games_scatter(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Scatter: Year vs Global Sales, sized by NA sales, colored by genre."""
    # Focus on notable titles (> 5M global sales for clarity)
    notable = df[df["Global_Sales"] >= 5.0].copy()
    genres_present = notable["Genre_Clean"].unique()

    for genre in genres_present:
        subset = notable[notable["Genre_Clean"] == genre]
        color = GENRE_COLORS.get(genre, "#aaaaaa")
        ax.scatter(
            subset["Year"],
            subset["Global_Sales"],
            s=subset["NA_Sales"] * 5 + 20,  # size proportional to NA sales
            color=color,
            alpha=0.70,
            label=genre,
            edgecolors="white",
            linewidths=0.4,
        )

    # Annotate a few landmark titles
    landmarks = [
        ("Wii Sports", 82.74, 2006),
        ("Minecraft PC", 238.00, 2011),
        ("GTA V PC", 185.00, 2015),
        ("Animal Crossing NH", 31.18, 2020),
        ("Pokemon Red/Blue", 31.37, 1996),
    ]
    for name, sales, year in landmarks:
        row = df[df["Global_Sales"].between(sales - 1, sales + 1)]
        if not row.empty:
            ax.annotate(
                name,
                xy=(year, sales),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=6.5,
                color="#222222",
            )

    ax.set_xlabel("Year", fontsize=9)
    ax.set_ylabel("Global Sales (millions)", fontsize=9)
    ax.set_title("Sales by Year — Each Bubble is a Title\n(Size = NA Sales)", fontweight="bold", fontsize=11)
    ax.legend(
        fontsize=6,
        loc="upper left",
        ncol=2,
        markerscale=0.8,
        framealpha=0.7,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Video Game Sales & Genre Evolution Analyzer")
    print("=" * 60)

    # Load data
    df = load_data()

    # Run analyses
    pub_df = top_publishers(df, n=10)
    era_genre = genre_sales_by_era(df)
    reg_shares = regional_shares(df)
    genre_reg = genre_by_region(df)
    plat_era = platform_era_share(df)

    # ── Print text summaries ──────────────────────────────────────
    print("\n📊 Top 5 Publishers:")
    print(pub_df.head(5).to_string(index=False))

    print("\n🌍 Regional Sales Share:")
    for region, pct in reg_shares.items():
        print(f"  {region:20s} {pct:.1f}%")

    print("\n🎮 Genre Sales by Era (top genres, millions):")
    top_genres_era = era_genre.loc[
        era_genre.sum(axis=1).nlargest(6).index
    ]
    print(top_genres_era.to_string())

    print("\n📈 Most Dominant Genre per Era:")
    for era in era_genre.columns:
        top = era_genre[era].idxmax()
        val = era_genre[era].max()
        print(f"  {str(era):8s}  →  {top:20s}  ({val:.1f}M)")

    # ── Build the 6-panel dashboard ──────────────────────────────
    print("\n🎨 Generating dashboard...")
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#f9f9f9")

    # Title banner
    fig.suptitle(
        "Video Game Industry: Sales & Genre Evolution (1980–2023)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
        color="#1a1a2e",
    )

    # Create a 3×2 grid
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35,
                          top=0.93, bottom=0.06, left=0.07, right=0.97)

    ax1 = fig.add_subplot(gs[0, 0])   # Top publishers
    ax2 = fig.add_subplot(gs[0, 1])   # Genre heatmap by era
    ax3 = fig.add_subplot(gs[1, 0])   # Regional pie
    ax4 = fig.add_subplot(gs[1, 1])   # Genre × region stacked bar
    ax5 = fig.add_subplot(gs[2, 0])   # Platform era area chart
    ax6 = fig.add_subplot(gs[2, 1])   # Scatter of notable titles

    plot_top_publishers(pub_df, ax1)
    plot_genre_heatmap(era_genre, ax2)
    plot_regional_pie(reg_shares, ax3)
    plot_genre_region_bars(genre_reg, ax4)
    plot_platform_stacked_area(plat_era, ax5)
    plot_top_games_scatter(df, ax6)

    # Footer
    fig.text(
        0.5, 0.01,
        "Data: Curated sample of ~120 notable titles | Analysis by weekly-ai-projects",
        ha="center", fontsize=7, color="#888888",
    )

    out_path = OUTPUT_DIR / "vg_sales_dashboard.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✓ Dashboard saved → {out_path}")

    # ── Single-focus: Genre Evolution Line Chart ──────────────────
    fig2, ax = plt.subplots(figsize=(12, 6))
    fig2.patch.set_facecolor("#f9f9f9")

    top6_genres = era_genre.sum(axis=1).nlargest(6).index.tolist()
    era_labels_str = [str(e) for e in era_genre.columns]

    for genre in top6_genres:
        values = era_genre.loc[genre].values.astype(float)
        color = GENRE_COLORS.get(genre, "#aaaaaa")
        ax.plot(era_labels_str, values, marker="o", linewidth=2.5,
                markersize=7, label=genre, color=color)
        # label the last point
        ax.annotate(
            genre,
            xy=(era_labels_str[-1], values[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=8,
            color=color,
            va="center",
        )

    ax.set_xlabel("Era", fontsize=11)
    ax.set_ylabel("Total Sales (millions)", fontsize=11)
    ax.set_title("Genre Sales Trajectory Across Eras (1980–2023)",
                 fontweight="bold", fontsize=13)
    ax.legend(fontsize=9, framealpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#f9f9f9")

    line_path = OUTPUT_DIR / "genre_trajectory.png"
    fig2.savefig(line_path, dpi=150, bbox_inches="tight",
                 facecolor=fig2.get_facecolor())
    plt.close(fig2)
    print(f"✓ Genre trajectory saved → {line_path}")

    # ── Correlation: NA vs EU vs JP ───────────────────────────────
    fig3, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig3.patch.set_facecolor("#f9f9f9")
    fig3.suptitle("Regional Sales Correlations", fontweight="bold", fontsize=12)

    region_pairs = [("NA_Sales", "EU_Sales"), ("NA_Sales", "JP_Sales"), ("EU_Sales", "JP_Sales")]
    region_labels_map = {"NA_Sales": "NA", "EU_Sales": "EU", "JP_Sales": "JP"}

    for ax, (x_col, y_col) in zip(axes, region_pairs):
        ax.set_facecolor("#f9f9f9")
        # Color-code by genre
        for genre in df["Genre_Clean"].unique():
            sub = df[df["Genre_Clean"] == genre]
            ax.scatter(sub[x_col], sub[y_col], alpha=0.55, s=30,
                       color=GENRE_COLORS.get(genre, "#aaaaaa"), label=genre)
        # Fit line
        valid = df[[x_col, y_col]].dropna()
        corr = valid.corr().iloc[0, 1]
        z = np.polyfit(valid[x_col], valid[y_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid[x_col].min(), valid[x_col].max(), 100)
        ax.plot(x_line, p(x_line), "k--", linewidth=1.2, alpha=0.6)
        ax.set_xlabel(region_labels_map[x_col] + " Sales (M)", fontsize=9)
        ax.set_ylabel(region_labels_map[y_col] + " Sales (M)", fontsize=9)
        ax.set_title(f"{region_labels_map[x_col]} vs {region_labels_map[y_col]}\nr = {corr:.2f}",
                     fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)

    # One shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig3.legend(handles, labels, loc="lower center", ncol=6,
                fontsize=7, framealpha=0.7, bbox_to_anchor=(0.5, -0.08))

    corr_path = OUTPUT_DIR / "regional_correlations.png"
    fig3.savefig(corr_path, dpi=150, bbox_inches="tight",
                 facecolor=fig3.get_facecolor())
    plt.close(fig3)
    print(f"✓ Regional correlations saved → {corr_path}")

    print("\n✅ Analysis complete! Check the outputs/ folder for all charts.")
    print("=" * 60)


if __name__ == "__main__":
    main()
