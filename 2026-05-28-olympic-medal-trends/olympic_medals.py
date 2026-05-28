"""
Olympic Games Medal Trends Visualizer
======================================
A comprehensive data analysis and visualization of Olympic Games medal history
across nations, decades, and game types (Summer/Winter).

This script generates four rich visualizations:
  1. Top 10 nations by total medals (all-time)
  2. Medal composition (Gold/Silver/Bronze) for top nations
  3. Medal trends over time for major powerhouse nations
  4. Heatmap of medal distribution by country and decade

All data is embedded directly in the script — no external downloads needed.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# DATA — Historical Olympic medal totals (Summer + Winter combined)
# Source: Condensed from public Olympic databases (1896–2022)
# Each row: (country, year, game_type, gold, silver, bronze)
# ──────────────────────────────────────────────────────────────────────────────

raw_data = [
    # USA — Summer
    ("USA", 1896, "Summer", 11, 7, 2), ("USA", 1900, "Summer", 19, 14, 14),
    ("USA", 1904, "Summer", 78, 82, 79), ("USA", 1908, "Summer", 23, 12, 12),
    ("USA", 1912, "Summer", 25, 19, 19), ("USA", 1920, "Summer", 41, 27, 27),
    ("USA", 1924, "Summer", 45, 27, 27), ("USA", 1928, "Summer", 22, 18, 16),
    ("USA", 1932, "Summer", 41, 32, 30), ("USA", 1936, "Summer", 24, 20, 12),
    ("USA", 1948, "Summer", 38, 27, 19), ("USA", 1952, "Summer", 40, 19, 17),
    ("USA", 1956, "Summer", 32, 25, 17), ("USA", 1960, "Summer", 34, 21, 16),
    ("USA", 1964, "Summer", 36, 26, 28), ("USA", 1968, "Summer", 45, 28, 34),
    ("USA", 1972, "Summer", 33, 31, 30), ("USA", 1976, "Summer", 34, 35, 25),
    ("USA", 1984, "Summer", 83, 61, 30), ("USA", 1988, "Summer", 36, 31, 27),
    ("USA", 1992, "Summer", 37, 34, 37), ("USA", 1996, "Summer", 44, 32, 25),
    ("USA", 2000, "Summer", 36, 24, 32), ("USA", 2004, "Summer", 35, 39, 29),
    ("USA", 2008, "Summer", 36, 38, 36), ("USA", 2012, "Summer", 46, 29, 29),
    ("USA", 2016, "Summer", 46, 37, 38), ("USA", 2020, "Summer", 39, 41, 33),
    # USA — Winter
    ("USA", 1924, "Winter", 1, 2, 1), ("USA", 1932, "Winter", 6, 4, 2),
    ("USA", 1936, "Winter", 1, 0, 3), ("USA", 1948, "Winter", 3, 4, 2),
    ("USA", 1952, "Winter", 4, 6, 1), ("USA", 1956, "Winter", 2, 3, 2),
    ("USA", 1960, "Winter", 3, 4, 3), ("USA", 1964, "Winter", 1, 2, 3),
    ("USA", 1968, "Winter", 1, 5, 1), ("USA", 1972, "Winter", 3, 2, 3),
    ("USA", 1976, "Winter", 3, 3, 4), ("USA", 1980, "Winter", 6, 4, 2),
    ("USA", 1984, "Winter", 4, 4, 0), ("USA", 1988, "Winter", 2, 1, 3),
    ("USA", 1992, "Winter", 5, 4, 2), ("USA", 1994, "Winter", 6, 5, 2),
    ("USA", 1998, "Winter", 6, 3, 4), ("USA", 2002, "Winter", 10, 13, 11),
    ("USA", 2006, "Winter", 9, 9, 7), ("USA", 2010, "Winter", 9, 15, 13),
    ("USA", 2014, "Winter", 9, 7, 12), ("USA", 2018, "Winter", 9, 8, 6),
    ("USA", 2022, "Winter", 8, 10, 7),

    # Soviet Union / Russia
    ("USSR/Russia", 1952, "Summer", 22, 30, 19), ("USSR/Russia", 1956, "Summer", 37, 29, 32),
    ("USSR/Russia", 1960, "Summer", 43, 29, 31), ("USSR/Russia", 1964, "Summer", 30, 31, 35),
    ("USSR/Russia", 1968, "Summer", 29, 32, 30), ("USSR/Russia", 1972, "Summer", 50, 27, 22),
    ("USSR/Russia", 1976, "Summer", 49, 41, 35), ("USSR/Russia", 1980, "Summer", 80, 69, 46),
    ("USSR/Russia", 1988, "Summer", 55, 31, 46), ("USSR/Russia", 1992, "Summer", 45, 38, 29),
    ("USSR/Russia", 1996, "Summer", 26, 21, 16), ("USSR/Russia", 2000, "Summer", 32, 28, 28),
    ("USSR/Russia", 2004, "Summer", 27, 27, 38), ("USSR/Russia", 2008, "Summer", 23, 21, 28),
    ("USSR/Russia", 2012, "Summer", 24, 25, 33), ("USSR/Russia", 2016, "Summer", 19, 17, 20),
    ("USSR/Russia", 2020, "Summer", 20, 28, 23),
    ("USSR/Russia", 1956, "Winter", 7, 3, 6), ("USSR/Russia", 1960, "Winter", 7, 5, 9),
    ("USSR/Russia", 1964, "Winter", 11, 8, 6), ("USSR/Russia", 1968, "Winter", 5, 5, 3),
    ("USSR/Russia", 1972, "Winter", 8, 5, 3), ("USSR/Russia", 1976, "Winter", 13, 6, 8),
    ("USSR/Russia", 1980, "Winter", 10, 6, 6), ("USSR/Russia", 1984, "Winter", 6, 10, 9),
    ("USSR/Russia", 1988, "Winter", 11, 9, 9), ("USSR/Russia", 1992, "Winter", 9, 6, 8),
    ("USSR/Russia", 1994, "Winter", 11, 8, 4), ("USSR/Russia", 1998, "Winter", 9, 6, 3),
    ("USSR/Russia", 2002, "Winter", 5, 4, 4), ("USSR/Russia", 2006, "Winter", 8, 6, 8),
    ("USSR/Russia", 2010, "Winter", 3, 5, 7), ("USSR/Russia", 2014, "Winter", 13, 11, 9),
    ("USSR/Russia", 2018, "Winter", 2, 6, 9), ("USSR/Russia", 2022, "Winter", 6, 12, 14),

    # Germany (East+West+Unified combined)
    ("Germany", 1896, "Summer", 25, 5, 2), ("Germany", 1900, "Summer", 4, 2, 2),
    ("Germany", 1928, "Summer", 10, 7, 14), ("Germany", 1932, "Summer", 3, 12, 5),
    ("Germany", 1936, "Summer", 38, 36, 26), ("Germany", 1952, "Summer", 0, 7, 17),
    ("Germany", 1956, "Summer", 6, 13, 7), ("Germany", 1960, "Summer", 12, 19, 11),
    ("Germany", 1964, "Summer", 10, 22, 18), ("Germany", 1968, "Summer", 9, 9, 6),
    ("Germany", 1972, "Summer", 20, 23, 23), ("Germany", 1976, "Summer", 10, 12, 17),
    ("Germany", 1980, "Summer", 3, 10, 20), ("Germany", 1984, "Summer", 17, 19, 23),
    ("Germany", 1988, "Summer", 11, 14, 15), ("Germany", 1992, "Summer", 33, 21, 28),
    ("Germany", 1996, "Summer", 20, 18, 27), ("Germany", 2000, "Summer", 13, 17, 26),
    ("Germany", 2004, "Summer", 14, 16, 18), ("Germany", 2008, "Summer", 16, 10, 15),
    ("Germany", 2012, "Summer", 11, 19, 14), ("Germany", 2016, "Summer", 17, 10, 15),
    ("Germany", 2020, "Summer", 10, 11, 16),
    ("Germany", 1924, "Winter", 0, 1, 0), ("Germany", 1928, "Winter", 0, 2, 0),
    ("Germany", 1936, "Winter", 3, 3, 0), ("Germany", 1952, "Winter", 3, 2, 2),
    ("Germany", 1956, "Winter", 2, 2, 1), ("Germany", 1960, "Winter", 4, 3, 1),
    ("Germany", 1964, "Winter", 3, 5, 5), ("Germany", 1968, "Winter", 2, 2, 3),
    ("Germany", 1972, "Winter", 3, 1, 1), ("Germany", 1976, "Winter", 2, 5, 3),
    ("Germany", 1980, "Winter", 2, 2, 4), ("Germany", 1984, "Winter", 2, 5, 5),
    ("Germany", 1988, "Winter", 8, 8, 6), ("Germany", 1992, "Winter", 26, 10, 8),
    ("Germany", 1994, "Winter", 9, 7, 8), ("Germany", 1998, "Winter", 12, 9, 8),
    ("Germany", 2002, "Winter", 12, 16, 8), ("Germany", 2006, "Winter", 11, 12, 6),
    ("Germany", 2010, "Winter", 10, 13, 7), ("Germany", 2014, "Winter", 8, 6, 5),
    ("Germany", 2018, "Winter", 14, 10, 7), ("Germany", 2022, "Winter", 12, 10, 5),

    # China
    ("China", 1984, "Summer", 15, 8, 9), ("China", 1988, "Summer", 5, 11, 12),
    ("China", 1992, "Summer", 16, 22, 16), ("China", 1996, "Summer", 16, 22, 12),
    ("China", 2000, "Summer", 28, 16, 15), ("China", 2004, "Summer", 32, 17, 14),
    ("China", 2008, "Summer", 51, 21, 28), ("China", 2012, "Summer", 38, 27, 23),
    ("China", 2016, "Summer", 26, 18, 26), ("China", 2020, "Summer", 38, 32, 18),
    ("China", 2002, "Winter", 2, 2, 4), ("China", 2006, "Winter", 2, 4, 5),
    ("China", 2010, "Winter", 5, 2, 4), ("China", 2014, "Winter", 3, 4, 2),
    ("China", 2018, "Winter", 1, 6, 2), ("China", 2022, "Winter", 9, 4, 2),

    # Great Britain
    ("Great Britain", 1896, "Summer", 2, 3, 2), ("Great Britain", 1900, "Summer", 15, 15, 13),
    ("Great Britain", 1908, "Summer", 56, 51, 39), ("Great Britain", 1912, "Summer", 10, 15, 16),
    ("Great Britain", 1920, "Summer", 14, 15, 13), ("Great Britain", 1924, "Summer", 9, 13, 12),
    ("Great Britain", 1948, "Summer", 3, 14, 6), ("Great Britain", 1952, "Summer", 1, 2, 8),
    ("Great Britain", 1956, "Summer", 6, 7, 11), ("Great Britain", 1960, "Summer", 2, 6, 4),
    ("Great Britain", 1964, "Summer", 4, 12, 2), ("Great Britain", 1968, "Summer", 5, 5, 3),
    ("Great Britain", 1972, "Summer", 4, 5, 9), ("Great Britain", 1976, "Summer", 3, 5, 5),
    ("Great Britain", 1984, "Summer", 5, 10, 22), ("Great Britain", 1988, "Summer", 5, 10, 9),
    ("Great Britain", 1992, "Summer", 5, 3, 12), ("Great Britain", 1996, "Summer", 1, 8, 6),
    ("Great Britain", 2000, "Summer", 11, 10, 7), ("Great Britain", 2004, "Summer", 9, 9, 12),
    ("Great Britain", 2008, "Summer", 19, 13, 15), ("Great Britain", 2012, "Summer", 29, 17, 19),
    ("Great Britain", 2016, "Summer", 27, 23, 17), ("Great Britain", 2020, "Summer", 22, 21, 22),

    # France
    ("France", 1896, "Summer", 5, 4, 2), ("France", 1900, "Summer", 26, 41, 34),
    ("France", 1908, "Summer", 5, 5, 9), ("France", 1912, "Summer", 7, 4, 3),
    ("France", 1920, "Summer", 9, 19, 13), ("France", 1924, "Summer", 13, 15, 10),
    ("France", 1928, "Summer", 6, 10, 5), ("France", 1932, "Summer", 10, 5, 4),
    ("France", 1936, "Summer", 7, 6, 6), ("France", 1948, "Summer", 6, 6, 6),
    ("France", 1952, "Summer", 6, 6, 6), ("France", 1956, "Summer", 4, 4, 6),
    ("France", 1960, "Summer", 5, 2, 3), ("France", 1964, "Summer", 1, 8, 6),
    ("France", 1968, "Summer", 7, 3, 5), ("France", 1972, "Summer", 2, 4, 7),
    ("France", 1976, "Summer", 2, 3, 4), ("France", 1984, "Summer", 5, 7, 16),
    ("France", 1988, "Summer", 6, 4, 6), ("France", 1992, "Summer", 8, 5, 16),
    ("France", 1996, "Summer", 15, 7, 15), ("France", 2000, "Summer", 13, 14, 11),
    ("France", 2004, "Summer", 11, 9, 13), ("France", 2008, "Summer", 7, 16, 18),
    ("France", 2012, "Summer", 11, 11, 12), ("France", 2016, "Summer", 10, 18, 14),
    ("France", 2020, "Summer", 10, 12, 11),

    # Australia
    ("Australia", 1896, "Summer", 2, 0, 1), ("Australia", 1900, "Summer", 2, 0, 4),
    ("Australia", 1948, "Summer", 2, 6, 5), ("Australia", 1952, "Summer", 6, 2, 3),
    ("Australia", 1956, "Summer", 13, 8, 14), ("Australia", 1960, "Summer", 8, 8, 6),
    ("Australia", 1964, "Summer", 6, 2, 10), ("Australia", 1968, "Summer", 5, 7, 5),
    ("Australia", 1972, "Summer", 8, 7, 2), ("Australia", 1976, "Summer", 1, 4, 4),
    ("Australia", 1984, "Summer", 4, 8, 12), ("Australia", 1988, "Summer", 3, 6, 5),
    ("Australia", 1992, "Summer", 7, 9, 11), ("Australia", 1996, "Summer", 9, 9, 23),
    ("Australia", 2000, "Summer", 16, 25, 17), ("Australia", 2004, "Summer", 17, 16, 16),
    ("Australia", 2008, "Summer", 14, 15, 17), ("Australia", 2012, "Summer", 7, 16, 12),
    ("Australia", 2016, "Summer", 8, 11, 10), ("Australia", 2020, "Summer", 17, 7, 22),

    # Japan
    ("Japan", 1920, "Summer", 0, 2, 0), ("Japan", 1928, "Summer", 2, 2, 1),
    ("Japan", 1932, "Summer", 7, 7, 4), ("Japan", 1936, "Summer", 6, 4, 8),
    ("Japan", 1952, "Summer", 1, 6, 2), ("Japan", 1956, "Summer", 4, 10, 5),
    ("Japan", 1960, "Summer", 4, 7, 7), ("Japan", 1964, "Summer", 16, 5, 8),
    ("Japan", 1968, "Summer", 11, 7, 7), ("Japan", 1972, "Summer", 13, 8, 8),
    ("Japan", 1976, "Summer", 9, 6, 10), ("Japan", 1984, "Summer", 10, 8, 14),
    ("Japan", 1988, "Summer", 4, 3, 7), ("Japan", 1992, "Summer", 3, 8, 11),
    ("Japan", 1996, "Summer", 3, 6, 5), ("Japan", 2000, "Summer", 5, 8, 5),
    ("Japan", 2004, "Summer", 16, 9, 12), ("Japan", 2008, "Summer", 9, 6, 10),
    ("Japan", 2012, "Summer", 7, 14, 17), ("Japan", 2016, "Summer", 12, 8, 21),
    ("Japan", 2020, "Summer", 27, 14, 17),

    # Norway (winter powerhouse)
    ("Norway", 1924, "Winter", 4, 7, 6), ("Norway", 1928, "Winter", 6, 4, 5),
    ("Norway", 1932, "Winter", 3, 4, 3), ("Norway", 1936, "Winter", 7, 5, 3),
    ("Norway", 1948, "Winter", 4, 3, 3), ("Norway", 1952, "Winter", 7, 3, 6),
    ("Norway", 1956, "Winter", 2, 1, 1), ("Norway", 1960, "Winter", 3, 3, 0),
    ("Norway", 1964, "Winter", 3, 6, 6), ("Norway", 1968, "Winter", 6, 6, 2),
    ("Norway", 1972, "Winter", 2, 5, 5), ("Norway", 1976, "Winter", 3, 3, 1),
    ("Norway", 1980, "Winter", 1, 3, 6), ("Norway", 1984, "Winter", 3, 2, 4),
    ("Norway", 1988, "Winter", 0, 3, 2), ("Norway", 1992, "Winter", 9, 6, 5),
    ("Norway", 1994, "Winter", 10, 11, 5), ("Norway", 1998, "Winter", 10, 10, 5),
    ("Norway", 2002, "Winter", 11, 7, 6), ("Norway", 2006, "Winter", 2, 8, 9),
    ("Norway", 2010, "Winter", 9, 8, 6), ("Norway", 2014, "Winter", 11, 5, 10),
    ("Norway", 2018, "Winter", 14, 14, 11), ("Norway", 2022, "Winter", 16, 8, 13),
]

# ──────────────────────────────────────────────────────────────────────────────
# BUILD DATAFRAME
# ──────────────────────────────────────────────────────────────────────────────

df = pd.DataFrame(raw_data, columns=["country", "year", "game_type", "gold", "silver", "bronze"])
df["total"] = df["gold"] + df["silver"] + df["bronze"]
df["decade"] = (df["year"] // 10) * 10  # group into decades for heatmap

print("Dataset Overview")
print("=" * 50)
print(f"Total records        : {len(df)}")
print(f"Countries covered    : {df['country'].nunique()}")
print(f"Years covered        : {df['year'].min()} - {df['year'].max()}")
print(f"Total medals in data : {df['total'].sum():,}")
print()

# ──────────────────────────────────────────────────────────────────────────────
# AGGREGATE FOR ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

# All-time totals per country
country_totals = (
    df.groupby("country")[["gold", "silver", "bronze", "total"]]
    .sum()
    .sort_values("total", ascending=False)
)

# Top 10 countries
top10 = country_totals.head(10)
print("Top 10 Nations by Total Medals (All-Time)")
print("-" * 50)
print(top10.to_string())
print()

# Trend over time for top 5 countries
top5_countries = country_totals.head(5).index.tolist()
trend_df = (
    df[df["country"].isin(top5_countries)]
    .groupby(["country", "year"])["total"]
    .sum()
    .reset_index()
)

# Heatmap: medals per country per decade
top8_countries = country_totals.head(8).index.tolist()
heatmap_df = (
    df[df["country"].isin(top8_countries)]
    .groupby(["country", "decade"])["total"]
    .sum()
    .unstack(fill_value=0)
)

# ──────────────────────────────────────────────────────────────────────────────
# VISUALIZATION -- 4-panel figure
# ──────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor("#0d1117")   # dark GitHub-style background
gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

GOLD_COLOR   = "#FFD700"
SILVER_COLOR = "#C0C0C0"
BRONZE_COLOR = "#CD7F32"
TEXT_COLOR   = "#e6edf3"
ACCENT       = "#58a6ff"
GRID_COLOR   = "#21262d"

def style_ax(ax, title):
    """Apply consistent dark-mode styling to an axes object."""
    ax.set_facecolor("#161b22")
    ax.set_title(title, color=TEXT_COLOR, fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.6, alpha=0.7)

# Panel 1: Horizontal bar -- total medals by top 10 nations
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "All-Time Olympic Medals — Top 10 Nations")

colors = [GOLD_COLOR if i == 0 else SILVER_COLOR if i == 1 else BRONZE_COLOR if i == 2 else ACCENT
          for i in range(len(top10))]
bars = ax1.barh(top10.index[::-1], top10["total"][::-1], color=colors[::-1], edgecolor="#0d1117", height=0.7)

for bar, val in zip(bars, top10["total"][::-1]):
    ax1.text(bar.get_width() + 30, bar.get_y() + bar.get_height() / 2,
             f"{val:,}", va="center", ha="left", color=TEXT_COLOR, fontsize=8, fontweight="bold")

ax1.set_xlabel("Total Medals", color=TEXT_COLOR)
ax1.set_xlim(0, top10["total"].max() * 1.15)
ax1.invert_yaxis()

# Panel 2: Stacked bar -- Gold/Silver/Bronze for top 10
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, "Medal Composition — Gold / Silver / Bronze")

y = np.arange(len(top10))
bar_w = 0.6
ax2.barh(y, top10["gold"],   color=GOLD_COLOR,   height=bar_w, label="Gold",   edgecolor="#0d1117")
ax2.barh(y, top10["silver"], color=SILVER_COLOR, height=bar_w, label="Silver",
         left=top10["gold"],                      edgecolor="#0d1117")
ax2.barh(y, top10["bronze"], color=BRONZE_COLOR, height=bar_w, label="Bronze",
         left=top10["gold"] + top10["silver"],    edgecolor="#0d1117")

ax2.set_yticks(y)
ax2.set_yticklabels(top10.index, color=TEXT_COLOR, fontsize=8)
ax2.invert_yaxis()
ax2.set_xlabel("Medal Count", color=TEXT_COLOR)
ax2.legend(loc="lower right", facecolor="#21262d", edgecolor=GRID_COLOR,
           labelcolor=TEXT_COLOR, fontsize=9)

# Panel 3: Line chart -- medal trends over time
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3, "Medal Trends Over Time — Top 5 Nations")

palette = [GOLD_COLOR, SILVER_COLOR, "#ff6b6b", "#69db7c", "#74c0fc"]
for nation, color in zip(top5_countries, palette):
    subset = trend_df[trend_df["country"] == nation].sort_values("year")
    ax3.plot(subset["year"], subset["total"], marker="o", markersize=4,
             linewidth=2, label=nation, color=color, alpha=0.9)

ax3.set_xlabel("Olympic Year", color=TEXT_COLOR)
ax3.set_ylabel("Total Medals Won", color=TEXT_COLOR)
ax3.xaxis.set_major_locator(mticker.MultipleLocator(16))
ax3.legend(facecolor="#21262d", edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8, loc="upper left")

# Panel 4: Heatmap -- medals by country x decade
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor("#161b22")
ax4.set_title("Medal Heatmap by Country & Decade", color=TEXT_COLOR, fontsize=13,
              fontweight="bold", pad=10)

sns.heatmap(
    heatmap_df,
    ax=ax4,
    cmap="YlOrRd",
    linewidths=0.5,
    linecolor="#0d1117",
    annot=True,
    fmt="d",
    annot_kws={"size": 7, "color": "#0d1117"},
    cbar_kws={"shrink": 0.8},
)
ax4.set_xlabel("Decade", color=TEXT_COLOR, fontsize=9)
ax4.set_ylabel("")
ax4.tick_params(colors=TEXT_COLOR, labelsize=8)
ax4.set_xticklabels([str(int(c)) + "s" for c in heatmap_df.columns], color=TEXT_COLOR, rotation=45, ha="right")
ax4.set_yticklabels(ax4.get_yticklabels(), color=TEXT_COLOR, rotation=0)
cbar = ax4.collections[0].colorbar
cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=7)

# Main title
fig.suptitle("Olympic Games Medal Trends Visualizer  |  1896 - 2022",
             fontsize=18, fontweight="bold", color=TEXT_COLOR, y=0.98)

# Save
output_file = "olympic_medal_analysis.png"
plt.savefig(output_file, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Visualization saved: {output_file}")
plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# BONUS: Print summary statistics to terminal
# ──────────────────────────────────────────────────────────────────────────────

print("\n── Fun Facts ──────────────────────────────────────────────")
most_gold_country = country_totals["gold"].idxmax()
print(f"Most gold medals all-time : {most_gold_country} ({country_totals.loc[most_gold_country, 'gold']:,})")

best_winter = df[df["game_type"] == "Winter"].groupby("country")["total"].sum().idxmax()
best_winter_count = df[df["game_type"] == "Winter"].groupby("country")["total"].sum().max()
print(f"Best Winter Olympics nation: {best_winter} ({best_winter_count:,} medals)")

best_single_games = df.loc[df["total"].idxmax()]
print(f"Most medals in a single Games: {best_single_games['country']} "
      f"at {best_single_games['year']} {best_single_games['game_type']} "
      f"({int(best_single_games['total'])} medals)")

gold_efficiency = (
    country_totals[country_totals["total"] >= 50]
    .assign(gold_rate=lambda x: x["gold"] / x["total"])
    .sort_values("gold_rate", ascending=False)
    .head(3)
)
print("\nHighest Gold-Medal Efficiency (nations with >=50 total medals):")
for nation, row in gold_efficiency.iterrows():
    print(f"  {nation}: {row['gold_rate']:.1%} gold rate")
