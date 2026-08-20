"""
coffee_trade_analysis.py
-------------------------
Global Coffee Trade & Consumption Analyzer

Loads `coffee_trade_data.csv` (generate it first with `generate_data.py`)
and produces a set of analyses and visualizations covering:

  1. Production trends over time for the top 10 coffee-producing nations
  2. Market share evolution (who dominates global supply, and how it's shifting)
  3. Correlation between world coffee prices and export revenue
  4. Domestic consumption vs. production ("who drinks what they grow")
  5. A regional breakdown of total 2023 production
  6. A simple linear trend forecast of Vietnam's production through 2028
     (illustrates basic time-series extrapolation with numpy/sklearn)

All charts are saved as PNG files into an `output/` folder, and a plain-text
summary of key findings is printed to the console and written to
`output/summary.txt`.

Usage:
    python generate_data.py
    python coffee_trade_analysis.py
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

sns.set_theme(style="whitegrid", palette="deep")
OUTPUT_DIR = "output"


def load_data(path="coffee_trade_data.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Run `python generate_data.py` first to "
            "generate the sample dataset."
        )
    return pd.read_csv(path)


def plot_production_trends(df, out_dir):
    """Line chart: production (million bags) over time per country."""
    pivot = df.pivot(index="year", columns="country", values="production_million_bags")
    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(ax=ax, linewidth=2)
    ax.set_title("Coffee Production by Country, 2000-2023", fontsize=14, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Production (million 60kg bags)")
    ax.legend(title="Country", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_production_trends.png"), dpi=150)
    plt.close(fig)


def plot_market_share(df, out_dir):
    """Stacked area chart: share of global production per country over time."""
    pivot = df.pivot(index="year", columns="country", values="production_million_bags")
    share = pivot.div(pivot.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.stackplot(share.index, share.T.values, labels=share.columns, alpha=0.85)
    ax.set_title("Global Coffee Production Market Share, 2000-2023", fontsize=14, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of total production (%)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    ax.set_xlim(share.index.min(), share.index.max())
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_market_share.png"), dpi=150)
    plt.close(fig)


def plot_price_vs_exports(df, out_dir):
    """Scatter + regression: world price index vs export revenue (Brazil)."""
    brazil = df[df["country"] == "Brazil"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(
        data=brazil, x="world_price_index", y="export_value_million_usd",
        ax=ax, scatter_kws={"s": 50, "alpha": 0.7}, line_kws={"color": "firebrick"},
    )
    corr = brazil["world_price_index"].corr(brazil["export_value_million_usd"])
    ax.set_title(f"Brazil: World Price Index vs. Export Revenue (r={corr:.2f})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("World Coffee Price Index")
    ax.set_ylabel("Export Value (million USD)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_price_vs_exports.png"), dpi=150)
    plt.close(fig)
    return corr


def plot_consumption_vs_production(df, out_dir):
    """Bubble scatter: 2023 production vs. domestic consumption per capita."""
    latest = df[df["year"] == df["year"].max()]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    sizes = latest["export_value_million_usd"] / latest["export_value_million_usd"].max() * 1200 + 80
    scatter = ax.scatter(
        latest["production_million_bags"],
        latest["domestic_consumption_kg_per_capita"],
        s=sizes, c=latest["export_value_million_usd"], cmap="YlOrBr",
        edgecolor="black", alpha=0.85,
    )
    for _, row in latest.iterrows():
        ax.annotate(row["country"], (row["production_million_bags"], row["domestic_consumption_kg_per_capita"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_title(f"Production vs. Domestic Consumption ({int(latest['year'].iloc[0])})\n"
                 "bubble size/color = export revenue", fontsize=13, fontweight="bold")
    ax.set_xlabel("Production (million bags)")
    ax.set_ylabel("Domestic consumption (kg per capita)")
    fig.colorbar(scatter, ax=ax, label="Export value (million USD)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_consumption_vs_production.png"), dpi=150)
    plt.close(fig)


def plot_regional_breakdown(df, out_dir):
    """Pie chart: share of latest-year production by region."""
    latest = df[df["year"] == df["year"].max()]
    by_region = latest.groupby("region")["production_million_bags"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        by_region.values, labels=by_region.index, autopct="%1.1f%%",
        startangle=140, colors=sns.color_palette("Set2", len(by_region)),
        wedgeprops={"edgecolor": "white", "linewidth": 1},
    )
    ax.set_title(f"Global Production Share by Region ({int(latest['year'].iloc[0])})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_regional_breakdown.png"), dpi=150)
    plt.close(fig)


def forecast_vietnam(df, out_dir, horizon_years=5):
    """Simple linear regression forecast of Vietnam's production."""
    vn = df[df["country"] == "Vietnam"].sort_values("year")
    X = vn[["year"]].values
    y = vn["production_million_bags"].values

    model = LinearRegression().fit(X, y)
    future_years = np.arange(vn["year"].max() + 1, vn["year"].max() + 1 + horizon_years).reshape(-1, 1)
    future_preds = model.predict(future_years)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(vn["year"], y, label="Historical", color="steelblue", s=40)
    ax.plot(vn["year"], model.predict(X), color="steelblue", linestyle="--", alpha=0.6, label="Fitted trend")
    ax.plot(future_years.flatten(), future_preds, color="darkorange", marker="o",
            label=f"Forecast ({horizon_years}y)", linewidth=2)
    ax.set_title("Vietnam Coffee Production: Linear Trend Forecast", fontsize=13, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Production (million bags)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "06_vietnam_forecast.png"), dpi=150)
    plt.close(fig)

    return dict(zip(future_years.flatten().tolist(), np.round(future_preds, 2).tolist()))


def build_summary(df, corr, forecast):
    latest_year = df["year"].max()
    latest = df[df["year"] == latest_year]
    top_producer = latest.loc[latest["production_million_bags"].idxmax()]
    top_consumer = latest.loc[latest["domestic_consumption_kg_per_capita"].idxmax()]
    fastest_growth = (
        df[df["year"].isin([df["year"].min(), latest_year])]
        .pivot(index="country", columns="year", values="production_million_bags")
    )
    fastest_growth["pct_change"] = (
        (fastest_growth[latest_year] - fastest_growth[df["year"].min()])
        / fastest_growth[df["year"].min()] * 100
    )
    fastest = fastest_growth["pct_change"].idxmax()
    fastest_val = fastest_growth["pct_change"].max()

    lines = [
        "=" * 60,
        "GLOBAL COFFEE TRADE & CONSUMPTION ANALYSIS -- SUMMARY",
        "=" * 60,
        f"Dataset: {df['country'].nunique()} countries, {df['year'].min()}-{df['year'].max()}",
        "",
        f"Top producer ({latest_year}): {top_producer['country']} "
        f"({top_producer['production_million_bags']:.1f}M bags)",
        f"Highest domestic per-capita consumption ({latest_year}): "
        f"{top_consumer['country']} ({top_consumer['domestic_consumption_kg_per_capita']:.2f} kg/person)",
        f"Fastest-growing producer since {df['year'].min()}: {fastest} "
        f"({fastest_val:+.1f}% change in output)",
        f"Brazil price-vs-export-revenue correlation: r = {corr:.2f} "
        f"({'strong positive' if corr > 0.6 else 'moderate' if corr > 0.3 else 'weak'} relationship)",
        "",
        "Vietnam production forecast (linear trend):",
    ]
    for year, val in forecast.items():
        lines.append(f"  {year}: {val:.1f}M bags")
    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()

    print(f"Loaded {len(df)} rows covering {df['country'].nunique()} countries.\n")

    plot_production_trends(df, OUTPUT_DIR)
    plot_market_share(df, OUTPUT_DIR)
    corr = plot_price_vs_exports(df, OUTPUT_DIR)
    plot_consumption_vs_production(df, OUTPUT_DIR)
    plot_regional_breakdown(df, OUTPUT_DIR)
    forecast = forecast_vietnam(df, OUTPUT_DIR)

    summary = build_summary(df, corr, forecast)
    print(summary)

    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
        f.write(summary)

    print(f"\nSaved 6 charts + summary.txt to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
