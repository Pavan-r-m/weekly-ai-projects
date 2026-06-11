"""
World Happiness Report Deep Dive (2023)
========================================
A comprehensive data analysis and visualization of global happiness scores.
Explores what factors predict happiness, how regions compare, and which
variables correlate most strongly with wellbeing.

Data source: World Happiness Report 2023 (bundled sample: happiness_data.csv)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for running without a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import os

# ── Configuration ────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.05)
PALETTE = "husl"
OUTPUT_DIR = "output_charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES = [
    "GDP_per_capita",
    "Social_support",
    "Healthy_life_expectancy",
    "Freedom",
    "Generosity",
    "Corruption_perception",
]

FEATURE_LABELS = {
    "GDP_per_capita":         "GDP per Capita",
    "Social_support":         "Social Support",
    "Healthy_life_expectancy":"Healthy Life Expectancy",
    "Freedom":                "Freedom",
    "Generosity":             "Generosity",
    "Corruption_perception":  "Corruption Perception",
}

# ── Load Data ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("  World Happiness Report 2023 — Deep Dive Analysis")
print("=" * 60)

df = pd.read_csv("happiness_data.csv")
print(f"\nLoaded {len(df)} countries across {df['Region'].nunique()} regions.\n")

# Drop duplicate rows (New Zealand appears twice in the sample)
df = df.drop_duplicates(subset="Country")
print(f"After dedup: {len(df)} countries.\n")

# ── 1. Summary Statistics ─────────────────────────────────────────────────────
print("── Summary Statistics ──────────────────────────────")
summary = df[["Score"] + FEATURES].describe().round(3)
print(summary.to_string())
print()

# ── 2. Top / Bottom 10 Countries ─────────────────────────────────────────────
top10    = df.nlargest(10,  "Score")[["Country", "Score", "Region"]]
bottom10 = df.nsmallest(10, "Score")[["Country", "Score", "Region"]]
print("── Top 10 Happiest Countries ───────────────────────")
print(top10.to_string(index=False))
print("\n── Bottom 10 Countries ─────────────────────────────")
print(bottom10.to_string(index=False))
print()

# ── 3. Correlation Analysis ───────────────────────────────────────────────────
correlations = df[FEATURES].corrwith(df["Score"]).rename(FEATURE_LABELS).sort_values(ascending=False)
print("── Correlation with Happiness Score ────────────────")
for feat, corr in correlations.items():
    bar = "█" * int(abs(corr) * 20)
    print(f"  {feat:<28} {corr:+.3f}  {bar}")
print()

# ── 4. Regional Averages ──────────────────────────────────────────────────────
region_stats = (
    df.groupby("Region")["Score"]
    .agg(["mean", "min", "max", "count"])
    .rename(columns={"mean": "Avg Score", "min": "Min", "max": "Max", "count": "N"})
    .sort_values("Avg Score", ascending=False)
    .round(3)
)
print("── Regional Happiness Averages ─────────────────────")
print(region_stats.to_string())
print()

# ── 5. OLS regression — feature importance ───────────────────────────────────
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

X = df[FEATURES].values
y = df["Score"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)
y_pred = model.predict(X_scaled)

r2 = r2_score(y, y_pred)
coef_df = pd.DataFrame({
    "Feature": [FEATURE_LABELS[f] for f in FEATURES],
    "Coefficient": model.coef_,
}).sort_values("Coefficient", ascending=False)

print(f"── Linear Regression R² = {r2:.3f} ─────────────────────")
print(coef_df.to_string(index=False))
print()

# ═══════════════════════════════════════════════════════════════════
#  VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════

# ── Figure 1: Top & Bottom 15 Countries (horizontal bar chart) ────
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle("World Happiness Report 2023 — Country Rankings", fontsize=16, fontweight="bold", y=1.01)

top15    = df.nlargest(15,  "Score").sort_values("Score")
bottom15 = df.nsmallest(15, "Score").sort_values("Score", ascending=False)

# Top 15
colors_top = sns.color_palette("YlGn", n_colors=15)
axes[0].barh(top15["Country"], top15["Score"], color=colors_top)
axes[0].set_xlabel("Happiness Score")
axes[0].set_title("Top 15 Happiest Countries", fontweight="bold")
axes[0].set_xlim(0, 9)
for i, (score, country) in enumerate(zip(top15["Score"], top15["Country"])):
    axes[0].text(score + 0.05, i, f"{score:.3f}", va="center", fontsize=9)

# Bottom 15
colors_bot = sns.color_palette("OrRd", n_colors=15)[::-1]
axes[1].barh(bottom15["Country"], bottom15["Score"], color=colors_bot)
axes[1].set_xlabel("Happiness Score")
axes[1].set_title("Bottom 15 Countries", fontweight="bold")
axes[1].set_xlim(0, 9)
for i, (score, country) in enumerate(zip(bottom15["Score"], bottom15["Country"])):
    axes[1].text(score + 0.05, i, f"{score:.3f}", va="center", fontsize=9)

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "01_country_rankings.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")

# ── Figure 2: Correlation Heatmap ────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
corr_matrix = df[["Score"] + FEATURES].rename(columns={**{"Score": "Happiness Score"}, **FEATURE_LABELS}).corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".2f",
    cmap="RdYlGn", center=0, linewidths=0.5,
    square=True, ax=ax, cbar_kws={"shrink": 0.8}
)
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "02_correlation_heatmap.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")

# ── Figure 3: Scatter matrix — top 3 predictors vs Score ─────────
top3_features = correlations.index[:3].tolist()
# Map back to column names
reverse_labels = {v: k for k, v in FEATURE_LABELS.items()}
top3_cols = [reverse_labels[f] for f in top3_features]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Top 3 Happiness Predictors vs Score", fontsize=15, fontweight="bold")

region_colors = {r: c for r, c in zip(df["Region"].unique(), sns.color_palette(PALETTE, df["Region"].nunique()))}

for ax, col, label in zip(axes, top3_cols, top3_features):
    for region, group in df.groupby("Region"):
        ax.scatter(group[col], group["Score"],
                   color=region_colors[region], label=region, alpha=0.75, s=60, edgecolors="white", linewidths=0.4)

    # Regression line
    slope, intercept, r, p, _ = stats.linregress(df[col], df["Score"])
    x_line = np.linspace(df[col].min(), df[col].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=1.5, linestyle="--", label=f"r={r:.2f}")

    # Annotate extreme countries
    for _, row in pd.concat([df.nlargest(2, "Score"), df.nsmallest(2, "Score")]).iterrows():
        ax.annotate(row["Country"], (row[col], row["Score"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7, color="dimgray")

    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel("Happiness Score", fontsize=11)
    ax.set_title(f"{label}\n(r = {r:.3f}, p < {p:.3f})", fontsize=11, fontweight="bold")
    ax.legend(fontsize=6, loc="upper left", ncol=2, framealpha=0.6)

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "03_top_predictors_scatter.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")

# ── Figure 4: Regional violin + box plot ─────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
region_order = region_stats.index.tolist()
palette = sns.color_palette(PALETTE, len(region_order))

sns.violinplot(data=df, x="Region", y="Score", hue="Region", order=region_order,
               palette=palette, inner=None, ax=ax, alpha=0.5, legend=False)
sns.stripplot(data=df, x="Region", y="Score", hue="Region", order=region_order,
              palette=palette, jitter=True, size=5, edgecolor="white", linewidth=0.5, ax=ax, legend=False)

ax.set_xticks(range(len(region_order)))
ax.set_xticklabels(region_order, rotation=30, ha="right", fontsize=9)
ax.set_title("Happiness Score Distribution by World Region", fontsize=14, fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("Happiness Score")
ax.axhline(df["Score"].mean(), color="red", linestyle="--", linewidth=1, alpha=0.6, label=f"Global mean ({df['Score'].mean():.2f})")
ax.legend()
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "04_regional_distribution.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")

# ── Figure 5: Feature importance (OLS coefficients) ──────────────
fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in coef_df["Coefficient"]]
bars = ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title(f"Feature Importance (Standardised OLS Coefficients)\nR² = {r2:.3f}", fontsize=13, fontweight="bold")
ax.set_xlabel("Coefficient (std units → happiness score)")
for bar, val in zip(bars, coef_df["Coefficient"]):
    ax.text(val + (0.01 if val >= 0 else -0.01), bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}", va="center", ha="left" if val >= 0 else "right", fontsize=9)
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "05_feature_importance.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")

# ── Figure 6: Score histogram with KDE ───────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(df["Score"], bins=20, kde=True, color="#3498db", edgecolor="white", ax=ax)
ax.axvline(df["Score"].mean(),   color="red",    linestyle="--", label=f"Mean  {df['Score'].mean():.2f}")
ax.axvline(df["Score"].median(), color="orange", linestyle="--", label=f"Median {df['Score'].median():.2f}")
ax.set_title("Global Distribution of Happiness Scores (2023)", fontsize=14, fontweight="bold")
ax.set_xlabel("Happiness Score")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "06_score_distribution.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")

# ── Final summary ─────────────────────────────────────────────────
print()
print("=" * 60)
print("  Analysis complete!")
print(f"  Charts saved to: ./{OUTPUT_DIR}/")
print(f"    01_country_rankings.png")
print(f"    02_correlation_heatmap.png")
print(f"    03_top_predictors_scatter.png")
print(f"    04_regional_distribution.png")
print(f"    05_feature_importance.png")
print(f"    06_score_distribution.png")
print("=" * 60)

# ── Key findings ──────────────────────────────────────────────────
print("\nKey Findings:")
print(f"  Happiest country  : {df.loc[df['Score'].idxmax(), 'Country']} ({df['Score'].max():.3f})")
print(f"  Least happy       : {df.loc[df['Score'].idxmin(), 'Country']} ({df['Score'].min():.3f})")
print(f"  Global mean score : {df['Score'].mean():.3f}")
print(f"  Strongest predictor: {correlations.index[0]} (r = {correlations.iloc[0]:+.3f})")
print(f"  Weakest predictor : {correlations.index[-1]} (r = {correlations.iloc[-1]:+.3f})")
print(f"  Happiest region   : {region_stats.index[0]} (avg {region_stats['Avg Score'].iloc[0]:.3f})")
