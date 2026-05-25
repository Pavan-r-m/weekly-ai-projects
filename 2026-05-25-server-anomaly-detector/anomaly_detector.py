"""
Server Metrics Anomaly Detector
================================
Uses Isolation Forest (an unsupervised ML algorithm) to detect anomalies
in synthetic server monitoring data (CPU, memory, latency, error rate).

Isolation Forest works by randomly partitioning data — anomalies are
isolated faster (shorter path length) than normal points, making it
highly effective for high-dimensional, unlabeled telemetry data.

Author: Claude AI | Date: 2026-05-25
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. GENERATE SYNTHETIC SERVER METRICS
# ─────────────────────────────────────────────

def generate_server_metrics(n_points: int = 1440, seed: int = 42) -> pd.DataFrame:
    """
    Simulate 24 hours of server metrics sampled every minute (1440 points).
    Injects realistic anomalies: CPU spikes, memory leaks, latency bursts,
    and error rate surges.
    """
    np.random.seed(seed)
    timestamps = pd.date_range(start="2026-05-25 00:00", periods=n_points, freq="1min")

    # ----- Normal baseline patterns -----
    # CPU: diurnal curve (higher during business hours) + Gaussian noise
    hour = np.array([t.hour + t.minute / 60 for t in timestamps])
    cpu_base = 30 + 20 * np.sin(np.pi * (hour - 6) / 12) * (hour >= 6) * (hour <= 22)
    cpu = cpu_base + np.random.normal(0, 4, n_points)
    cpu = np.clip(cpu, 5, 95)

    # Memory: slowly cycling (simulates GC + usage growth)
    memory_base = 45 + (0.01 * np.arange(n_points)) % 30
    memory = memory_base + np.random.normal(0, 3, n_points)
    memory = np.clip(memory, 20, 95)

    # Latency: log-normal (most requests fast, tail of slow ones)
    latency = np.random.lognormal(mean=3.5, sigma=0.4, size=n_points)
    latency = np.clip(latency, 10, 500)

    # Error rate: mostly near 0.5%, occasional small blips
    error_rate = np.abs(np.random.normal(0.5, 0.5, n_points))
    error_rate = np.clip(error_rate, 0, 5)

    df = pd.DataFrame({
        "timestamp":       timestamps,
        "cpu_pct":         cpu,
        "memory_pct":      memory,
        "latency_ms":      latency,
        "error_rate_pct":  error_rate,
    })

    # ----- Inject labelled anomalies -----
    anomaly_mask = np.zeros(n_points, dtype=bool)

    # Type 1: CPU spike — simulates a runaway process (3 events)
    for center in [200, 650, 1100]:
        w = np.random.randint(5, 20)
        idx = list(range(max(0, center - w // 2), min(n_points, center + w // 2)))
        df.loc[idx, "cpu_pct"] = np.random.uniform(88, 99, len(idx))
        anomaly_mask[idx] = True

    # Type 2: Memory leak burst — rapid climb (2 events)
    for center in [400, 980]:
        w = np.random.randint(10, 30)
        idx = list(range(max(0, center - w // 2), min(n_points, center + w // 2)))
        df.loc[idx, "memory_pct"] = np.random.uniform(88, 98, len(idx))
        anomaly_mask[idx] = True

    # Type 3: Latency spike — slow DB query or network blip (4 events)
    for center in [150, 500, 800, 1300]:
        w = np.random.randint(3, 10)
        idx = list(range(max(0, center - w // 2), min(n_points, center + w // 2)))
        df.loc[idx, "latency_ms"] = np.random.uniform(800, 2000, len(idx))
        anomaly_mask[idx] = True

    # Type 4: Error rate surge — downstream service failure (2 events)
    for center in [300, 1200]:
        w = np.random.randint(5, 15)
        idx = list(range(max(0, center - w // 2), min(n_points, center + w // 2)))
        df.loc[idx, "error_rate_pct"] = np.random.uniform(15, 40, len(idx))
        anomaly_mask[idx] = True

    df["is_anomaly_true"] = anomaly_mask.astype(int)
    return df


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling statistics and cyclic time encoding.
    Rolling windows let the model sense how a value deviates from
    its recent history, not just its absolute level.
    """
    df = df.copy()

    for col in ["cpu_pct", "memory_pct", "latency_ms", "error_rate_pct"]:
        # 5-min and 15-min rolling means
        df[f"{col}_roll5"]  = df[col].rolling(window=5,  min_periods=1).mean()
        df[f"{col}_roll15"] = df[col].rolling(window=15, min_periods=1).mean()
        # Rolling z-score: how many stdevs from the local 15-min average?
        roll_std = df[col].rolling(window=15, min_periods=1).std().fillna(1)
        df[f"{col}_zscore"] = (df[col] - df[f"{col}_roll15"]) / (roll_std + 1e-6)

    # Cyclic hour encoding (avoids artificial 23→0 jump)
    hour_frac = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24)

    return df


# ─────────────────────────────────────────────
# 3. TRAIN ISOLATION FOREST
# ─────────────────────────────────────────────

def train_model(df: pd.DataFrame, contamination: float = 0.05):
    """
    Train an Isolation Forest on the engineered feature matrix.

    Key hyper-parameters:
      n_estimators=200  — more trees → more stable anomaly scores
      contamination=0.05 — we expect ~5% of points to be anomalies
      max_samples='auto' — uses min(256, n_samples) per tree for speed
    """
    feature_cols = [c for c in df.columns
                    if c not in ("timestamp", "is_anomaly_true")]

    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # score_samples() returns negative average path lengths
    raw_scores = model.score_samples(X_scaled)
    # Normalise to [0, 1] where 1.0 = most anomalous
    scores_norm = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    df = df.copy()
    df["anomaly_score"]   = 1.0 - scores_norm
    df["is_anomaly_pred"] = (model.predict(X_scaled) == -1).astype(int)

    return df, model, scaler, feature_cols


# ─────────────────────────────────────────────
# 4. EVALUATION
# ─────────────────────────────────────────────

def evaluate(df: pd.DataFrame):
    """Compare predicted vs ground-truth anomaly labels."""
    print("\n" + "=" * 52)
    print("   ANOMALY DETECTION — EVALUATION REPORT")
    print("=" * 52)

    y_true = df["is_anomaly_true"]
    y_pred = df["is_anomaly_pred"]
    total      = len(df)
    true_anom  = int(y_true.sum())
    pred_anom  = int(y_pred.sum())
    true_pos   = int(((y_true == 1) & (y_pred == 1)).sum())

    print(f"\n  Total data points : {total:,}")
    print(f"  True anomalies    : {true_anom:,}  ({100*true_anom/total:.1f}%)")
    print(f"  Predicted anomalies: {pred_anom:,}  ({100*pred_anom/total:.1f}%)")
    print(f"  True positives    : {true_pos:,}")
    print()
    print(classification_report(y_true, y_pred,
                                target_names=["Normal", "Anomaly"]))


# ─────────────────────────────────────────────
# 5. VISUALISATION
# ─────────────────────────────────────────────

def plot_results(df: pd.DataFrame, save_path: str = "anomaly_detection_results.png"):
    """
    5-panel chart:
      Panels 1–4: Raw metric traces (CPU, Memory, Latency, Error Rate)
        - Red background shading = ground-truth anomaly windows
        - Orange dots = model-predicted anomalies
      Panel 5: Normalised anomaly score time series
    """
    fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(
        "Server Metrics Anomaly Detection — Isolation Forest\n"
        "24-hour Synthetic Server Data  |  2026-05-25",
        fontsize=14, fontweight="bold", y=0.99,
    )

    ts        = df["timestamp"]
    true_mask = df["is_anomaly_true"].astype(bool)
    pred_mask = df["is_anomaly_pred"].astype(bool)

    panels = [
        ("cpu_pct",        "CPU Usage (%)",    "#2196F3",  0,   100),
        ("memory_pct",     "Memory (%)",       "#9C27B0",  0,   100),
        ("latency_ms",     "Latency (ms)",     "#FF9800",  0,  2100),
        ("error_rate_pct", "Error Rate (%)",   "#F44336",  0,    45),
    ]

    for ax, (col, label, color, ymin, ymax) in zip(axes[:4], panels):
        ax.plot(ts, df[col], color=color, linewidth=0.7, alpha=0.85)

        # Shade true anomaly windows
        in_window, start = False, None
        for i, flag in enumerate(true_mask):
            if flag and not in_window:
                start, in_window = ts.iloc[i], True
            elif not flag and in_window:
                ax.axvspan(start, ts.iloc[i], color="red", alpha=0.12)
                in_window = False
        if in_window:
            ax.axvspan(start, ts.iloc[-1], color="red", alpha=0.12)

        # Mark predicted anomalies
        ax.scatter(ts[pred_mask], df.loc[pred_mask, col],
                   color="darkorange", s=14, zorder=5, alpha=0.85,
                   label="Predicted anomaly")

        ax.set_ylabel(label, fontsize=9)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    # Panel 5: anomaly score
    ax5 = axes[4]
    ax5.fill_between(ts, df["anomaly_score"], color="#E91E63", alpha=0.5)
    ax5.axhline(0.5, color="black", linestyle="--", linewidth=0.8,
                label="Decision threshold 0.5")
    ax5.set_ylabel("Anomaly\nScore", fontsize=9)
    ax5.set_ylim(0, 1)
    ax5.set_xlabel("Time (UTC)", fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc="upper right", fontsize=8)

    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax5.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.xticks(rotation=30)

    # Legend for red shading (add proxy patch)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="red",        alpha=0.3, label="True anomaly window"),
        Patch(facecolor="darkorange", alpha=0.8, label="Predicted anomaly"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[✓] Plot saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# 6. TOP ANOMALY SUMMARY
# ─────────────────────────────────────────────

def print_top_anomalies(df: pd.DataFrame, top_n: int = 10):
    """Print the highest-scoring predicted anomaly points."""
    top = (
        df[df["is_anomaly_pred"] == 1]
        .nlargest(top_n, "anomaly_score")[
            ["timestamp", "cpu_pct", "memory_pct",
             "latency_ms", "error_rate_pct", "anomaly_score"]
        ]
    )
    top.columns = ["Timestamp", "CPU%", "Mem%", "Latency(ms)", "ErrRate%", "Score"]
    top["CPU%"]        = top["CPU%"].round(1)
    top["Mem%"]        = top["Mem%"].round(1)
    top["Latency(ms)"] = top["Latency(ms)"].round(1)
    top["ErrRate%"]    = top["ErrRate%"].round(2)
    top["Score"]       = top["Score"].round(4)

    print(f"\nTop {top_n} Most Anomalous Points:")
    print("─" * 82)
    print(top.to_string(index=False))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("┌─────────────────────────────────────────────────┐")
    print("│   Server Metrics Anomaly Detector               │")
    print("│   Algorithm : Isolation Forest (scikit-learn)   │")
    print("│   Data      : Synthetic 24-hour server telemetry│")
    print("└─────────────────────────────────────────────────┘\n")

    print("[1/5] Generating 24 h of synthetic server metrics …")
    df = generate_server_metrics(n_points=1440)
    print(f"      {len(df):,} data points  |  "
          f"{int(df['is_anomaly_true'].sum())} anomaly points injected")

    print("[2/5] Engineering features (rolling stats + cyclic time) …")
    df = engineer_features(df)
    print(f"      Feature set size: {df.shape[1] - 2} columns")

    print("[3/5] Training Isolation Forest (200 estimators, contamination=5%) …")
    df, model, scaler, feature_cols = train_model(df, contamination=0.05)
    print(f"      Trained on {len(feature_cols)} features")

    print("[4/5] Evaluating …")
    evaluate(df)

    print("[5/5] Plotting results …")
    plot_results(df, "anomaly_detection_results.png")
    print_top_anomalies(df)

    out_cols = [
        "timestamp", "cpu_pct", "memory_pct", "latency_ms",
        "error_rate_pct", "anomaly_score", "is_anomaly_pred", "is_anomaly_true",
    ]
    df[out_cols].to_csv("anomaly_results.csv", index=False)
    print("\n[✓] Results saved → anomaly_results.csv")
    print("\nAll done! ✓")


if __name__ == "__main__":
    main()
