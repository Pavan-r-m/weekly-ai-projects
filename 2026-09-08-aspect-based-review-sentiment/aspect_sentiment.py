"""
Aspect-Based Sentiment Analysis (ABSA) for Product Reviews
=============================================================
Goes beyond a single "positive/negative" label for a whole review and
figures out how customers feel about *specific aspects* of a product
(battery, camera, price, performance, build quality, customer service...).

A single review often praises one aspect and criticizes another in the
same breath ("battery is amazing but the camera is disappointing") -- a
plain sentiment classifier collapses that into one confusing score. This
script splits each review into sentences, detects which aspect(s) each
sentence talks about using a keyword lexicon, and scores the sentiment of
that sentence with VADER (a rule-based sentiment analyzer tuned for
short, informal text). The per-sentence scores are then aggregated per
aspect and per product.

No API key or internet access is required at runtime -- everything runs
locally with a small, pinned dependency (vaderSentiment).

Usage:
    python aspect_sentiment.py
    python aspect_sentiment.py --input data/product_reviews_sample.csv --product PhoneX
"""

import argparse
import csv
import os
import re
from collections import defaultdict

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend for saving PNGs without a display
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ---------------------------------------------------------------------------
# 1. Aspect lexicon: maps a canonical aspect name to a list of keywords /
#    phrases that signal that aspect is being discussed. Matching is done
#    on lowercased text with simple substring checks, so short phrases like
#    "battery life" also match the parent keyword "battery".
# ---------------------------------------------------------------------------
ASPECT_LEXICON = {
    "battery": ["battery", "charge", "charging", "standby"],
    "camera": ["camera", "photo", "photos", "zoom", "night mode", "webcam"],
    "screen": ["screen", "display", "resolution", "brightness", "color", "flicker"],
    "price": ["price", "priced", "overpriced", "cost", "expensive", "cheap price", "worth"],
    "performance": ["performance", "speed", "fast", "lag", "lags", "stutter", "multitasking", "benchmarks", "gaming"],
    "build_quality": ["build quality", "build", "chassis", "sturdy", "flimsy", "plasticky", "waterproof", "creaks"],
    "customer_service": ["customer service", "support", "service", "replaced", "ticket", "response"],
    "sound": ["sound", "bass", "audio", "call quality", "calls"],
}


def load_reviews(csv_path: str) -> pd.DataFrame:
    """Load the reviews CSV into a DataFrame, failing loudly if it's missing."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find '{csv_path}'. Run this script from the project "
            "folder, or pass --input with the correct path."
        )
    return pd.read_csv(csv_path)


def split_sentences(text: str):
    """
    Very lightweight sentence splitter: breaks on '.', '!' or '?' followed
    by whitespace. Good enough for review-style text and avoids pulling in
    a heavyweight NLP model just to tokenize sentences.
    """
    text = text.strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def find_aspects(sentence: str):
    """Return the list of canonical aspect names mentioned in a sentence."""
    lowered = sentence.lower()
    found = []
    for aspect, keywords in ASPECT_LEXICON.items():
        if any(keyword in lowered for keyword in keywords):
            found.append(aspect)
    return found


def label_from_score(compound: float) -> str:
    """Map a VADER compound score (-1..1) to a human-readable label."""
    if compound >= 0.25:
        return "positive"
    if compound <= -0.25:
        return "negative"
    return "neutral"


def analyze_reviews(df: pd.DataFrame, analyzer: SentimentIntensityAnalyzer):
    """
    Walk every review, split into sentences, tag aspects per sentence, and
    score sentiment per sentence with VADER.

    Returns a list of row-dicts (one per aspect mention) ready to become a
    DataFrame/CSV, e.g.:
        {review_id, product, aspect, sentence, compound, label}
    """
    rows = []
    for _, review in df.iterrows():
        sentences = split_sentences(str(review["review_text"]))
        for sentence in sentences:
            aspects = find_aspects(sentence)
            if not aspects:
                continue  # sentence doesn't mention a tracked aspect, skip it
            scores = analyzer.polarity_scores(sentence)
            compound = scores["compound"]
            label = label_from_score(compound)
            for aspect in aspects:
                rows.append(
                    {
                        "review_id": review["review_id"],
                        "product": review["product"],
                        "aspect": aspect,
                        "sentence": sentence,
                        "compound": round(compound, 4),
                        "label": label,
                    }
                )
    return rows


def aggregate_by_aspect(rows, product_filter=None):
    """
    Compute average sentiment and mention count per aspect, optionally
    restricted to a single product. Returns a dict:
        {aspect: {"avg_compound": float, "count": int,
                   "positive": int, "neutral": int, "negative": int}}
    """
    buckets = defaultdict(lambda: {"scores": [], "positive": 0, "neutral": 0, "negative": 0})
    for row in rows:
        if product_filter and row["product"] != product_filter:
            continue
        bucket = buckets[row["aspect"]]
        bucket["scores"].append(row["compound"])
        bucket[row["label"]] += 1

    summary = {}
    for aspect, bucket in buckets.items():
        scores = bucket["scores"]
        summary[aspect] = {
            "avg_compound": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "count": len(scores),
            "positive": bucket["positive"],
            "neutral": bucket["neutral"],
            "negative": bucket["negative"],
        }
    return summary


def print_summary(summary: dict, title: str):
    print(f"\n=== Aspect Sentiment Summary: {title} ===")
    print(f"{'Aspect':<18}{'Avg Score':>10}{'Mentions':>10}{'Pos':>6}{'Neu':>6}{'Neg':>6}")
    for aspect, s in sorted(summary.items(), key=lambda kv: kv[1]["avg_compound"], reverse=True):
        print(
            f"{aspect:<18}{s['avg_compound']:>10.3f}{s['count']:>10}"
            f"{s['positive']:>6}{s['neutral']:>6}{s['negative']:>6}"
        )


def save_chart(summary: dict, out_path: str, title: str):
    """Bar chart of average sentiment compound score per aspect, colored by polarity."""
    aspects = sorted(summary.keys(), key=lambda a: summary[a]["avg_compound"])
    scores = [summary[a]["avg_compound"] for a in aspects]
    colors = ["#d9534f" if s < 0 else ("#f0ad4e" if abs(s) < 0.05 else "#5cb85c") for s in scores]

    plt.figure(figsize=(9, 5.5))
    bars = plt.barh(aspects, scores, color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Average VADER Compound Sentiment (-1 = very negative, +1 = very positive)")
    plt.title(f"Aspect-Based Sentiment — {title}")
    plt.xlim(-1, 1)
    for bar, score in zip(bars, scores):
        plt.text(
            score + (0.02 if score >= 0 else -0.02),
            bar.get_y() + bar.get_height() / 2,
            f"{score:+.2f}",
            va="center",
            ha="left" if score >= 0 else "right",
            fontsize=9,
        )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved chart -> {out_path}")


def save_detail_csv(rows, out_path: str):
    """Save the full per-sentence, per-aspect breakdown for auditing/inspection."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["review_id", "product", "aspect", "sentence", "compound", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved detail report -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Aspect-based sentiment analysis for product reviews.")
    parser.add_argument("--input", default="data/product_reviews_sample.csv", help="Path to reviews CSV.")
    parser.add_argument("--product", default=None, help="Restrict analysis to one product (optional).")
    parser.add_argument("--outdir", default="output", help="Directory to write the report/chart into.")
    args = parser.parse_args()

    df = load_reviews(args.input)
    if args.product:
        available = sorted(df["product"].unique())
        if args.product not in available:
            raise SystemExit(f"Unknown product '{args.product}'. Available: {available}")

    analyzer = SentimentIntensityAnalyzer()
    rows = analyze_reviews(df, analyzer)

    if not rows:
        raise SystemExit("No aspect mentions found -- check the input data or ASPECT_LEXICON.")

    save_detail_csv(rows, os.path.join(args.outdir, "aspect_sentiment_detail.csv"))

    # Overall summary across all products
    overall_summary = aggregate_by_aspect(rows)
    print_summary(overall_summary, "All Products")
    save_chart(overall_summary, os.path.join(args.outdir, "aspect_sentiment_overall.png"), "All Products")

    # Per-product summaries (or just the one requested via --product)
    products = [args.product] if args.product else sorted(df["product"].unique())
    for product in products:
        product_summary = aggregate_by_aspect(rows, product_filter=product)
        print_summary(product_summary, product)
        safe_name = re.sub(r"\W+", "_", product.lower())
        save_chart(product_summary, os.path.join(args.outdir, f"aspect_sentiment_{safe_name}.png"), product)

    print("\nDone. Open the PNG charts in the output/ folder to see aspect sentiment at a glance.")


if __name__ == "__main__":
    main()
