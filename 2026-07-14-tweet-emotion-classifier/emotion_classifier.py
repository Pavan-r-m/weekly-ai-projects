"""
emotion_classifier.py
----------------------
A multi-class emotion classifier for short, tweet-style text.

Given a piece of informal social-media text, this script predicts the
dominant emotion behind it: joy, sadness, anger, fear, surprise, or love.

Pipeline:
    1. Load labeled tweets from data/tweets_emotions.csv
       (generate it first with `python generate_dataset.py`)
    2. Clean the text (lowercase, strip URLs/mentions, normalize punctuation)
    3. Vectorize with TF-IDF (unigrams + bigrams)
    4. Train a Logistic Regression classifier (one-vs-rest, multi-class)
    5. Evaluate with a held-out test split (accuracy, per-class precision/
       recall/F1, confusion matrix)
    6. Save a confusion matrix heatmap to output/
    7. Run the trained model on a handful of brand-new example sentences

Run:
    python emotion_classifier.py
"""

import re
import sys
import os

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe backend for saving PNGs
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

DATA_PATH = "data/tweets_emotions.csv"
OUTPUT_DIR = "output"
MODEL_LABELS = ["anger", "fear", "joy", "love", "sadness", "surprise"]


def clean_text(text: str) -> str:
    """Lowercase and strip URLs, @mentions, and non-letter noise, but keep
    the emotional signal words and simple punctuation like '!' and '?'."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"@\w+", " ", text)                        # @mentions
    text = re.sub(r"#(\w+)", r"\1", text)                     # keep hashtag word, drop '#'
    text = re.sub(r"[^a-z0-9!?'\s]", " ", text)               # strip emojis/punct noise
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(
            f"Dataset not found at '{path}'. Run `python generate_dataset.py` first."
        )
    df = pd.read_csv(path)
    df["clean_text"] = df["text"].apply(clean_text)
    return df


def build_pipeline() -> Pipeline:
    """TF-IDF + Logistic Regression pipeline. Bigrams help catch phrases
    like 'not happy' vs. 'happy', which a pure unigram model would conflate."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=5.0,
            random_state=42,
        )),
    ])


def plot_confusion_matrix(y_true, y_pred, labels, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted emotion")
    ax.set_ylabel("True emotion")
    ax.set_title("Emotion Classifier — Confusion Matrix (test set)")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading dataset...")
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} labeled tweets across {df['emotion'].nunique()} emotions.\n")

    X = df["clean_text"]
    y = df["emotion"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print("Training TF-IDF + Logistic Regression classifier...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nTest accuracy: {acc:.3f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred, labels=MODEL_LABELS, zero_division=0))

    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plot_confusion_matrix(y_test, y_pred, MODEL_LABELS, cm_path)
    print(f"Saved confusion matrix to {cm_path}")

    # -----------------------------------------------------------------
    # Show the model's top predictive words per emotion (interpretability)
    # -----------------------------------------------------------------
    print("\nTop TF-IDF features per emotion:")
    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    feature_names = np.array(vectorizer.get_feature_names_out())
    for idx, label in enumerate(clf.classes_):
        top_idx = np.argsort(clf.coef_[idx])[-8:][::-1]
        top_words = feature_names[top_idx]
        print(f"  {label:10s}: {', '.join(top_words)}")

    # -----------------------------------------------------------------
    # Demo: predict emotion on brand-new, hand-written example sentences
    # -----------------------------------------------------------------
    demo_sentences = [
        "I can't believe my team actually won the game, I'm ecstatic!",
        "My package never showed up and support just hung up on me again.",
        "There's something scratching outside my window and I'm frozen in fear.",
        "I didn't expect to see my old roommate at the wedding, what a shock.",
        "My dad drove six hours just to help me move, I feel so cared for.",
        "Ever since the layoffs were announced I've felt completely empty.",
    ]

    print("\nDemo predictions on new sentences:")
    demo_clean = [clean_text(s) for s in demo_sentences]
    demo_preds = pipeline.predict(demo_clean)
    demo_probs = pipeline.predict_proba(demo_clean)

    for sentence, pred, probs in zip(demo_sentences, demo_preds, demo_probs):
        confidence = probs[list(clf.classes_).index(pred)]
        print(f"  [{pred:9s} {confidence:.2f}]  {sentence}")

    print("\nDone. See output/confusion_matrix.png for the visual breakdown.")


if __name__ == "__main__":
    main()
