#!/usr/bin/env python3
"""
author_identifier.py
=====================
A stylometric author-identification tool.

Given a labelled corpus of texts (one or more per author), this script:
  1. Splits each text into overlapping word-windows to create many
     training samples per author.
  2. Extracts a "stylistic fingerprint" for each window: relative
     frequencies of common English function words plus a handful of
     surface statistics (sentence length, word length, vocabulary
     richness, punctuation usage).
  3. Trains a Random Forest classifier on those fingerprints.
  4. Evaluates it with cross-validation (since function words are
     topic-independent, this works even though every author here wrote
     about a totally different subject).
  5. Uses the trained model to guess the author of a held-out "mystery"
     passage, or of any text you pass in yourself.

Why function words? Content words (nouns, verbs describing the plot)
give away the *topic* of a passage, not the *author*. Function words
("the", "of", "and", "which", "upon" ...) are used almost unconsciously
and vary surprisingly consistently from writer to writer -- this is the
same core idea used in real forensic linguistics and disputed-authorship
cases (e.g. the Federalist Papers studies).

No API key or internet connection is required -- the demo corpus is
bundled in corpus_data.py.

Usage
-----
    python author_identifier.py                     # run full demo
    python author_identifier.py --plot               # also save a feature-importance chart
    python author_identifier.py --text "some prose"  # classify your own text
    python author_identifier.py --file mytext.txt    # classify a text file
"""

import argparse
import re
import sys
from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from corpus_data import CORPUS, MYSTERY_TEXT

# ---------------------------------------------------------------------------
# 1. Feature vocabulary
# ---------------------------------------------------------------------------
# A curated list of common English function words (articles, prepositions,
# conjunctions, pronouns, auxiliary verbs). These are topic-agnostic and
# form the backbone of the stylistic fingerprint.
FUNCTION_WORDS = [
    "the", "of", "and", "to", "a", "in", "that", "is", "was", "he", "for",
    "it", "with", "as", "his", "on", "be", "at", "by", "i", "this", "had",
    "not", "are", "but", "from", "or", "have", "an", "they", "which",
    "one", "you", "were", "her", "all", "she", "there", "would", "their",
    "we", "him", "been", "has", "when", "who", "will", "no", "if", "out",
    "so", "what", "up", "its", "about", "into", "than", "them", "can",
    "only", "some", "could", "did", "do", "any", "my", "now", "such",
    "like", "our", "me", "very", "than",
]
FUNCTION_WORDS = sorted(set(FUNCTION_WORDS))  # dedupe, stable order

# Names of the extra (non function-word) stylistic features, appended
# after the function-word frequency block in every feature vector.
EXTRA_FEATURE_NAMES = [
    "avg_sentence_len",
    "avg_word_len",
    "type_token_ratio",
    "comma_rate",
    "semicolon_rate",
    "exclaim_question_rate",
    "long_word_rate",
]
FEATURE_NAMES = FUNCTION_WORDS + EXTRA_FEATURE_NAMES

WORD_RE = re.compile(r"[A-Za-z']+")
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


# ---------------------------------------------------------------------------
# 2. Text processing helpers
# ---------------------------------------------------------------------------
def tokenize(text: str):
    """Lower-case word tokens (keeps apostrophes, drops other punctuation)."""
    return [w.lower() for w in WORD_RE.findall(text)]


def count_sentences(text: str) -> int:
    parts = [p for p in SENTENCE_SPLIT_RE.split(text) if p.strip()]
    return max(1, len(parts))


def sliding_window_chunks(words, window=40, stride=20):
    """Yield overlapping chunks of `window` words, moving `stride` words
    at a time. This turns one short excerpt into several training
    samples, which is essential since our demo corpus is small."""
    chunks = []
    if len(words) <= window:
        chunks.append(words)
        return chunks
    for start in range(0, len(words) - window + 1, stride):
        chunks.append(words[start:start + window])
    return chunks


def extract_features(text: str) -> np.ndarray:
    """Convert raw text into the numeric stylistic fingerprint vector."""
    words = tokenize(text)
    n_words = max(1, len(words))
    counts = Counter(words)

    # Function-word relative frequencies (per 100 words)
    fw_freqs = [counts.get(w, 0) / n_words * 100 for w in FUNCTION_WORDS]

    n_sentences = count_sentences(text)
    avg_sentence_len = n_words / n_sentences
    avg_word_len = sum(len(w) for w in words) / n_words
    type_token_ratio = len(set(words)) / n_words
    comma_rate = text.count(",") / n_words * 100
    semicolon_rate = text.count(";") / n_words * 100
    exclaim_question_rate = (text.count("!") + text.count("?")) / n_words * 100
    long_word_rate = sum(1 for w in words if len(w) >= 7) / n_words * 100

    extra = [
        avg_sentence_len, avg_word_len, type_token_ratio,
        comma_rate, semicolon_rate, exclaim_question_rate, long_word_rate,
    ]
    return np.array(fw_freqs + extra, dtype=float)


# ---------------------------------------------------------------------------
# 3. Dataset construction
# ---------------------------------------------------------------------------
def build_dataset(corpus, window=40, stride=20):
    X, y = [], []
    for author, excerpts in corpus.items():
        for excerpt in excerpts:
            clean = re.sub(r"\s+", " ", excerpt).strip()
            words = tokenize(clean)
            for chunk_words in sliding_window_chunks(words, window, stride):
                chunk_text = " ".join(chunk_words)
                X.append(extract_features(chunk_text))
                y.append(author)
    return np.vstack(X), np.array(y)


# ---------------------------------------------------------------------------
# 4. Training / evaluation
# ---------------------------------------------------------------------------
def evaluate_with_cross_validation(X, y, n_splits=4, seed=42):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    n_splits = min(n_splits, min(Counter(y_enc).values()))
    n_splits = max(2, n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    clf = RandomForestClassifier(n_estimators=300, random_state=seed)
    y_pred = cross_val_predict(clf, X, y_enc, cv=skf)

    acc = accuracy_score(y_enc, y_pred)
    print(f"\nCross-validated accuracy ({n_splits}-fold): {acc:.2%}\n")
    print("Classification report:")
    print(classification_report(y_enc, y_pred, target_names=le.classes_, zero_division=0))

    print("Confusion matrix (rows = true author, cols = predicted):")
    cm = confusion_matrix(y_enc, y_pred)
    header = "".join(f"{name[:10]:>12}" for name in le.classes_)
    print(" " * 18 + header)
    for name, row in zip(le.classes_, cm):
        print(f"{name[:16]:>16}  " + "".join(f"{v:>12}" for v in row))
    return le


def train_final_model(X, y, seed=42):
    clf = RandomForestClassifier(n_estimators=300, random_state=seed)
    clf.fit(X, y)
    return clf


def predict_author(clf, text, top_k=3):
    features = extract_features(text).reshape(1, -1)
    probs = clf.predict_proba(features)[0]
    classes = clf.classes_
    ranked = sorted(zip(classes, probs), key=lambda p: -p[1])
    return ranked[:top_k]


def top_distinguishing_features(clf, n=10):
    importances = clf.feature_importances_
    ranked = sorted(zip(FEATURE_NAMES, importances), key=lambda p: -p[1])
    return ranked[:n]


def maybe_plot_importance(clf, path="feature_importance.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = top_distinguishing_features(clf, n=15)
    names = [t[0] for t in top][::-1]
    values = [t[1] for t in top][::-1]

    plt.figure(figsize=(8, 6))
    plt.barh(names, values, color="#4C72B0")
    plt.xlabel("Random Forest feature importance")
    plt.title("Top stylistic features for author identification")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"\nSaved feature-importance chart to {path}")


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Stylometric author identifier")
    parser.add_argument("--text", type=str, default=None,
                         help="Classify this raw text instead of running the demo mystery passage.")
    parser.add_argument("--file", type=str, default=None,
                         help="Classify the contents of this text file.")
    parser.add_argument("--window", type=int, default=40, help="Sliding window size in words.")
    parser.add_argument("--stride", type=int, default=20, help="Sliding window stride in words.")
    parser.add_argument("--plot", action="store_true", help="Save a feature-importance bar chart PNG.")
    args = parser.parse_args()

    print("=" * 70)
    print("STYLOMETRIC AUTHOR IDENTIFIER")
    print("=" * 70)
    print(f"Authors in training corpus: {', '.join(CORPUS.keys())}")

    X, y = build_dataset(CORPUS, window=args.window, stride=args.stride)
    print(f"Built {len(X)} training samples ({X.shape[1]} features each) "
          f"from {len(CORPUS)} authors.")

    evaluate_with_cross_validation(X, y)

    clf = train_final_model(X, y)

    print("\nTop distinguishing stylistic features overall:")
    for name, importance in top_distinguishing_features(clf, n=10):
        print(f"  {name:<22} {importance:.4f}")

    if args.plot:
        maybe_plot_importance(clf)

    # Decide what to classify: user text/file, or the bundled mystery excerpt.
    if args.text:
        query_text, query_label = args.text, "your --text input"
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            query_text = f.read()
        query_label = f"contents of {args.file}"
    else:
        query_text, query_label = MYSTERY_TEXT, "bundled mystery excerpt (held-out Melville text)"

    print("\n" + "-" * 70)
    print(f"Classifying: {query_label}")
    print("-" * 70)
    print(query_text.strip()[:300] + ("..." if len(query_text) > 300 else ""))

    ranked = predict_author(clf, query_text)
    print("\nPredicted author (ranked by probability):")
    for author, prob in ranked:
        bar = "#" * int(prob * 40)
        print(f"  {author:<20} {prob:>6.1%}  {bar}")


if __name__ == "__main__":
    sys.exit(main())
