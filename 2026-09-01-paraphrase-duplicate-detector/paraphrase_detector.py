"""
Paraphrase & Duplicate Question Detector
==========================================
An NLP tool that decides whether two sentences (typically questions) are
"duplicates" of each other -- i.e. they ask the same thing in different
words -- similar in spirit to the classic Quora Question Pairs problem.

Instead of relying on a big pretrained transformer, this project builds a
lightweight, fully-interpretable pipeline out of classic NLP features:

  1. TF-IDF cosine similarity        -> how similar is the overall wording?
  2. Jaccard similarity of tokens    -> how much vocabulary overlap is there?
  3. Common-word ratio               -> fraction of the shorter sentence's
                                          words that also appear in the longer one
  4. Length difference (normalized)  -> big length gaps rarely mean "same question"
  5. difflib SequenceMatcher ratio   -> character-level similarity (catches
                                          typos / minor rewordings)

These five numeric features are fed into a Logistic Regression classifier
trained on a small hand-built dataset of duplicate / non-duplicate sentence
pairs covering everyday topics (weather, cooking, tech support, travel,
health, etc).

No API key or internet connection is required -- everything runs locally
with scikit-learn.
"""

import re
import string
import difflib
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


# ---------------------------------------------------------------------------
# 1. A small, hand-crafted labeled dataset of sentence pairs.
#    label = 1 -> duplicate / paraphrase, label = 0 -> different question
# ---------------------------------------------------------------------------
DATASET = [
    # --- duplicates (paraphrases) ---
    ("How do I reset my password?", "What steps do I follow to change my password?", 1),
    ("What is the capital of France?", "Which city is the capital of France?", 1),
    ("How can I lose weight fast?", "What is the fastest way to lose weight?", 1),
    ("Why is the sky blue?", "What causes the sky to appear blue?", 1),
    ("How do I bake a chocolate cake?", "What is the recipe for a chocolate cake?", 1),
    ("What time does the store close?", "At what time does the store shut down?", 1),
    ("How do I install Python on Windows?", "What are the steps to install Python on a Windows PC?", 1),
    ("Is it going to rain tomorrow?", "Will it rain tomorrow?", 1),
    ("How much does a flight to Tokyo cost?", "What is the price of a flight to Tokyo?", 1),
    ("Can you recommend a good laptop for programming?", "What laptop would you suggest for coding?", 1),
    ("How do I stop my phone from overheating?", "What can I do to prevent my phone from getting too hot?", 1),
    ("What are the symptoms of the flu?", "What signs indicate someone has the flu?", 1),
    ("How do I learn to play guitar?", "What is the best way to learn guitar?", 1),
    ("Where can I watch the new Marvel movie?", "What streaming service has the latest Marvel movie?", 1),
    ("How do I cancel my subscription?", "What is the process to cancel my subscription?", 1),
    ("What causes traffic jams?", "Why do traffic jams happen?", 1),
    ("How do I improve my sleep quality?", "What can I do to sleep better?", 1),
    ("What is the best way to save money?", "How can I save money effectively?", 1),
    ("How do airplanes stay in the air?", "What keeps airplanes flying?", 1),
    ("How do I remove a stain from a shirt?", "What is the best method to get a stain out of a shirt?", 1),
    ("What's the difference between a virus and a worm?", "How do computer viruses differ from worms?", 1),
    ("How do I write a cover letter?", "What should I include when writing a cover letter?", 1),
    ("What are good exercises for lower back pain?", "Which exercises help relieve lower back pain?", 1),
    ("How do I connect my printer to WiFi?", "What are the steps to connect a printer over WiFi?", 1),
    ("Why does my car make a rattling noise?", "What could cause a rattling sound in my car?", 1),

    # --- non-duplicates (different questions, sometimes on similar topics) ---
    ("How do I reset my password?", "How do I delete my account?", 0),
    ("What is the capital of France?", "What is the population of France?", 0),
    ("How can I lose weight fast?", "How can I gain muscle fast?", 0),
    ("Why is the sky blue?", "Why is the ocean salty?", 0),
    ("How do I bake a chocolate cake?", "How do I bake bread?", 0),
    ("What time does the store close?", "What time does the store open?", 0),
    ("How do I install Python on Windows?", "How do I install Java on Windows?", 0),
    ("Is it going to rain tomorrow?", "Is it going to be sunny this weekend?", 0),
    ("How much does a flight to Tokyo cost?", "How much does a hotel in Tokyo cost?", 0),
    ("Can you recommend a good laptop for programming?", "Can you recommend a good phone for photography?", 0),
    ("How do I stop my phone from overheating?", "How do I stop my phone from lagging?", 0),
    ("What are the symptoms of the flu?", "What are the symptoms of a cold?", 0),
    ("How do I learn to play guitar?", "How do I learn to play piano?", 0),
    ("Where can I watch the new Marvel movie?", "Where can I buy tickets for a concert?", 0),
    ("How do I cancel my subscription?", "How do I upgrade my subscription?", 0),
    ("What causes traffic jams?", "What causes power outages?", 0),
    ("How do I improve my sleep quality?", "How do I improve my credit score?", 0),
    ("What is the best way to save money?", "What is the best way to invest money?", 0),
    ("How do airplanes stay in the air?", "How do submarines stay underwater?", 0),
    ("How do I remove a stain from a shirt?", "How do I remove wrinkles from a shirt?", 0),
    ("What's the difference between a virus and a worm?", "What's the difference between RAM and storage?", 0),
    ("How do I write a cover letter?", "How do I write a resignation letter?", 0),
    ("What are good exercises for lower back pain?", "What are good exercises for neck pain?", 0),
    ("How do I connect my printer to WiFi?", "How do I connect my TV to WiFi?", 0),
    ("Why does my car make a rattling noise?", "Why does my car use so much gas?", 0),
    ("What is the tallest mountain in the world?", "What is the longest river in the world?", 0),
    ("How do I make coffee without a machine?", "How do I make tea without a kettle?", 0),
    ("How do I make my WiFi faster?", "How do I set up a new WiFi router?", 0),
    ("How do I speed up my internet connection?", "How do I make my WiFi faster?", 1),
    ("What is the best diet for weight loss?", "What is the best diet for muscle gain?", 0),
    ("How do I fix a slow laptop?", "How do I make my laptop run faster?", 1),
]


# ---------------------------------------------------------------------------
# 2. Text preprocessing helpers
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> set:
    return set(clean_text(text).split())


# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
@dataclass
class PairFeatures:
    tfidf_cosine: float
    jaccard: float
    common_word_ratio: float
    length_diff_norm: float
    sequence_ratio: float

    def as_array(self):
        return np.array([
            self.tfidf_cosine,
            self.jaccard,
            self.common_word_ratio,
            self.length_diff_norm,
            self.sequence_ratio,
        ])


def build_vectorizer(all_sentences):
    """Fit one shared TF-IDF vectorizer over every sentence in the dataset
    so that cosine similarity between any two sentences is comparable."""
    vectorizer = TfidfVectorizer(preprocessor=clean_text)
    vectorizer.fit(all_sentences)
    return vectorizer


def extract_features(q1: str, q2: str, vectorizer: TfidfVectorizer) -> PairFeatures:
    # TF-IDF cosine similarity
    vecs = vectorizer.transform([q1, q2])
    tfidf_cosine = float(cosine_similarity(vecs[0], vecs[1])[0][0])

    # Jaccard similarity of token sets
    tokens1, tokens2 = tokenize(q1), tokenize(q2)
    union = tokens1 | tokens2
    jaccard = len(tokens1 & tokens2) / len(union) if union else 0.0

    # Common word ratio (relative to the shorter sentence)
    shorter_len = min(len(tokens1), len(tokens2)) or 1
    common_word_ratio = len(tokens1 & tokens2) / shorter_len

    # Normalized length difference (0 = same length, 1 = very different)
    len1, len2 = len(tokens1), len(tokens2)
    max_len = max(len1, len2) or 1
    length_diff_norm = abs(len1 - len2) / max_len

    # Character-level similarity via difflib (catches typos/rewording)
    sequence_ratio = difflib.SequenceMatcher(None, clean_text(q1), clean_text(q2)).ratio()

    return PairFeatures(tfidf_cosine, jaccard, common_word_ratio, length_diff_norm, sequence_ratio)


# ---------------------------------------------------------------------------
# 4. Model training
# ---------------------------------------------------------------------------
def build_feature_matrix(pairs, vectorizer):
    X, y = [], []
    for q1, q2, label in pairs:
        feats = extract_features(q1, q2, vectorizer)
        X.append(feats.as_array())
        y.append(label)
    return np.vstack(X), np.array(y)


def train_model():
    all_sentences = [s for q1, q2, _ in DATASET for s in (q1, q2)]
    vectorizer = build_vectorizer(all_sentences)

    X, y = build_feature_matrix(DATASET, vectorizer)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.8, class_weight="balanced"))
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    print("=" * 60)
    print("Paraphrase / Duplicate Question Detector -- Evaluation")
    print("=" * 60)
    print(f"Train size: {len(X_train)}   Test size: {len(X_test)}")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1 score : {f1:.3f}")
    print("Confusion matrix (rows=true, cols=predicted) [0=not dup, 1=dup]:")
    print(cm)
    print()

    feature_names = ["tfidf_cosine", "jaccard", "common_word_ratio",
                      "length_diff_norm", "sequence_ratio"]
    print("Learned feature weights (positive => pushes toward 'duplicate'):")
    log_reg = clf.named_steps["logisticregression"]
    for name, weight in zip(feature_names, log_reg.coef_[0]):
        print(f"  {name:20s}: {weight:+.3f}")
    print()

    return clf, vectorizer


# ---------------------------------------------------------------------------
# 5. Convenience predictor for arbitrary sentence pairs
# ---------------------------------------------------------------------------
def predict_pair(q1: str, q2: str, clf: LogisticRegression, vectorizer: TfidfVectorizer):
    feats = extract_features(q1, q2, vectorizer)
    proba = clf.predict_proba(feats.as_array().reshape(1, -1))[0][1]
    label = "DUPLICATE" if proba >= 0.5 else "NOT duplicate"
    return label, proba, feats


# ---------------------------------------------------------------------------
# 6. Demo
# ---------------------------------------------------------------------------
DEMO_PAIRS = [
    ("How do I make my WiFi faster?", "What can I do to speed up my WiFi connection?"),
    ("How do I make my WiFi faster?", "How do I set up a new WiFi router?"),
    ("What's a good beginner recipe for pasta?", "Can you suggest an easy pasta recipe for beginners?"),
    ("What's a good beginner recipe for pasta?", "What's a good beginner recipe for pizza dough?"),
    ("How does photosynthesis work?", "Can you explain how plants make food through photosynthesis?"),
]


def run_demo(clf, vectorizer):
    print("=" * 60)
    print("Demo predictions on new, unseen sentence pairs")
    print("=" * 60)
    for q1, q2 in DEMO_PAIRS:
        label, proba, feats = predict_pair(q1, q2, clf, vectorizer)
        print(f'Q1: "{q1}"')
        print(f'Q2: "{q2}"')
        print(f"  -> {label}  (duplicate probability: {proba:.3f}, "
              f"tfidf_cosine={feats.tfidf_cosine:.2f}, jaccard={feats.jaccard:.2f})")
        print()


if __name__ == "__main__":
    trained_clf, trained_vectorizer = train_model()
    run_demo(trained_clf, trained_vectorizer)

    print("=" * 60)
    print("Try your own pair (press Enter with empty input to quit)")
    print("=" * 60)
    try:
        while True:
            q1 = input("Sentence 1: ").strip()
            if not q1:
                break
            q2 = input("Sentence 2: ").strip()
            if not q2:
                break
            label, proba, _ = predict_pair(q1, q2, trained_clf, trained_vectorizer)
            print(f"  -> {label} (probability: {proba:.3f})\n")
    except (EOFError, KeyboardInterrupt):
        pass
