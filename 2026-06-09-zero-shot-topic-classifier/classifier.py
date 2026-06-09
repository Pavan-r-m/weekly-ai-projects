"""
Zero-Shot Text Topic Classifier
================================
Classify any text into user-defined topic categories WITHOUT any labeled training data.

How it works:
  1. For each topic, define a handful of "prototype" sentences that capture its meaning.
  2. Vectorize both prototypes and input texts using TF-IDF.
  3. Compute cosine similarity between input text and each topic's prototype centroid.
  4. The topic with the highest similarity score wins.

This is a lightweight, dependency-light approach to zero-shot classification that
works surprisingly well for many real-world use cases.
"""

import sys
import math
import argparse
import re
import csv
import json
from collections import defaultdict


# ─────────────────────────────────────────────
# 1. Text preprocessing utilities
# ─────────────────────────────────────────────

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "as", "up", "out", "about", "into",
    "than", "then", "so", "if", "not", "no", "nor", "very", "just", "also",
    "more", "most", "some", "any", "all", "each", "both", "few", "many",
    "other", "such", "what", "which", "who", "whom", "how", "when", "where",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "they",
    "him", "her", "them", "his", "their",
}


def tokenize(text):
    """Lowercase, remove punctuation, split into tokens, drop stop words."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


# ─────────────────────────────────────────────
# 2. TF-IDF vectorizer (built from scratch)
# ─────────────────────────────────────────────

class TFIDFVectorizer:
    """
    A minimal TF-IDF vectorizer.

    Steps:
      - fit(corpus)  : compute IDF weights from a list of documents
      - transform(docs): return TF-IDF vectors as dicts {term: weight}
    """

    def __init__(self):
        self.idf = {}
        self.vocab = []

    def fit(self, corpus):
        """Compute IDF over the corpus."""
        n = len(corpus)
        df = defaultdict(int)
        for doc in corpus:
            tokens = set(tokenize(doc))
            for token in tokens:
                df[token] += 1

        # IDF with smoothing: log((1 + n) / (1 + df)) + 1
        self.idf = {
            term: math.log((1 + n) / (1 + count)) + 1
            for term, count in df.items()
        }
        self.vocab = sorted(self.idf.keys())
        return self

    def _tf(self, tokens):
        """Raw term frequency (normalized by document length)."""
        if not tokens:
            return {}
        freq = defaultdict(int)
        for t in tokens:
            freq[t] += 1
        n = len(tokens)
        return {t: count / n for t, count in freq.items()}

    def transform(self, docs):
        """Return TF-IDF sparse vectors (dicts) for each document."""
        vectors = []
        for doc in docs:
            tokens = tokenize(doc)
            tf = self._tf(tokens)
            vec = {}
            for term, tf_val in tf.items():
                if term in self.idf:
                    vec[term] = tf_val * self.idf[term]
            vectors.append(vec)
        return vectors

    def fit_transform(self, corpus):
        return self.fit(corpus).transform(corpus)


# ─────────────────────────────────────────────
# 3. Cosine similarity between sparse vectors
# ─────────────────────────────────────────────

def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two sparse TF-IDF vectors."""
    dot = sum(vec_a.get(t, 0) * vec_b.get(t, 0) for t in vec_b)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def centroid(vectors):
    """Compute the average (centroid) of a list of sparse vectors."""
    combined = defaultdict(float)
    n = len(vectors)
    if n == 0:
        return {}
    for vec in vectors:
        for term, val in vec.items():
            combined[term] += val / n
    return dict(combined)


# ─────────────────────────────────────────────
# 4. Zero-shot topic classifier
# ─────────────────────────────────────────────

class ZeroShotClassifier:
    """
    Classifies text into one of several topics using prototype sentences.

    Args:
        topics: dict mapping topic name -> list of prototype sentences
    """

    def __init__(self, topics):
        self.topics = topics
        self.vectorizer = TFIDFVectorizer()
        self.topic_centroids = {}
        self._fit()

    def _fit(self):
        """Vectorize all prototypes and compute per-topic centroids."""
        all_prototypes = []
        for protos in self.topics.values():
            all_prototypes.extend(protos)

        # Fit on ALL prototypes so IDF captures vocabulary breadth
        self.vectorizer.fit(all_prototypes)

        for topic, protos in self.topics.items():
            vecs = self.vectorizer.transform(protos)
            self.topic_centroids[topic] = centroid(vecs)

    def classify(self, text):
        """
        Classify a single text.

        Returns:
            (best_topic, scores_dict) where scores_dict maps topic -> similarity [0, 1]
        """
        vec = self.vectorizer.transform([text])[0]
        scores = {
            topic: cosine_similarity(vec, c)
            for topic, c in self.topic_centroids.items()
        }
        best = max(scores, key=scores.__getitem__)
        return best, scores

    def classify_batch(self, texts):
        """Classify a list of texts."""
        return [self.classify(t) for t in texts]


# ─────────────────────────────────────────────
# 5. Visualization: ASCII bar chart
# ─────────────────────────────────────────────

def bar_chart(scores, width=40):
    """Render scores as an ASCII horizontal bar chart."""
    lines = []
    max_score = max(scores.values()) if scores else 1
    topic_width = max(len(t) for t in scores) + 2
    for topic, score in sorted(scores.items(), key=lambda x: -x[1]):
        bar_len = int((score / max_score) * width) if max_score > 0 else 0
        bar = "█" * bar_len + "░" * (width - bar_len)
        label = topic.ljust(topic_width)
        lines.append(f"  {label} {bar}  {score:.4f}")
    return "\n".join(lines)


def print_result(text, best, scores, idx=None):
    """Pretty-print a single classification result."""
    header = f"Sample #{idx}" if idx is not None else "Result"
    print(f"\n{'─'*60}")
    print(f"Sample {header}")
    preview = text[:120] + ("..." if len(text) > 120 else "")
    print(f"   Text: \"{preview}\"")
    print(f"\n   Predicted Topic: {best.upper()}")
    print(f"\n   Similarity Scores:")
    print(bar_chart(scores))
    print()


# ─────────────────────────────────────────────
# 6. Default built-in topics + sample texts
# ─────────────────────────────────────────────

DEFAULT_TOPICS = {
    "technology": [
        "The latest smartphone features a powerful processor and advanced AI camera system.",
        "Software developers use programming languages like Python, JavaScript, and Rust.",
        "Machine learning models are trained on large datasets to recognize patterns.",
        "Cloud computing enables scalable infrastructure for web applications.",
        "Cybersecurity protects networks from hackers, malware, and data breaches.",
        "Artificial intelligence is transforming industries from healthcare to finance.",
        "Open source software allows developers to collaborate on shared codebases.",
        "Silicon chips power everything from laptops to autonomous vehicles.",
    ],
    "sports": [
        "The basketball team won the championship after an incredible overtime finish.",
        "Soccer players train for hours every day to improve their speed and endurance.",
        "The athlete broke the world record in the 100-meter sprint.",
        "Football season begins in autumn with a packed schedule of exciting games.",
        "Tennis players compete at Grand Slam tournaments throughout the year.",
        "The Olympic games bring athletes from every country to compete for gold medals.",
        "Baseball statistics like batting averages help evaluate player performance.",
        "Cycling through mountain stages is one of the most physically demanding sports.",
    ],
    "health": [
        "Regular exercise reduces the risk of heart disease and improves mental well-being.",
        "A balanced diet rich in fruits and vegetables promotes long-term health.",
        "Doctors recommend getting at least eight hours of sleep every night.",
        "Vaccines have saved millions of lives by preventing infectious diseases.",
        "Mental health awareness is crucial for reducing stigma around depression and anxiety.",
        "Physical therapy helps patients recover from injuries and surgeries.",
        "Meditation and mindfulness practices reduce stress and improve focus.",
        "High blood pressure is a silent killer that can lead to strokes and heart attacks.",
    ],
    "politics": [
        "The senator delivered a speech on healthcare reform and economic inequality.",
        "Elections are the cornerstone of democracy, allowing citizens to choose their leaders.",
        "The government passed new legislation to address climate change and carbon emissions.",
        "International diplomacy requires careful negotiation between world leaders.",
        "Political parties compete for voter support through campaigns and debates.",
        "The Supreme Court ruled on a landmark case involving civil rights protections.",
        "Foreign policy decisions can have far-reaching consequences on global stability.",
        "Tax reform is a contentious political issue that divides voters along economic lines.",
    ],
    "science": [
        "Scientists discovered a new exoplanet in the habitable zone of a distant star.",
        "The Large Hadron Collider accelerates particles to near the speed of light.",
        "CRISPR gene editing technology allows researchers to modify DNA sequences.",
        "Climate scientists warn that rising CO2 levels are accelerating global warming.",
        "Quantum computing promises to solve problems beyond classical computer capabilities.",
        "Marine biologists study coral reef ecosystems threatened by ocean acidification.",
        "Astronomers observed gravitational waves produced by merging black holes.",
        "Neuroscience research maps the brain neural connections to understand consciousness.",
    ],
    "business": [
        "The startup raised fifty million dollars in Series B venture capital funding.",
        "Supply chain disruptions have led to product shortages and rising inflation.",
        "Quarterly earnings exceeded analyst expectations, boosting the stock price.",
        "The merger created a global company with operations in thirty countries.",
        "E-commerce continues to grow as consumers shift to online shopping platforms.",
        "The CEO announced a restructuring plan to cut costs and improve profitability.",
        "Interest rate hikes by the central bank aim to control rising consumer prices.",
        "Brand marketing and customer loyalty programs drive repeat sales for retailers.",
    ],
}

SAMPLE_TEXTS = [
    "Researchers have developed a new vaccine that shows 94% efficacy against the virus in clinical trials.",
    "The company stock surged 12% after announcing record quarterly revenue and expanding into new markets.",
    "Python 4.0 introduces optional static typing and significantly faster execution through a new JIT compiler.",
    "The midfielder scored a hat-trick as the team advanced to the finals of the European Championship.",
    "Scientists at CERN announced preliminary evidence for a new subatomic particle not predicted by the Standard Model.",
    "The prime minister called for an emergency session of parliament to debate the new immigration bill.",
    "A Mediterranean diet high in olive oil and fish is linked to lower rates of Alzheimer disease.",
    "The autonomous drone startup secured a government contract to deliver medical supplies to rural areas.",
]


# ─────────────────────────────────────────────
# 7. CLI interface
# ─────────────────────────────────────────────

def load_custom_topics(path):
    """Load custom topics from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("JSON must be a top-level object mapping topic names to lists of strings.")
    return data


def load_texts_from_csv(path, column="text"):
    """Load texts from a CSV file, reading the specified column."""
    texts = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if column not in row:
                raise ValueError(f"Column '{column}' not found. Available: {list(row.keys())}")
            texts.append(row[column].strip())
    return [t for t in texts if t]


def main():
    parser = argparse.ArgumentParser(
        description="Zero-Shot Text Topic Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run the built-in demo with default topics and sample texts
  python classifier.py --demo

  # Classify a single piece of text
  python classifier.py --text "The election results surprised many political analysts."

  # Classify texts from a CSV file
  python classifier.py --csv my_texts.csv --column body

  # Use custom topics defined in a JSON file
  python classifier.py --demo --topics my_topics.json

  # List all available built-in topics
  python classifier.py --list-topics
        """,
    )
    parser.add_argument("--demo", action="store_true", help="Run the built-in demo")
    parser.add_argument("--text", type=str, help="Classify a single text string")
    parser.add_argument("--csv", type=str, help="Path to a CSV file of texts to classify")
    parser.add_argument("--column", type=str, default="text", help="Column name in CSV (default: text)")
    parser.add_argument("--topics", type=str, help="Path to custom topics JSON file")
    parser.add_argument("--list-topics", action="store_true", help="List built-in topics and exit")
    parser.add_argument("--no-bar", action="store_true", help="Suppress bar chart output")

    args = parser.parse_args()

    # List topics mode
    if args.list_topics:
        print("\nBuilt-in topics:")
        for topic, protos in DEFAULT_TOPICS.items():
            print(f"  - {topic}  ({len(protos)} prototypes)")
        print()
        return

    # Load topics
    if args.topics:
        topics = load_custom_topics(args.topics)
        print(f"\nLoaded {len(topics)} custom topics from {args.topics}")
    else:
        topics = DEFAULT_TOPICS

    # Initialize classifier
    print(f"\nInitializing Zero-Shot Classifier with {len(topics)} topics...")
    clf = ZeroShotClassifier(topics)
    print(f"Vocabulary size: {len(clf.vectorizer.vocab)} unique terms")

    # Determine texts to classify
    texts = []
    if args.text:
        texts = [args.text]
    elif args.csv:
        texts = load_texts_from_csv(args.csv, args.column)
        print(f"Loaded {len(texts)} texts from {args.csv}")
    elif args.demo:
        texts = SAMPLE_TEXTS
        print(f"Running demo on {len(texts)} sample texts")
    else:
        parser.print_help()
        return

    # Classify
    results = clf.classify_batch(texts)

    # Display results
    print(f"\n{'='*60}")
    print("  CLASSIFICATION RESULTS")
    print(f"{'='*60}")

    for i, (text, (best, scores)) in enumerate(zip(texts, results), 1):
        if args.no_bar:
            preview = text[:80] + ("..." if len(text) > 80 else "")
            print(f"  [{i:02d}] {best.upper():15s}  \"{preview}\"")
        else:
            print_result(text, best, scores, idx=i)

    # Summary table
    if len(texts) > 1:
        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        from collections import Counter
        counts = Counter(best for _, (best, _) in zip(texts, results))
        for topic, count in sorted(counts.items(), key=lambda x: -x[1]):
            pct = count / len(texts) * 100
            print(f"  {topic:15s}  {count:3d} texts  ({pct:.0f}%)")
        print()


if __name__ == "__main__":
    main()
