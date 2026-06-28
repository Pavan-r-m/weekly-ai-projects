"""
TextRank Extractive Summarizer
==============================
Implements the TextRank algorithm from scratch to rank and extract the most
important sentences from any document -- no external API or model needed.

How it works:
  1. Tokenize text into sentences.
  2. Build TF-IDF vectors for each sentence.
  3. Compute a cosine-similarity matrix (the "graph").
  4. Run PageRank on that graph to score sentences.
  5. Extract the top-N highest-scored sentences in original order.
  6. Optionally visualise the similarity matrix and sentence-score bar chart.

Usage:
    python summarizer.py                  # demo on built-in articles
    python summarizer.py --file my.txt   # summarise a file
    python summarizer.py --ratio 0.3     # keep 30% of sentences (default 0.25)
    python summarizer.py --plot          # save visualisation PNGs
"""

import re
import sys
import math
import argparse
import textwrap
from typing import List, Tuple, Dict

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Text pre-processing
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> List[str]:
    """Split text into sentences on '.', '!', '?' followed by whitespace."""
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 10]


def tokenize(sentence: str) -> List[str]:
    """Lowercase, remove punctuation, split on whitespace."""
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-z0-9\s]", " ", sentence)
    return sentence.split()


STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "as", "is", "was", "are",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "shall",
    "can", "it", "its", "that", "this", "these", "those", "i", "you",
    "he", "she", "we", "they", "what", "which", "who", "how", "when",
    "where", "why", "not", "no", "so", "if", "about", "into", "than",
    "then", "there", "their", "them", "s", "t", "re",
}


def clean_tokens(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


# ---------------------------------------------------------------------------
# 2. TF-IDF matrix
# ---------------------------------------------------------------------------

def build_tfidf(sentences: List[str]) -> np.ndarray:
    """Return an (N x V) TF-IDF matrix for N sentences and vocabulary size V."""
    tokenized = [clean_tokens(tokenize(s)) for s in sentences]
    N = len(tokenized)

    # Build vocabulary
    vocab: Dict[str, int] = {}
    for tokens in tokenized:
        for t in tokens:
            if t not in vocab:
                vocab[t] = len(vocab)
    V = len(vocab)
    if V == 0:
        return np.zeros((N, 1))

    # Term frequency: raw count / sentence length
    tf = np.zeros((N, V), dtype=float)
    for i, tokens in enumerate(tokenized):
        for t in tokens:
            tf[i, vocab[t]] += 1
        total = tf[i].sum()
        if total > 0:
            tf[i] /= total

    # Smoothed inverse document frequency
    df = np.zeros(V, dtype=float)
    for i in range(N):
        df[tf[i] > 0] += 1
    idf = np.log((N + 1) / (df + 1)) + 1

    return tf * idf  # (N x V)


# ---------------------------------------------------------------------------
# 3. Cosine similarity matrix
# ---------------------------------------------------------------------------

def cosine_similarity_matrix(tfidf: np.ndarray) -> np.ndarray:
    """Return an (N x N) pairwise cosine-similarity matrix."""
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    norm_matrix = tfidf / norms
    sim = norm_matrix @ norm_matrix.T   # (N x N)
    np.fill_diagonal(sim, 0.0)          # no self-loops
    return sim


# ---------------------------------------------------------------------------
# 4. PageRank via power iteration
# ---------------------------------------------------------------------------

def pagerank(sim: np.ndarray, damping: float = 0.85,
             tol: float = 1e-6, max_iter: int = 100) -> np.ndarray:
    """
    Run PageRank on a weighted adjacency matrix.
    Returns a score vector of length N.
    """
    N = sim.shape[0]
    row_sums = sim.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-10
    trans = sim / row_sums              # row-stochastic transition matrix

    scores = np.ones(N) / N             # uniform start
    for _ in range(max_iter):
        new_scores = (1 - damping) / N + damping * trans.T @ scores
        if np.linalg.norm(new_scores - scores) < tol:
            break
        scores = new_scores
    return new_scores


# ---------------------------------------------------------------------------
# 5. Main summarise function
# ---------------------------------------------------------------------------

def summarise(text: str, ratio: float = 0.25,
              min_sentences: int = 1, max_sentences: int = 20
              ) -> Tuple[str, List[Tuple[int, float, str]]]:
    """
    Summarise *text* by extracting the most important sentences.

    Returns
    -------
    summary : Extracted summary string.
    ranked  : List of (original_index, score, sentence) sorted by score DESC.
    """
    sentences = split_sentences(text)
    if not sentences:
        return "", []

    n_keep = max(min_sentences,
                 min(max_sentences, int(math.ceil(len(sentences) * ratio))))
    n_keep = min(n_keep, len(sentences))

    tfidf  = build_tfidf(sentences)
    sim    = cosine_similarity_matrix(tfidf)
    scores = pagerank(sim)

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    ranked_with_text = [(idx, score, sentences[idx]) for idx, score in ranked]

    # Keep top-N sentences, restore original document order
    top_indices = sorted(idx for idx, _, _ in ranked_with_text[:n_keep])
    summary = " ".join(sentences[i] for i in top_indices)

    return summary, ranked_with_text


# ---------------------------------------------------------------------------
# 6. Visualisations
# ---------------------------------------------------------------------------

def plot_similarity_matrix(sim: np.ndarray, title: str = "",
                            out_path: str = "similarity_matrix.png"):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(sim, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_xlabel("Sentence index")
    ax.set_ylabel("Sentence index")
    ax.set_title(f"Sentence Similarity Matrix — {title}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [plot] {out_path}")


def plot_scores(sentences: List[str], scores: np.ndarray, n_top: int,
                title: str = "", out_path: str = "sentence_scores.png"):
    threshold = sorted(scores, reverse=True)[n_top - 1]
    colors = ["#e07b54" if s >= threshold else "#90b8d4" for s in scores]
    labels = [f"S{i+1}" for i in range(len(sentences))]

    fig, ax = plt.subplots(figsize=(max(8, len(sentences) * 0.65), 4))
    ax.bar(labels, scores, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Sentence")
    ax.set_ylabel("TextRank score")
    ax.set_title(f"TextRank Scores (orange = selected) — {title}", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [plot] {out_path}")


# ---------------------------------------------------------------------------
# 7. Built-in demo articles
# ---------------------------------------------------------------------------

DEMO_ARTICLES = {
    "machine_learning": """\
Machine learning is a subset of artificial intelligence that enables systems to learn and
improve from experience without being explicitly programmed. It focuses on developing
computer programs that can access data and use it to learn for themselves.
The process begins with observations or data, such as examples, direct experience, or
instruction, so that computers can look for patterns in data and make better decisions.
The primary aim is to allow computers to learn automatically without human intervention
and adjust actions accordingly.
Supervised learning algorithms are trained using labeled examples where the desired output
is known, enabling the model to generalise to unseen data.
Unsupervised learning is used against data that has no historical labels, and the algorithm
must discover hidden structure on its own.
Reinforcement learning is often used for robotics, gaming and navigation, where an agent
discovers through trial and error which actions yield the greatest cumulative reward.
Deep learning is a subfield of machine learning inspired by the structure of the brain,
using artificial neural networks with many layers to learn hierarchical representations.
Neural networks consist of layers of interconnected nodes that allow the model to learn
representations of data with multiple levels of abstraction.
Transfer learning is a technique where a model trained on one task is re-purposed on a
related task, dramatically reducing the amount of labeled data and compute required.""",

    "climate_change": """\
Climate change refers to long-term shifts in temperatures and weather patterns on Earth.
Since the 1800s, human activities have been the main driver of climate change, primarily
through the burning of fossil fuels such as coal, oil, and natural gas.
Burning fossil fuels generates greenhouse gas emissions that act like a blanket around
the Earth, trapping the sun's heat and raising global average temperatures.
Carbon dioxide and methane are the primary greenhouse gases contributing to global warming.
Consequences include intense droughts, water scarcity, severe wildfires, rising sea levels,
flooding, melting polar ice, catastrophic storms, and declining biodiversity.
International agreements such as the Paris Agreement aim to limit global temperature rise
to 1.5 degrees Celsius above pre-industrial levels.
Renewable energy sources, electric vehicles, energy-efficient buildings, and sustainable
agriculture are among the key solutions to reduce greenhouse gas emissions.
Carbon capture technologies can remove carbon dioxide directly from the atmosphere,
offering another pathway to address accumulated historical emissions.
Climate adaptation involves preparing for warming already locked in, such as building
sea walls, developing drought-resistant crops, and redesigning cities for extreme heat.
Individual actions such as reducing meat consumption, flying less, and choosing sustainable
products collectively contribute a meaningful portion of necessary emissions reductions.""",

    "quantum_computing": """\
Quantum computing is a type of computation that harnesses quantum mechanical phenomena
such as superposition and entanglement to process information.
Unlike classical computers that store information as binary bits, quantum computers use
quantum bits, or qubits, which can exist in multiple states simultaneously.
Superposition allows quantum computers to explore many possible solutions to a problem
at the same time, giving them potential exponential speedups over classical machines.
Entanglement links two qubits such that the state of one instantly influences the state
of the other, regardless of the physical distance between them.
Quantum computers could solve specific problems exponentially faster, including simulating
molecular interactions for drug discovery, breaking certain encryption schemes, and
optimising complex logistics and financial networks.
Shor's algorithm, on a large enough quantum computer, could factor integers efficiently,
threatening the RSA cryptography that underlies most internet security today.
Error correction is a critical challenge because qubits are highly sensitive to
environmental noise and decoherence, causing rapid loss of quantum information.
Companies like IBM, Google, and IonQ are racing to achieve quantum advantage, the point
at which a quantum device outperforms any classical computer on a practically useful task.
Post-quantum cryptography is developing encryption algorithms that remain secure even
against attacks by future large-scale quantum computers.
The field is advancing rapidly, with qubit counts and coherence times improving yearly.""",
}


# ---------------------------------------------------------------------------
# 8. CLI entry point
# ---------------------------------------------------------------------------

def separator(char="─", width=70):
    print(char * width)


def run_demo(ratio: float, do_plot: bool):
    for name, text in DEMO_ARTICLES.items():
        separator("═")
        print(f"  ARTICLE: {name.replace('_', ' ').title()}")
        separator("═")
        sentences = split_sentences(text)
        print(f"  Original : {len(sentences)} sentences, ~{len(text.split())} words")

        summary, ranked = summarise(text, ratio=ratio)
        n_keep = max(1, int(math.ceil(len(sentences) * ratio)))
        print(f"  Summary  : {n_keep} sentences  (ratio={ratio})\n")
        for line in textwrap.wrap(summary, width=68):
            print(f"    {line}")

        print(f"\n  Top-3 by TextRank score:")
        for pos, (idx, score, sent) in enumerate(ranked[:3], 1):
            short = sent[:88] + ("…" if len(sent) > 88 else "")
            print(f"    #{pos} [S{idx+1}  score={score:.4f}]  {short}")

        if do_plot:
            tfidf  = build_tfidf(sentences)
            sim    = cosine_similarity_matrix(tfidf)
            scores = pagerank(sim)
            plot_similarity_matrix(sim, title=name, out_path=f"{name}_sim.png")
            plot_scores(sentences, scores, n_keep, title=name,
                        out_path=f"{name}_scores.png")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="TextRank extractive summariser — no API key needed",
    )
    parser.add_argument("--file",  default=None, help="Plain-text file to summarise")
    parser.add_argument("--ratio", type=float, default=0.25,
                        help="Fraction of sentences to keep (default 0.25)")
    parser.add_argument("--min",   type=int, default=1,
                        help="Min sentences in output (default 1)")
    parser.add_argument("--max",   type=int, default=20,
                        help="Max sentences in output (default 20)")
    parser.add_argument("--plot",  action="store_true",
                        help="Save similarity matrix + score bar chart PNGs")
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
        sentences = split_sentences(text)
        print(f"File: {args.file}  |  {len(sentences)} sentences")
        summary, ranked = summarise(text, ratio=args.ratio,
                                    min_sentences=args.min, max_sentences=args.max)
        print("\nSummary:\n")
        for line in textwrap.wrap(summary, width=72):
            print(f"  {line}")
        if args.plot:
            n_keep = max(args.min, min(args.max, int(math.ceil(len(sentences) * args.ratio))))
            tfidf  = build_tfidf(sentences)
            sim    = cosine_similarity_matrix(tfidf)
            scores = pagerank(sim)
            plot_similarity_matrix(sim, out_path="file_sim.png")
            plot_scores(sentences, scores, n_keep, out_path="file_scores.png")
    else:
        run_demo(ratio=args.ratio, do_plot=args.plot)


if __name__ == "__main__":
    main()
