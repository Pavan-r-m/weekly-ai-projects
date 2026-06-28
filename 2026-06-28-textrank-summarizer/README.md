# TextRank Extractive Summarizer

An extractive text summarizer built **from scratch** using the TextRank algorithm — no LLM or API key required. TextRank applies graph-based ranking (PageRank) to sentences, treating each sentence as a node and cosine similarity between TF-IDF vectors as edge weights.

## What it does

Given any plain-text document, the summarizer:
1. Splits the text into sentences
2. Converts each sentence into a TF-IDF feature vector (custom implementation, no sklearn dependency)
3. Builds a fully-connected graph where edge weight = cosine similarity
4. Runs PageRank (power-iteration) over the graph to score each sentence
5. Selects the top-N highest-ranked sentences and returns them in original document order
6. Optionally saves a similarity-matrix heatmap and a sentence-score bar chart

## Key concepts

| Concept | Role |
|---|---|
| TF-IDF | Represent each sentence as a sparse numeric vector |
| Cosine similarity | Measure topical overlap between any two sentences |
| PageRank | Rank sentences by "vote" from other similar sentences |
| Power iteration | Efficient algorithm to compute PageRank without matrix inversion |

## Tech stack

- **Python 3.8+** — standard library for sentence splitting and tokenization
- **NumPy** — TF-IDF matrix, cosine similarity, PageRank iteration
- **NetworkX** — graph construction (optional, used for export)
- **Matplotlib** — similarity heatmap and score bar chart

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run on three built-in demo articles (Machine Learning, Climate Change, Quantum Computing)
python summarizer.py

# Summarise a custom text file
python summarizer.py --file my_article.txt

# Keep 30% of sentences instead of the default 25%
python summarizer.py --ratio 0.3

# Save visualisation PNGs
python summarizer.py --plot

# All options together
python summarizer.py --file report.txt --ratio 0.2 --min 2 --max 10 --plot
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--file` | (demo) | Path to a plain-text file |
| `--ratio` | `0.25` | Fraction of sentences to keep |
| `--min` | `1` | Minimum sentences in summary |
| `--max` | `20` | Maximum sentences in summary |
| `--plot` | off | Save similarity matrix + score chart PNGs |

## Example output

```
══════════════════════════════════════════════════════════════════════
  ARTICLE: Machine Learning
══════════════════════════════════════════════════════════════════════
  Original : 10 sentences, ~168 words
  Summary  : 3 sentences  (ratio=0.25)

    Machine learning is a subset of artificial intelligence that
    enables systems to learn and improve from experience without
    being explicitly programmed. Deep learning is a subfield of
    machine learning inspired by the structure of the brain, using
    artificial neural networks with many layers to learn hierarchical
    representations. Reinforcement learning is often used for
    robotics, gaming and navigation, where an agent discovers through
    trial and error which actions yield the greatest cumulative reward.

  Top-3 by TextRank score:
    #1 [S1  score=0.1823]  Machine learning is a subset of artificial intelligence…
    #2 [S8  score=0.1641]  Deep learning is a subfield of machine learning…
    #3 [S6  score=0.1587]  Supervised learning algorithms are trained using labeled…
```

## How it works

### 1 — Sentence graph

Each sentence is a **node**. An **edge** connects every pair of sentences with weight equal to their TF-IDF cosine similarity. High overlap in key terms → strong edge.

### 2 — PageRank intuition

A sentence is important if *many other important sentences are similar to it*. This is identical to how a web page earns a high PageRank when authoritative pages link to it. The algorithm iterates:

```
score(s) = (1 - d) / N  +  d × Σ [ sim(t, s) / Σ sim(t, ·) × score(t) ]
```

where `d = 0.85` is the damping factor. Convergence is reached in < 30 iterations on typical texts.

### 3 — Extraction

After scoring, the top-`k` sentences (where `k = ceil(N × ratio)`) are selected and returned **in their original document order**, preserving narrative flow.

## Extending the project

- **Sentence embeddings**: swap TF-IDF for `sentence-transformers` embeddings for semantic similarity
- **Query-focused summarisation**: bias PageRank with similarity to a user query
- **Multi-document**: build the graph across sentences from multiple articles
- **Streaming**: process very long documents sentence-by-sentence without loading all into memory
