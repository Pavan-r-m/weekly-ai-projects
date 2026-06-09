# Zero-Shot Text Topic Classifier

A lightweight NLP tool that classifies any text into user-defined topic categories **without any labeled training data**. Just describe each topic with a handful of example sentences and you're ready to classify.

## What It Does & Why It's Interesting

Most text classifiers require hundreds or thousands of labeled examples to train. This project takes a different approach: **zero-shot classification via TF-IDF prototype matching**.

You define each topic with a few "prototype" sentences (7–10 is plenty), and the classifier measures how semantically similar an unknown text is to each topic using TF-IDF vectorization and cosine similarity. No training loop, no GPU, no API keys — just pure Python and linear algebra.

This approach is surprisingly effective for well-defined, distinct topics and is fast enough to classify thousands of texts per second on a laptop.

## Tech Stack & Key Concepts

- **TF-IDF (Term Frequency–Inverse Document Frequency)** — weights rare, discriminative words more heavily than common ones
- **Cosine Similarity** — measures the angular similarity between two document vectors, independent of length
- **Prototype Centroids** — each topic is represented by the average TF-IDF vector of its prototype sentences
- **Zero-shot classification** — no training data needed; topic definitions are specified at runtime
- Everything implemented from scratch using only the **Python standard library** (`math`, `re`, `json`, `csv`, `collections`)

## Installation

```bash
# No dependencies needed! Just Python 3.8+
python classifier.py --demo
```

If you want optional matplotlib visualizations in future extensions:
```bash
pip install -r requirements.txt
```

## How to Run

### Run the built-in demo (8 sample texts across 6 topics)
```bash
python classifier.py --demo
```

### Classify a single text
```bash
python classifier.py --text "The election results surprised many political analysts."
```

### Classify texts from a CSV file
```bash
python classifier.py --csv my_articles.csv --column headline
```

### Use your own custom topics (JSON file)
```bash
python classifier.py --demo --topics my_topics.json
python classifier.py --csv news.csv --topics my_topics.json
```

### List available built-in topics
```bash
python classifier.py --list-topics
```

### Compact output (no bar charts)
```bash
python classifier.py --demo --no-bar
```

## Custom Topics Format

Create a `my_topics.json` file:

```json
{
  "cooking": [
    "Bake the chicken at 375 degrees for 45 minutes until golden brown.",
    "Sauté the onions in olive oil until translucent, then add garlic.",
    "Season the pasta water with a generous pinch of salt before boiling."
  ],
  "travel": [
    "The flight from New York to London takes about seven hours.",
    "The hotel offered stunning views of the Mediterranean coastline.",
    "Pack light and bring a universal adapter for international trips."
  ]
}
```

## Example Output

```
Initializing Zero-Shot Classifier with 6 topics...
Vocabulary size: 312 unique terms

============================================================
  CLASSIFICATION RESULTS
============================================================

------------------------------------------------------------
Sample #1
   Text: "Researchers have developed a new vaccine that shows 94% efficacy against the virus..."

   Predicted Topic: HEALTH

   Similarity Scores:
  health          ████████████████████████████████████████  0.3821
  science         ██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░  0.1402
  technology      ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.0891
  business        ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.0312
  politics        ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.0244
  sports          ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.0000
```

## How It Works

### 1. Prototype Definition
Each topic is defined by 7–10 representative sentences that capture its vocabulary and style. These are the only "supervision" the classifier receives.

### 2. TF-IDF Vectorization
All prototype sentences are pooled together to build a shared vocabulary. Each sentence (and later, each input text) is converted into a sparse vector where:
- **TF (term frequency)** = how often a word appears in this document
- **IDF (inverse document frequency)** = how rare the word is across all documents

Words that appear only in "technology" prototypes get high weights for that topic.

### 3. Topic Centroids
Each topic's prototype vectors are averaged to produce a single **centroid vector** that represents the "middle" of that topic's semantic space.

### 4. Classification
To classify a new text:
1. Vectorize it with the same TF-IDF weights
2. Compute cosine similarity to each topic centroid
3. Assign the topic with the highest similarity

### 5. Why Cosine Similarity?
Cosine similarity measures the **angle** between two vectors rather than their magnitude. This means a short tweet and a long article can match well if they use similar vocabulary — length doesn't bias the score.

## Limitations & Extensions

- **Vocabulary overlap is required** — if a text uses completely different words than the prototypes, similarity will be low for all topics. Stemming or lemmatization can help.
- **For better accuracy**, consider replacing TF-IDF with sentence embeddings (e.g., `sentence-transformers`) for semantic matching beyond keyword overlap.
- **More prototypes = better coverage** — adding 20–30 prototypes per topic significantly improves recall for edge cases.

## License

MIT — free to use, modify, and distribute.
