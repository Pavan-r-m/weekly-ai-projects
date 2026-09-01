# Paraphrase & Duplicate Question Detector

A lightweight NLP tool that decides whether two sentences (typically
questions) are **duplicates** — i.e. they ask the same thing in different
words — in the spirit of the classic "Quora Question Pairs" problem, but
built entirely from classic, interpretable NLP features instead of a large
pretrained transformer.

## Why it's interesting

Duplicate/paraphrase detection powers real products: deduplicating support
tickets, merging similar FAQ entries, catching repeated forum questions, and
feeding retrieval systems that shouldn't return the same answer twice. This
project shows that you don't need a giant language model to get useful
results — five hand-engineered similarity features plus a simple linear
classifier already separate duplicates from non-duplicates with ~90%+
accuracy on a held-out test set.

It's also a good demonstration of **feature engineering** in NLP: instead of
throwing raw text at a model, we compute numeric signals that each capture a
different notion of "similarity," then let logistic regression learn how to
combine them.

## Tech stack & key concepts

- **Python 3** + **scikit-learn** (TF-IDF vectorization, Logistic
  Regression, train/test split, evaluation metrics)
- **TF-IDF cosine similarity** — measures overall wording/topic overlap
- **Jaccard similarity** — set overlap between the two sentences' vocabularies
- **Common-word ratio** — how much of the *shorter* sentence's vocabulary
  reappears in the longer one
- **Normalized length difference** — very different sentence lengths rarely
  mean "the same question"
- **`difflib.SequenceMatcher` ratio** — character-level similarity, useful
  for catching typos or minor rewordings
- **`StandardScaler` + `LogisticRegression` pipeline** — features are scaled
  before fitting so the linear model's coefficients are comparable and stable

No API key, GPU, or internet connection is required — everything trains and
runs locally in a couple of seconds.

## Installation

```bash
pip install -r requirements.txt
```

## How to run

```bash
python paraphrase_detector.py
```

This will:
1. Train the model on a built-in dataset of ~56 hand-labeled sentence pairs
2. Print accuracy / precision / recall / F1 and a confusion matrix on a held-out test split
3. Print the learned feature weights (which signals matter most)
4. Run the model on 5 fresh demo sentence pairs it never saw during training
5. Drop you into an interactive prompt where you can type your own two
   sentences and see the model's verdict (press Enter on an empty line to quit)

### Example session

```
Sentence 1: How do I fix a leaking faucet?
Sentence 2: What's the best way to repair a dripping tap?
  -> DUPLICATE (probability: 0.81)

Sentence 1: How do I fix a leaking faucet?
Sentence 2: How do I unclog a drain?
  -> NOT duplicate (probability: 0.22)
```

## Example output

```
Accuracy : 0.929
Precision: 0.875
Recall   : 1.000
F1 score : 0.933
Confusion matrix (rows=true, cols=predicted) [0=not dup, 1=dup]:
[[6 1]
 [0 7]]

Learned feature weights (positive => pushes toward 'duplicate'):
  tfidf_cosine        : +0.792
  jaccard             : -0.812
  common_word_ratio   : -0.623
  length_diff_norm    : +1.055
  sequence_ratio      : -0.603
```

## How it works

1. **Preprocessing** — each sentence is lowercased, stripped of punctuation,
   and whitespace-normalized.
2. **Feature extraction** — for every pair of sentences, five numeric
   features are computed (TF-IDF cosine similarity, Jaccard similarity,
   common-word ratio, normalized length difference, and a character-level
   sequence-matching ratio). A single TF-IDF vectorizer is fit across *all*
   sentences in the dataset so that cosine similarities are computed in a
   shared vocabulary space.
3. **Training** — the five features per pair become a 5-dimensional input
   vector. These vectors are standardized (zero mean, unit variance) and fed
   into a `LogisticRegression` classifier, trained on 75% of the labeled
   pairs and evaluated on the remaining 25%.
4. **Inference** — `predict_pair(q1, q2)` extracts the same five features
   for a brand-new sentence pair and returns the model's probability that
   they're duplicates.

### Limitations (by design)

The training set is intentionally small (~56 pairs) so the whole project
stays self-contained and runs instantly without downloading external
corpora. On sentence pairs that are far outside that small dataset's topics
and phrasing patterns, the model can occasionally misjudge close calls —
this is expected behavior for a from-scratch, feature-based classifier
rather than a bug. Swapping in a larger labeled dataset (e.g. the real Quora
Question Pairs dataset) or adding sentence-embedding features would be the
natural next step to improve robustness.
