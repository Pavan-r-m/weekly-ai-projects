# Stylometric Author Identifier

Guess who wrote a passage of text — using only *how* it's written, not what
it's about. This project builds a stylometric fingerprint for each author
(function-word frequencies, sentence length, vocabulary richness,
punctuation habits) and trains a classifier to tell authors apart.

## Why it's interesting

Most text classifiers key off *content words* (nouns, topical vocabulary),
which means they're really just topic detectors in disguise. Stylometry
does the opposite: it deliberately looks at "boring" function words like
*the*, *of*, *which*, *upon* — words every writer uses constantly and
mostly unconsciously. Frequencies of these words turn out to be a
surprisingly stable authorial fingerprint, independent of subject matter.
This is the same core technique used in real forensic linguistics and
disputed-authorship investigations (famously, statistical analysis of
function words helped resolve authorship disputes over the *Federalist
Papers*). Here it's applied to four classic novelists — Melville, Austen,
Dickens, and Carroll — using only their opening chapters, and the model
still correctly separates their voices with ~90% cross-validated accuracy.

## Tech stack & key concepts

- **Python 3** with `scikit-learn` (`RandomForestClassifier`,
  `StratifiedKFold` cross-validation), `numpy`, `matplotlib`
- **Feature engineering**: relative frequency of ~65 common English
  function words + 7 surface statistics (average sentence length, average
  word length, type-token ratio, comma/semicolon rate, exclaim/question
  rate, long-word rate)
- **Sliding-window chunking**: turns a handful of short excerpts into many
  overlapping training samples per author
- **Cross-validation** with `cross_val_predict` for an honest accuracy
  estimate on a small dataset, plus a confusion matrix
- **Feature importance** analysis to see which stylistic cues the model
  actually leans on

No API key, internet connection, or large dataset download is required —
a small public-domain demo corpus is bundled directly in `corpus_data.py`.

## Installation

```bash
pip install -r requirements.txt
```

## How to run

Run the full demo (trains on the bundled corpus, cross-validates, and
classifies a held-out "mystery" Melville excerpt that was *not* used in
training):

```bash
python author_identifier.py
```

Also save a feature-importance bar chart (`feature_importance.png`):

```bash
python author_identifier.py --plot
```

Classify your own text:

```bash
python author_identifier.py --text "It is a truth universally acknowledged that a young man in want of good fortune must marry a wife of similar prospects."
```

Classify a text file:

```bash
python author_identifier.py --file some_passage.txt
```

Tune the sliding window used to build training samples:

```bash
python author_identifier.py --window 50 --stride 25
```

## Example output

```
Cross-validated accuracy (4-fold): 89.66%

Classification report:
                 precision    recall  f1-score   support

Charles Dickens       0.83      0.71      0.77         7
Herman Melville       0.82      1.00      0.90         9
    Jane Austen       1.00      0.86      0.92         7
  Lewis Carroll       1.00      1.00      1.00         6

Predicted author (ranked by probability):
  Herman Melville       42.7%  #################
  Jane Austen           26.7%  ##########
  Lewis Carroll         18.7%  #######
```

The top prediction (Herman Melville) is correct — and note the mystery
excerpt is from a *different* Moby-Dick paragraph than any used in
training, so this is a genuine held-out test, not memorization.

Feeding it an Austen-style pastiche sentence ("It is a truth universally
acknowledged...") correctly ranks Jane Austen as the top match too.

## How it works

1. **Corpus prep** (`corpus_data.py`): a handful of famous public-domain
   opening passages, one dictionary entry per author.
2. **Chunking** (`sliding_window_chunks`): each excerpt is tokenized and
   cut into overlapping 40-word windows (stride 20), multiplying a few
   paragraphs into dozens of labeled training samples.
3. **Feature extraction** (`extract_features`): every chunk is converted
   into a numeric vector — the relative frequency (per 100 words) of each
   function word in the vocabulary, plus sentence length, word length,
   type-token ratio, and punctuation rates.
4. **Training & evaluation**: a `RandomForestClassifier` is evaluated with
   stratified k-fold cross-validation so the accuracy estimate isn't
   inflated by testing on training data, then refit on the full dataset.
5. **Prediction**: any new text is run through the same feature extractor
   and the trained forest outputs a probability for each candidate author.

### Scaling it up

The bundled corpus is intentionally tiny so the project runs instantly
with zero setup. For a more robust real-world model, swap `CORPUS` in
`corpus_data.py` for full-length public-domain novels (e.g. downloaded
from Project Gutenberg) — more text per author means more chunks, more
training samples, and a sturdier fingerprint. The pipeline (chunking →
feature extraction → cross-validated Random Forest) scales to that
directly with no code changes.
