# Tweet Emotion Classifier

A multi-class NLP classifier that reads short, informal, tweet-style text
and predicts the dominant emotion behind it — **joy, sadness, anger, fear,
surprise, or love**.

## Why this is interesting

Emotion detection goes a step beyond simple positive/negative sentiment
analysis: it tries to identify *which* feeling is present, which is much
more useful for things like customer support triage, mental-health check-in
tools, social listening dashboards, or content moderation. This project
shows the full, classic NLP pipeline end-to-end — from raw text to a
trained, evaluated, and interpretable model — using nothing but
lightweight, CPU-friendly tools.

## Tech stack & key concepts

- **scikit-learn** — `TfidfVectorizer` (unigrams + bigrams) and
  `LogisticRegression` for a fast, interpretable multi-class classifier
- **pandas** — loading and manipulating the labeled dataset
- **matplotlib** — rendering a confusion matrix heatmap
- **Text preprocessing** — lowercasing, URL/@mention stripping, hashtag
  normalization, light punctuation cleanup (regex-based, no heavyweight
  NLP downloads required)
- **Model interpretability** — printing the top TF-IDF-weighted words per
  emotion class straight from the trained logistic regression coefficients

## Installation

```bash
pip install -r requirements.txt
```

## How to run

```bash
# 1. Generate the labeled synthetic tweet dataset (only needs to be run once)
python generate_dataset.py

# 2. Train, evaluate, and demo the classifier
python emotion_classifier.py
```

This produces:
- A held-out test accuracy score and full classification report in the
  terminal
- `output/confusion_matrix.png` — a heatmap of predicted vs. true emotions
- A printed list of each emotion's most predictive words
- Live predictions on six brand-new, hand-written example sentences the
  model has never seen

## About the dataset

Real tweet datasets typically require API keys or scraping, which this
project intentionally avoids so it runs fully offline. Instead,
`generate_dataset.py` builds **360 unique, natural-sounding, tweet-style
sentences** (60 per emotion) by randomly combining hand-written trigger
phrases, feeling statements, emojis, and hashtags for each emotion —
seeded for reproducibility. The result reads like real social media
text (e.g. *"Just got the job offer and I am so happy right now 🎉
#blessed"*) without any copyright or privacy concerns.

## Example output

```
Test accuracy: 1.000

Classification report:
              precision    recall  f1-score   support
       anger       1.00      1.00      1.00        15
        fear       1.00      1.00      1.00        15
         joy       1.00      1.00      1.00        15
        love       1.00      1.00      1.00        15
     sadness       1.00      1.00      1.00        15
    surprise       1.00      1.00      1.00        15

Top TF-IDF features per emotion:
  anger     : overit, furious, done, seething, seething about, had it
  love      : my heart, heart, love, grateful, loved, could burst
  ...

Demo predictions on new sentences:
  [surprise  0.48]  I can't believe my team actually won the game, I'm ecstatic!
  [anger     0.63]  My package never showed up and support just hung up on me again.
  [love      0.57]  My dad drove six hours just to help me move, I feel so cared for.
  [joy       0.34]  Ever since the layoffs were announced I've felt completely empty.
```

Note the test accuracy is a perfect 1.0 because the synthetic dataset is
template-generated (the vocabulary per emotion is fairly distinct). The
demo predictions on freely-written new sentences are more revealing: the
model nails most of them but gets tripped up on subtler cases (e.g.
"layoffs...completely empty" gets misread as joy because "layoffs" alone
isn't in its vocabulary and "completely" appears near positive contexts
in training) — a realistic illustration of how a lexical, TF-IDF-based
model can miss context and negation compared to a deep contextual model.

## How it works

1. **Data generation** (`generate_dataset.py`): word banks of triggers,
   feeling phrases, emojis, and hashtags for six emotions are randomly
   assembled into full sentences using several sentence templates,
   producing a diverse, de-duplicated, labeled CSV.
2. **Preprocessing** (`clean_text` in `emotion_classifier.py`): lowercases
   text, strips URLs and @mentions, unwraps `#hashtags` into plain words,
   and removes emoji/punctuation noise while keeping `!` and `?` since
   they carry emotional signal.
3. **Vectorization**: `TfidfVectorizer` converts cleaned text into
   weighted unigram + bigram features, so phrases like "not happy" are
   distinguishable from "happy" alone.
4. **Classification**: a `LogisticRegression` model is trained on a
   75/25 stratified train/test split across the six emotion classes.
5. **Evaluation**: accuracy, per-class precision/recall/F1, and a
   confusion matrix heatmap are generated to visualize where the model
   confuses emotions.
6. **Interpretability**: the logistic regression's learned coefficients
   are used to surface each class's most emotion-indicative words.
7. **Live demo**: the trained pipeline predicts emotions (with confidence
   scores) on six new sentences it never saw during training.

## Extending this project

- Swap in a real labeled dataset (e.g. the public "Emotion" dataset on
  Hugging Face) by matching the `text,emotion` CSV schema
- Try a transformer-based model (e.g. `distilbert-base-uncased` fine-tuned
  for emotion classification) for better contextual understanding
- Add multi-label support so a single tweet can carry more than one emotion
- Wrap the trained pipeline in a small Flask/FastAPI endpoint for live
  predictions
