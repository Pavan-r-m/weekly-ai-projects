# Aspect-Based Sentiment Analysis for Product Reviews

## What it does

Most sentiment tools give you a single positive/negative score per review. But
real reviews are rarely one-note — "battery life is amazing but the camera is
disappointing" is both praise and complaint in one sentence. This project
performs **aspect-based sentiment analysis (ABSA)**: it identifies which
specific product aspects (battery, camera, screen, price, performance, build
quality, customer service, sound) are mentioned in each sentence of a review,
and scores the sentiment of that mention independently.

The result is a much more actionable breakdown — instead of "PhoneX reviews
are mixed," you get "PhoneX camera and screen are loved, but battery and
customer service are hurting the product."

This is genuinely how product/UX teams triage feedback at scale (Amazon,
Yelp, app store reviews) — pinpointing *which* feature to fix rather than
just knowing overall sentiment is trending down.

## Tech stack & key concepts

- **Python 3** with **pandas** for data handling
- **VADER** (`vaderSentiment`) — a lexicon- and rule-based sentiment analyzer
  tuned for short, informal text (handles negation, intensifiers like "very",
  punctuation emphasis, etc.) without needing to train or download a model
- **matplotlib** for per-aspect sentiment bar charts
- A hand-built **aspect lexicon**: a keyword dictionary mapping canonical
  aspect names to the words/phrases that signal them (e.g. `battery` →
  `["battery", "charge", "charging", "standby"]`)
- Lightweight **regex sentence splitting** — no heavyweight NLP model
  download required, so the whole pipeline runs fully offline

## How it works

1. **Sentence splitting** — each review is split into sentences on
   `. ! ?` boundaries.
2. **Aspect tagging** — every sentence is scanned against the aspect
   lexicon; a sentence can mention zero, one, or multiple aspects.
3. **Sentiment scoring** — sentences that mention at least one aspect are
   scored with VADER's compound score (-1 to +1), then labeled
   positive / neutral / negative using ±0.25 thresholds.
4. **Aggregation** — scores are averaged per aspect, both overall and
   per-product, along with mention counts and label breakdowns.
5. **Reporting** — a detailed CSV (one row per sentence-aspect mention) and
   horizontal bar charts (green = positive, red = negative) are saved to
   `output/`.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run on the included sample dataset (28 reviews across 3 products: PhoneX,
LaptopZ, Earbuds):

```bash
python aspect_sentiment.py
```

Restrict to a single product:

```bash
python aspect_sentiment.py --product PhoneX
```

Use your own review data (must have `review_id`, `product`, `review_text`
columns):

```bash
python aspect_sentiment.py --input path/to/your_reviews.csv
```

## Example output

```
=== Aspect Sentiment Summary: All Products ===
Aspect             Avg Score  Mentions   Pos   Neu   Neg
screen                 0.364         7     4     2     1
sound                  0.291         5     2     3     0
build_quality          0.248         8     4     4     0
price                  0.197         9     4     5     0
camera                 0.156         5     2     1     2
performance            0.086        11     2     7     2
battery                0.075        13     4     7     2
customer_service       0.038         7     2     3     2

=== Aspect Sentiment Summary: PhoneX ===
Aspect             Avg Score  Mentions   Pos   Neu   Neg
camera                 0.440         3     2     0     1
screen                 0.389         3     2     0     1
price                  0.106         3     1     2     0
build_quality          0.077         2     0     2     0
performance           -0.065         4     0     3     1
battery               -0.142         4     0     3     1
customer_service      -0.266         2     0     1     1
```

PhoneX's camera and screen are clear strengths, while battery and customer
service are the pain points dragging down the overall impression — exactly
the kind of insight a flat "3.5 stars average" rating would hide.

Running the script also produces, per product and overall, in `output/`:
- `aspect_sentiment_detail.csv` — every sentence-aspect mention with its score
- `aspect_sentiment_overall.png`, `aspect_sentiment_<product>.png` — bar
  charts of average sentiment per aspect

## Notes

- No API key needed — VADER is a fully local, rule-based lexicon, so this
  runs with zero external calls.
- The aspect lexicon (`ASPECT_LEXICON` in `aspect_sentiment.py`) is easy to
  extend — add a new aspect name and its trigger keywords to track more
  product dimensions.
- Swap in your own `product_reviews_sample.csv`-shaped CSV to analyze real
  review data (Amazon/Yelp export, app store reviews, survey free-text, etc.).
