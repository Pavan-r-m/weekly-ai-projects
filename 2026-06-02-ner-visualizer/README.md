# Named Entity Recognition (NER) Visualizer

A colorful terminal tool that finds and highlights **named entities** (people, organizations, locations, dates, money, and more) inside any text — powered by **spaCy** with an automatic regex fallback so it always runs, even without a model download.

---

## What it does

Given a piece of text, the tool:

1. **Extracts** named entities using spaCy's `en_core_web_sm` model (or built-in regex patterns when spaCy is unavailable)
2. **Highlights** entities inline with ANSI color codes — each entity type gets its own color
3. **Tabulates** every detected entity with its type and character position
4. **Summarizes** entity-type frequencies and shows the most common entity per type
5. Supports **single text**, **batch file**, **interactive**, and **JSON output** modes

---

## Tech stack

| Layer | Tools |
|---|---|
| NLP backbone | [spaCy](https://spacy.io/) `en_core_web_sm` |
| Fallback extraction | Hand-crafted regex patterns |
| Output | ANSI terminal colors (no extra libs) |
| CLI | `argparse` (stdlib) |

**Key concepts:** tokenization, named entity recognition, IOB tagging, entity linking, ANSI escape codes

---

## Installation

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Download the spaCy English model  (optional but recommended)
python -m spacy download en_core_web_sm
```

> **No model?** No problem. The script detects missing dependencies and switches to a regex-based extractor automatically.

---

## How to run

### Built-in demo (three sample texts)
```bash
python ner_visualizer.py --demo
```

### Analyze a single string
```bash
python ner_visualizer.py --text "Elon Musk founded SpaceX in 2002 in Hawthorne, California."
```

### Analyze every line in a file
```bash
python ner_visualizer.py --file my_news_articles.txt
```

### Interactive mode (type text, see entities)
```bash
python ner_visualizer.py
```

### Export as JSON
```bash
python ner_visualizer.py --text "Apple is based in Cupertino." --json
```

---

## Example output

```
NER Visualizer  —  Named Entity Recognition
  Backend: spaCy en_core_web_sm  ✓

──────────────────────────────────────────────────────────────
  Tech News

Highlighted Text:
  Tech giant [Apple Inc.](ORG) announced on [June 1, 2026](DATE)
  that CEO [Tim Cook](PERSON) will visit [Berlin](GPE) and [Tokyo](GPE)
  next week to meet with [European](NORP) and [Japanese](NORP) regulators.
  The company's market cap crossed [$3.5 trillion](MONEY) in [May 2026](DATE).

Entities Found (8):
  Entity        Type    Position
  ──────────────────────────────────
  Apple Inc.    ORG     chars 11–21
  June 1, 2026  DATE    chars 36–48
  Tim Cook      PERSON  chars 57–65
  Berlin        GPE     chars 77–83
  Tokyo         GPE     chars 88–93
  European      NORP    chars 111–119
  Japanese      NORP    chars 124–132
  $3.5 trillion MONEY   chars 172–185

Entity Type Breakdown:
  ORG         █ (1)
  DATE        ██ (2)
  PERSON      █ (1)
  GPE         ██ (2)
  NORP        ██ (2)
  MONEY       █ (1)
```

---

## How it works

1. **spaCy pipeline** — text is passed through a pre-trained transformer-light model that assigns IOB labels to every token. Consecutive labeled tokens are merged into entity spans.
2. **Regex fallback** — when spaCy is absent, a set of hand-crafted patterns matches dates, money values, organizations (via common suffixes like "Inc", "Corp"), GPE proper nouns, and person names (Title + Capitalized words).
3. **Inline rendering** — entities are injected into the original string in *reverse* positional order so earlier character offsets stay valid as the string grows.
4. **JSON mode** — the same structured dict is available for downstream use (piping into `jq`, feeding another script, etc.)

---

## Entity types (spaCy labels)

| Label | Meaning |
|---|---|
| PERSON | People, fictional characters |
| ORG | Companies, agencies, institutions |
| GPE | Countries, cities, states |
| LOC | Non-GPE locations |
| DATE | Dates and periods |
| MONEY | Monetary values |
| PRODUCT | Objects, vehicles, foods |
| EVENT | Named events (wars, summits) |
| LAW | Legal documents |
| NORP | Nationalities, political/religious groups |
