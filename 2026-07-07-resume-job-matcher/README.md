# Resume-to-Job Semantic Matcher & Skill Gap Analyzer

A recruiting-tech NLP tool that ranks candidate resumes against a job
description using TF-IDF semantic similarity, combined with an
explainable, rule-based skill-gap breakdown.

## What it does and why it's interesting

Most naive resume screeners just count exact keyword hits, which misses
paraphrasing and context. This tool blends two complementary signals:

1. **Semantic similarity** — TF-IDF vectors (unigrams + bigrams) built
   jointly over the job description and all resumes, compared via cosine
   similarity. This captures overall topical relevance even when wording
   differs.
2. **Skill coverage** — a curated taxonomy of ~65 common
   technical/soft skills is matched against both the job description and
   each resume using word-boundary-safe regex, producing a transparent
   list of **matched** vs **missing** skills per candidate.

The final score is a weighted blend (`0.7 * similarity + 0.3 *
skill_coverage`), so candidates are rewarded both for overall textual fit
*and* for explicitly having the skills the role requires — mirroring how
real Applicant Tracking Systems (ATS) combine statistical and rule-based
signals. Unlike a black-box model, a recruiter can see *exactly* why a
candidate ranked where they did.

## Tech stack & key concepts

- **scikit-learn** — `TfidfVectorizer` + `cosine_similarity`
- **Custom skill-extraction taxonomy** — regex-based phrase matching for
  explainability (no ML "black box" for this part)
- **pandas-free CSV reporting** via the standard `csv` module
- **matplotlib** — horizontal bar chart ranking candidates
- Concepts: TF-IDF, cosine similarity, information retrieval, explainable
  AI, weighted scoring

No API keys or model downloads needed — runs fully offline.

## Installation

```bash
pip install -r requirements.txt
```

## How to run

Run with the bundled sample job description and 4 sample resumes:

```bash
python resume_matcher.py
```

Or point it at your own files:

```bash
python resume_matcher.py --job path/to/job_description.txt \
                          --resumes path/to/resumes_folder \
                          --output path/to/output_folder
```

Resumes must be plain `.txt` files in the target folder.

## Example output

```
======================================================================
JOB REQUIREMENTS DETECTED (24 skills):
  agile, aws, communication, data analysis, data visualization, deep
  learning, docker, etl, fastapi, flask, kubernetes, machine learning,
  natural language processing, nlp, numpy, pandas, problem solving,
  python, pytorch, rest api, scikit-learn, scrum, sql, tensorflow
======================================================================

#1  alice_chen
    Final match score : 50.2%
    Text similarity    : 32.4%
    Skill coverage     : 91.7%
    Matched skills     : agile, aws, communication, data analysis, deep
    learning, docker, etl, flask, kubernetes, machine learning, natural
    language processing, nlp, numpy, pandas, problem solving, python,
    pytorch, rest api, scikit-learn, scrum, sql, tensorflow
    Missing skills     : data visualization, fastapi

#2  carmen_diaz    -- Final match score: 25.4%
#3  brian_okafor   -- Final match score: 15.2%
#4  derek_smith    -- Final match score: 1.6%

Saved CSV report to output/match_report.csv
Saved ranking chart to output/match_ranking.png
```

The tool correctly ranks the ML-focused candidate (Alice) first, the
data scientist (Carmen) second despite lower skill overlap with DevOps
terms, the backend engineer (Brian) third, and the front-end-only
candidate (Derek) last — matching human intuition about role fit.

## How it works

1. `extract_skills()` scans text against `SKILL_TAXONOMY` (65+ curated
   terms spanning languages, ML/data tools, cloud/infra, databases, and
   soft skills) using word-boundary regex so short tokens like `r` or
   `go` don't false-positive inside other words.
2. `compute_similarity_scores()` fits one `TfidfVectorizer` across the
   job description plus every resume (so vocabulary and IDF weights are
   shared), then computes cosine similarity between the job vector and
   each resume vector.
3. `analyze_candidates()` combines both signals into a final weighted
   score, ranks candidates, prints a readable report, writes a CSV
   (`output/match_report.csv`), and renders a horizontal bar chart
   (`output/match_ranking.png`) sorted best-to-worst.

## Files

- `resume_matcher.py` — main script (CLI via `argparse`)
- `data/job_description.txt` — sample ML Engineer job posting
- `data/resumes/*.txt` — 4 sample candidate resumes of varying fit
- `requirements.txt` — pinned dependencies
