"""
Resume-to-Job Semantic Matcher & Skill Gap Analyzer
=====================================================

What it does
------------
Given a job description and a folder of candidate resumes (plain text),
this tool:
  1. Computes a semantic similarity score between each resume and the job
     description using TF-IDF vectorization + cosine similarity.
  2. Extracts recognized skills from a curated taxonomy for both the job
     description and each resume.
  3. Reports which required skills each candidate HAS and which they're
     MISSING, alongside an overall match score.
  4. Ranks all candidates from best to worst fit and saves a bar chart.

Why it's interesting
---------------------
Resume screening is one of the most common real-world NLP applications.
Rather than relying on exact keyword matches (which miss synonyms and
phrasing differences), this tool uses TF-IDF weighted term vectors and
cosine similarity to capture overall textual relevance, while a separate
rule-based skill extractor gives recruiters an explainable breakdown of
*why* a candidate scored the way they did. Combining a statistical
similarity score with explainable keyword extraction is a pattern used in
real applicant-tracking systems (ATS).

No API keys or heavyweight model downloads required — everything runs
with scikit-learn, so it works fully offline.
"""

import argparse
import os
import re
import sys
import csv
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Skill taxonomy — a curated list of common tech/soft skills to look for.
#    Using word-boundary regex matching so "Python" matches "python," etc.
#    Multi-word skills (e.g. "machine learning") are matched as phrases.
# ---------------------------------------------------------------------------
SKILL_TAXONOMY = [
    # Programming languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "sql", "r", "scala", "kotlin", "swift",
    # Data / ML
    "machine learning", "deep learning", "nlp", "natural language processing",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
    "data analysis", "data visualization", "statistics", "computer vision",
    "matplotlib", "seaborn", "power bi", "tableau", "etl",
    # Web / backend
    "react", "node.js", "django", "flask", "fastapi", "rest api",
    "graphql", "html", "css", "spring boot",
    # Cloud / infra
    "aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "terraform",
    "linux", "git", "microservices",
    # Databases
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
    # Soft / process skills
    "agile", "scrum", "project management", "communication", "leadership",
    "problem solving", "team collaboration", "public speaking",
]


def extract_skills(text: str, taxonomy=SKILL_TAXONOMY) -> set:
    """Return the subset of skills from `taxonomy` found in `text`.

    Uses case-insensitive whole-word/phrase matching so short tokens like
    "r" or "go" don't spuriously match inside other words.
    """
    text_lower = text.lower()
    found = set()
    for skill in taxonomy:
        # Escape special regex chars (e.g. "c++", "c#") and require
        # word boundaries around the skill phrase.
        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
        if re.search(pattern, text_lower):
            found.add(skill)
    return found


def load_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def compute_similarity_scores(job_text: str, resume_texts: list) -> np.ndarray:
    """Fit TF-IDF over [job_text] + all resumes, then compute cosine
    similarity of each resume vector against the job vector.
    """
    corpus = [job_text] + resume_texts
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),   # capture short phrases like "machine learning"
        max_df=0.95,
        min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    job_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]
    similarities = cosine_similarity(job_vector, resume_vectors).flatten()
    return similarities


def analyze_candidates(job_path: Path, resumes_dir: Path, output_dir: Path):
    job_text = load_text(job_path)
    job_skills = extract_skills(job_text)

    resume_files = sorted(
        [p for p in resumes_dir.iterdir() if p.suffix == ".txt"]
    )
    if not resume_files:
        print(f"No .txt resumes found in {resumes_dir}")
        sys.exit(1)

    resume_texts = [load_text(p) for p in resume_files]
    similarities = compute_similarity_scores(job_text, resume_texts)

    results = []
    for path, text, sim in zip(resume_files, resume_texts, similarities):
        candidate_skills = extract_skills(text)
        matched = job_skills & candidate_skills
        missing = job_skills - candidate_skills
        skill_coverage = len(matched) / len(job_skills) if job_skills else 0.0
        # Blend TF-IDF similarity (70%) with explicit skill coverage (30%)
        # for a final score that rewards both textual relevance and having
        # the literal required skills.
        final_score = 0.7 * sim + 0.3 * skill_coverage
        results.append({
            "candidate": path.stem,
            "similarity": sim,
            "skill_coverage": skill_coverage,
            "final_score": final_score,
            "matched_skills": sorted(matched),
            "missing_skills": sorted(missing),
        })

    results.sort(key=lambda r: r["final_score"], reverse=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Print a readable report to the console ---
    print("=" * 70)
    print(f"JOB REQUIREMENTS DETECTED ({len(job_skills)} skills):")
    print("  " + ", ".join(sorted(job_skills)))
    print("=" * 70)
    for rank, r in enumerate(results, start=1):
        print(f"\n#{rank}  {r['candidate']}")
        print(f"    Final match score : {r['final_score']*100:.1f}%")
        print(f"    Text similarity    : {r['similarity']*100:.1f}%")
        print(f"    Skill coverage     : {r['skill_coverage']*100:.1f}%")
        print(f"    Matched skills     : {', '.join(r['matched_skills']) or '(none)'}")
        print(f"    Missing skills     : {', '.join(r['missing_skills']) or '(none)'}")

    # --- Save CSV report ---
    csv_path = output_dir / "match_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "candidate", "final_score_pct", "similarity_pct",
            "skill_coverage_pct", "matched_skills", "missing_skills",
        ])
        for rank, r in enumerate(results, start=1):
            writer.writerow([
                rank,
                r["candidate"],
                round(r["final_score"] * 100, 1),
                round(r["similarity"] * 100, 1),
                round(r["skill_coverage"] * 100, 1),
                "; ".join(r["matched_skills"]),
                "; ".join(r["missing_skills"]),
            ])
    print(f"\nSaved CSV report to {csv_path}")

    # --- Save bar chart of final scores ---
    fig, ax = plt.subplots(figsize=(9, 5))
    names = [r["candidate"] for r in results]
    scores = [r["final_score"] * 100 for r in results]
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(names)))
    bars = ax.barh(names, scores, color=colors)
    ax.set_xlabel("Final Match Score (%)")
    ax.set_title("Resume-to-Job Match Ranking")
    ax.invert_yaxis()  # best candidate on top
    ax.set_xlim(0, 100)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"{score:.1f}%", va="center", fontsize=9)
    plt.tight_layout()
    chart_path = output_dir / "match_ranking.png"
    plt.savefig(chart_path, dpi=150)
    print(f"Saved ranking chart to {chart_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Match resumes against a job description using TF-IDF "
                    "similarity + skill-gap analysis."
    )
    here = Path(__file__).parent
    parser.add_argument(
        "--job", type=str, default=str(here / "data" / "job_description.txt"),
        help="Path to job description .txt file",
    )
    parser.add_argument(
        "--resumes", type=str, default=str(here / "data" / "resumes"),
        help="Path to folder containing candidate resume .txt files",
    )
    parser.add_argument(
        "--output", type=str, default=str(here / "output"),
        help="Directory to write the CSV report and chart",
    )
    args = parser.parse_args()

    analyze_candidates(Path(args.job), Path(args.resumes), Path(args.output))


if __name__ == "__main__":
    main()
