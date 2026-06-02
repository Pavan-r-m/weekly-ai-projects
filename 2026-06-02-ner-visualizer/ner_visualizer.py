"""
Named Entity Recognition (NER) Visualizer
==========================================
Extracts and visualizes named entities from text using spaCy.
Supports single-text analysis, batch file processing, and an interactive mode.

Entity types detected (spaCy en_core_web_sm):
  PERSON    - People, including fictional characters
  ORG       - Companies, agencies, institutions
  GPE       - Geopolitical entities (countries, cities, states)
  LOC       - Non-GPE locations (mountains, rivers, etc.)
  DATE      - Absolute or relative dates or periods
  MONEY     - Monetary values, including units
  PRODUCT   - Objects, vehicles, foods, etc.
  EVENT     - Named hurricanes, battles, wars, sports events
  LAW       - Named legal documents
  NORP      - Nationalities, religious or political groups

Fallback mode: if spaCy / its model is not installed, a regex-based heuristic
extractor kicks in automatically so the script always runs.
"""

import re
import sys
import json
import argparse
from collections import Counter, defaultdict
from typing import Optional

# ── Attempt spaCy import; graceful fallback ────────────────────────────────────
try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load("en_core_web_sm")
        MODEL_LOADED = True
    except OSError:
        MODEL_LOADED = False
except ImportError:
    SPACY_AVAILABLE = False
    MODEL_LOADED = False


# ── ANSI color map per entity type ────────────────────────────────────────────
COLORS = {
    "PERSON":  "\033[95m",   # magenta
    "ORG":     "\033[94m",   # blue
    "GPE":     "\033[92m",   # green
    "LOC":     "\033[96m",   # cyan
    "DATE":    "\033[93m",   # yellow
    "MONEY":   "\033[33m",   # dark yellow
    "PRODUCT": "\033[91m",   # red
    "EVENT":   "\033[35m",   # purple
    "LAW":     "\033[34m",   # dark blue
    "NORP":    "\033[32m",   # dark green
    "DEFAULT": "\033[37m",   # white
}
RESET = "\033[0m"
BOLD  = "\033[1m"


# ── Regex-based fallback NER ───────────────────────────────────────────────────
_REGEX_PATTERNS = [
    # Dates
    ("DATE", r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
             r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
             r"Dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?\b"),
    ("DATE", r"\b\d{4}\b"),
    ("DATE", r"\b(?:yesterday|today|tomorrow|last\s+(?:year|month|week)|"
             r"next\s+(?:year|month|week))\b"),
    # Money
    ("MONEY", r"\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|trillion))?"),
    ("MONEY", r"\b\d+(?:\.\d+)?\s*(?:dollars|euros|pounds|yen)\b"),
    # Organizations
    ("ORG", r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|Ltd|LLC|Co|"
            r"Foundation|Institute|University|College|Group|Association|Labs)\b"),
    ("ORG", r"\b[A-Z]{2,6}\b"),
    # GPE
    ("GPE", r"\b(?:United\s+States|USA|UK|China|India|Germany|France|Japan|"
            r"Canada|Australia|Brazil|Russia|Mexico|New\s+York|London|Paris|"
            r"Tokyo|Beijing|Berlin|Sydney|Toronto|Los\s+Angeles|Chicago|"
            r"Washington|San\s+Francisco|Seattle|Boston|Houston|Dallas|"
            r"California|Texas|Florida|New\s+York\s+City|Silicon\s+Valley|Poland|Warsaw)\b"),
    # Persons (title + name)
    ("PERSON", r"\b(?:Mr|Ms|Mrs|Dr|Prof|President|CEO|Director|Senator|"
               r"Representative|Chancellor)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"),
    # Persons (First Last)
    ("PERSON", r"\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b"),
]


def _regex_ner(text: str) -> list:
    """Fallback NER using hand-crafted regex patterns."""
    entities = []
    seen_spans = set()

    for label, pattern in _REGEX_PATTERNS:
        flags = re.IGNORECASE if label == "DATE" else 0
        for m in re.finditer(pattern, text, flags):
            span = (m.start(), m.end())
            # Skip overlapping spans
            if any(s[0] <= span[0] < s[1] or span[0] <= s[0] < span[1]
                   for s in seen_spans):
                continue
            seen_spans.add(span)
            entities.append({
                "text":  m.group().strip(),
                "label": label,
                "start": m.start(),
                "end":   m.end(),
            })

    entities.sort(key=lambda e: e["start"])
    return entities


def extract_entities(text: str) -> list:
    """Extract named entities. Uses spaCy if available, else regex fallback."""
    if MODEL_LOADED:
        doc = nlp(text)
        return [
            {
                "text":  ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end":   ent.end_char,
            }
            for ent in doc.ents
        ]
    return _regex_ner(text)


# ── Visualization helpers ──────────────────────────────────────────────────────

def colorize_entity(entity_text: str, label: str) -> str:
    """Wrap entity text in ANSI color + label tag."""
    color = COLORS.get(label, COLORS["DEFAULT"])
    return f"{color}{BOLD}[{entity_text}]{RESET}{color}({label}){RESET}"


def render_inline(text: str, entities: list) -> str:
    """Return text with entities highlighted inline (ANSI colors)."""
    result = text
    for ent in reversed(entities):   # reverse to preserve positions
        highlighted = colorize_entity(ent["text"], ent["label"])
        result = result[: ent["start"]] + highlighted + result[ent["end"]:]
    return result


def print_entity_table(entities: list) -> None:
    """Print a formatted table of discovered entities."""
    if not entities:
        print("  (no entities found)")
        return

    max_text  = max(max(len(e["text"])  for e in entities), 6)
    max_label = max(max(len(e["label"]) for e in entities), 5)
    header    = f"  {'Entity':<{max_text}}  {'Type':<{max_label}}  Position"
    print(BOLD + header + RESET)
    print("  " + "─" * (len(header)))

    for ent in entities:
        color = COLORS.get(ent["label"], COLORS["DEFAULT"])
        print(
            f"  {color}{ent['text']:<{max_text}}{RESET}"
            f"  {BOLD}{ent['label']:<{max_label}}{RESET}"
            f"  chars {ent['start']}–{ent['end']}"
        )


def print_summary(entities: list) -> None:
    """Print entity-type frequency summary and top entity per type."""
    if not entities:
        return

    counter = Counter(e["label"] for e in entities)

    print(f"\n{BOLD}Entity Type Breakdown:{RESET}")
    for label, count in counter.most_common():
        color = COLORS.get(label, COLORS["DEFAULT"])
        bar   = "█" * count
        print(f"  {color}{label:<10}{RESET}  {bar} ({count})")

    by_type = defaultdict(Counter)
    for ent in entities:
        by_type[ent["label"]][ent["text"]] += 1

    print(f"\n{BOLD}Most Frequent Entity Per Type:{RESET}")
    for label in counter:
        top_ent, top_cnt = by_type[label].most_common(1)[0]
        color = COLORS.get(label, COLORS["DEFAULT"])
        print(f"  {color}{label:<10}{RESET}  \"{top_ent}\" x{top_cnt}")


def analyze_text(text: str, label: Optional[str] = None, json_output: bool = False) -> dict:
    """
    Full pipeline: extract entities, display/return results.

    Args:
        text:        Input string to analyze.
        label:       Optional section label for display.
        json_output: If True, suppress pretty-print (caller handles JSON).

    Returns:
        dict with keys: input_length, entity_count, entities
    """
    entities = extract_entities(text)
    result   = {
        "label":        label or "",
        "input_length": len(text),
        "entity_count": len(entities),
        "entities":     entities,
    }

    if json_output:
        return result

    # ── Pretty print ──────────────────────────────────────────────────────
    if label:
        print(f"\n{BOLD}{'─'*62}{RESET}")
        print(f"{BOLD}  {label}{RESET}")

    print(f"\n{BOLD}Highlighted Text:{RESET}")
    rendered = render_inline(text, entities)
    # Indent and word-wrap for readability
    for line in rendered.split("\n"):
        print("  " + line)

    print(f"\n{BOLD}Entities Found ({len(entities)}):{RESET}")
    print_entity_table(entities)
    print_summary(entities)
    return result


# ── Sample texts for --demo mode ──────────────────────────────────────────────
SAMPLE_TEXTS = [
    (
        "Tech giant Apple Inc. announced on June 1, 2026 that CEO Tim Cook "
        "will visit Berlin and Tokyo next week to meet with European and Japanese "
        "regulators. The company's market cap crossed $3.5 trillion in May 2026.",
        "Tech News"
    ),
    (
        "Dr. Marie Curie was born in Warsaw, Poland in 1867. She later moved to Paris "
        "where she conducted groundbreaking research at the University of Paris. "
        "She won the Nobel Prize in 1903 and again in 1911, becoming the first person "
        "to win Nobel Prizes in two different sciences.",
        "Historical Biography"
    ),
    (
        "The United Nations climate summit scheduled for next December in New York "
        "will bring together leaders from China, India, the USA, and Germany. "
        "The European Union has pledged $50 billion toward green energy initiatives. "
        "Senator Jane Williams introduced the Clean Future Act last Tuesday.",
        "Policy & Politics"
    ),
]


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Named Entity Recognition Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ner_visualizer.py --demo
  python ner_visualizer.py --text "Elon Musk founded SpaceX in 2002 in California."
  python ner_visualizer.py --file my_texts.txt
  python ner_visualizer.py --text "Apple is based in Cupertino." --json
        """,
    )
    parser.add_argument("--text",  help="Text string to analyze")
    parser.add_argument("--file",  help="Path to a .txt file (one document per line)")
    parser.add_argument("--demo",  action="store_true", help="Run on built-in sample texts")
    parser.add_argument("--json",  action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    # ── Backend banner ─────────────────────────────────────────────────────
    if not args.json:
        print(f"\n{BOLD}NER Visualizer  —  Named Entity Recognition{RESET}")
        if MODEL_LOADED:
            print("  Backend: spaCy en_core_web_sm  ✓")
        elif SPACY_AVAILABLE:
            print("  Backend: regex heuristics  (spaCy installed but model missing)")
            print("  Tip: python -m spacy download en_core_web_sm  for better results")
        else:
            print("  Backend: regex heuristics  (spaCy not installed)")
            print("  Tip: pip install spacy && python -m spacy download en_core_web_sm")

    # ── Dispatch ───────────────────────────────────────────────────────────
    if args.demo:
        all_results = [analyze_text(t, label=l, json_output=args.json)
                       for t, l in SAMPLE_TEXTS]
        if args.json:
            print(json.dumps(all_results, indent=2))

    elif args.text:
        r = analyze_text(args.text, json_output=args.json)
        if args.json:
            print(json.dumps(r, indent=2))

    elif args.file:
        try:
            with open(args.file, encoding="utf-8") as fh:
                lines = [ln.rstrip() for ln in fh if ln.strip()]
        except FileNotFoundError:
            print(f"Error: file not found: {args.file}", file=sys.stderr)
            sys.exit(1)

        all_results = [analyze_text(ln, label=f"Line {i}", json_output=args.json)
                       for i, ln in enumerate(lines, 1)]
        if args.json:
            print(json.dumps(all_results, indent=2))
        else:
            total = sum(r["entity_count"] for r in all_results)
            print(f"\n{BOLD}{'='*62}{RESET}")
            print(f"{BOLD}Batch Summary:{RESET} {len(lines)} texts, {total} total entities")

    else:
        # Interactive mode
        if not args.json:
            print("\nNo input provided. Entering interactive mode (Ctrl+C to quit).\n")
        try:
            while True:
                text = input("Enter text: ").strip()
                if text:
                    r = analyze_text(text, json_output=args.json)
                    if args.json:
                        print(json.dumps(r, indent=2))
                    print()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")


if __name__ == "__main__":
    main()
