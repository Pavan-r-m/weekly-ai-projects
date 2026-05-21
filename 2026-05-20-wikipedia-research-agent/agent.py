#!/usr/bin/env python3
"""
Research Agent — A multi-step AI agent that autonomously researches topics
using Wikipedia as a knowledge source.

The agent:
  1. Searches Wikipedia for the initial query
  2. Fetches page summaries and extracts key facts
  3. Identifies the most relevant related sub-topics
  4. Iteratively follows those links (breadth-first, configurable depth)
  5. Synthesizes a structured, markdown-formatted research report

No LLM API key required — demonstrates core agentic patterns (tool use,
memory, iterative planning, synthesis) using only the free Wikipedia REST API.

Optional: Set OPENAI_API_KEY in your environment and use --llm flag to have
GPT-4o-mini write the final synthesis instead.

Usage:
    python agent.py "quantum computing"
    python agent.py "climate change" --depth 2 --pages 8
    python agent.py "CRISPR gene editing" --output report.md
"""

import argparse
import os
import re
import sys
import textwrap
import time
from collections import deque
from datetime import datetime
from typing import Optional

import requests

# ── Wikipedia Tool Functions ──────────────────────────────────────────────────


def search_wikipedia(query: str, limit: int = 5) -> list:
    """
    Search Wikipedia for pages matching the query.
    Returns a list of dicts with 'title' and 'snippet'.
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json",
        "utf8": 1,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("query", {}).get("search", [])
        return [
            {"title": r["title"], "snippet": re.sub(r"<[^>]+>", "", r["snippet"])}
            for r in results
        ]
    except Exception as e:
        print(f"  [search error] {e}", file=sys.stderr)
        return []


def fetch_page_summary(title: str) -> Optional[dict]:
    """
    Fetch a Wikipedia page's REST summary (intro extract).
    Returns dict with: title, extract, url, description.
    """
    encoded = requests.utils.quote(title.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
    try:
        resp = requests.get(
            url, timeout=10, headers={"User-Agent": "ResearchAgent/1.0 (educational project)"}
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        return {
            "title": data.get("title", title),
            "extract": data.get("extract", ""),
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "description": data.get("description", ""),
        }
    except Exception as e:
        print(f"  [fetch error for '{title}'] {e}", file=sys.stderr)
        return None


def fetch_related_links(title: str, limit: int = 12) -> list:
    """
    Fetch internal Wikipedia links from a page (related topics to explore).
    Returns a list of related page titles.
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "links",
        "pllimit": limit,
        "plnamespace": 0,   # main article namespace only
        "format": "json",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        links = []
        for page in pages.values():
            for link in page.get("links", []):
                links.append(link["title"])
        return links[:limit]
    except Exception as e:
        print(f"  [links error for '{title}'] {e}", file=sys.stderr)
        return []


# ── Text Processing Utilities ─────────────────────────────────────────────────


def extract_key_sentences(text: str, n: int = 4) -> list:
    """
    Extract the most information-dense sentences using simple heuristics:
      - Prefer sentences with named entities (capitalized words)
      - Prefer medium-length sentences (not too short, not too long)
      - Score by capital-word density and length
    """
    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    # Filter out very short fragments
    sentences = [s.strip() for s in sentences if len(s.split()) >= 8]

    def score(s: str) -> float:
        words = s.split()
        if not words:
            return 0.0
        # Ratio of capitalized words (likely named entities / important terms)
        cap_ratio = sum(1 for w in words if len(w) > 1 and w[0].isupper()) / len(words)
        # Length score: reward sentences of 15-40 words
        length_score = min(len(words) / 35.0, 1.0)
        # Small penalty for very long sentences (over 60 words)
        penalty = max(0.0, (len(words) - 60) * 0.01)
        return cap_ratio * 0.35 + length_score * 0.65 - penalty

    ranked = sorted(sentences, key=score, reverse=True)
    return ranked[:n]


def score_relevance(candidate_title: str, query_terms: set) -> float:
    """
    Score how relevant a candidate Wikipedia title is to the original query.
    Uses simple term-overlap heuristic.
    """
    title_terms = set(re.sub(r"[^\w\s]", "", candidate_title).lower().split())
    overlap = len(title_terms & query_terms)
    return overlap / max(len(query_terms), 1)


# ── Agent Memory ──────────────────────────────────────────────────────────────


class ResearchMemory:
    """Stores all pages the agent has fetched and facts it has extracted."""

    def __init__(self):
        self.visited = set()              # page titles already fetched
        self.pages = []                   # full page data dicts
        self.key_facts = {}               # title -> list[str] key sentences

    def record(self, page: dict, facts: list):
        """Store a page and its extracted key facts."""
        self.visited.add(page["title"])
        self.pages.append(page)
        self.key_facts[page["title"]] = facts

    def already_visited(self, title: str) -> bool:
        return title in self.visited

    def summary(self) -> str:
        total_facts = sum(len(f) for f in self.key_facts.values())
        return f"{len(self.pages)} pages | {total_facts} key facts"


# ── Research Agent ────────────────────────────────────────────────────────────


class ResearchAgent:
    """
    Multi-step research agent implementing a simple agentic loop:

      PLAN  →  Search Wikipedia for starting points
      ACT   →  Fetch page content and extract key facts
      OBSERVE → Identify related sub-topics via internal links
      REPEAT → Continue until budget (max_pages) exhausted
      SYNTHESIZE → Compile all findings into a structured report

    This pattern mirrors how LLM agents with tool use work, but replaces
    the LLM reasoning step with deterministic heuristics — making it
    runnable offline without any API key.
    """

    def __init__(self, max_pages: int = 6, max_depth: int = 2, verbose: bool = True):
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.verbose = verbose
        self.memory = ResearchMemory()

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def run(self, query: str) -> str:
        """
        Main entry point: research the given query, return a markdown report.
        """
        self._log(f"\n🔍 ResearchAgent starting: '{query}'\n{'─'*55}")
        # Build query terms for relevance scoring
        query_terms = set(re.sub(r"[^\w\s]", "", query).lower().split())

        # ── Step 1: Search for best starting pages ────────────────────────
        self._log("📡 [PLAN] Searching Wikipedia for entry points...")
        results = search_wikipedia(query, limit=5)
        if not results:
            return f"# Research Report: {query}\n\nNo Wikipedia results found."

        self._log(f"   Candidates: {[r['title'] for r in results]}")

        # ── Step 2: Breadth-first exploration loop ────────────────────────
        # Seed queue with top-2 search results at depth 0
        queue = deque()
        for r in results[:2]:
            queue.append((r["title"], 0))

        pages_fetched = 0

        while queue and pages_fetched < self.max_pages:
            title, depth = queue.popleft()

            # Skip if already processed
            if self.memory.already_visited(title):
                continue

            self._log(f"\n📖 [ACT] Fetching '{title}' (depth={depth})")
            time.sleep(0.35)    # polite rate limiting

            # Fetch page summary from Wikipedia REST API
            page = fetch_page_summary(title)
            if not page or not page["extract"]:
                self._log("   ⚠️  No content, skipping.")
                self.memory.visited.add(title)  # mark as visited to avoid retry
                continue

            # Extract the most information-dense sentences
            facts = extract_key_sentences(page["extract"], n=4)
            self.memory.record(page, facts)
            pages_fetched += 1
            self._log(f"   ✅ Stored {len(facts)} facts | {self.memory.summary()}")

            # ── Step 3: Discover related topics ──────────────────────────
            if depth < self.max_depth:
                self._log("   🔗 [OBSERVE] Discovering related topics...")
                related = fetch_related_links(title, limit=15)
                # Score and pick the most relevant unvisited links
                scored = [
                    (t, score_relevance(t, query_terms))
                    for t in related
                    if not self.memory.already_visited(t)
                ]
                scored.sort(key=lambda x: x[1], reverse=True)
                top_related = [t for t, _ in scored[:3]]
                self._log(f"   → Queuing: {top_related}")
                for rel_title in top_related:
                    queue.append((rel_title, depth + 1))

        # ── Step 4: Synthesize the report ──────────────────────────────────
        self._log(f"\n✍️  [SYNTHESIZE] Compiling report ({self.memory.summary()})...")
        return self._synthesize_report(query)

    def _synthesize_report(self, query: str) -> str:
        """
        Compile all memory into a structured, markdown-formatted report.
        Each researched page becomes its own section with bullet-point facts.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            f"# Research Report: {query}",
            f"",
            f"*Generated by ResearchAgent · {timestamp}*  ",
            f"*Pages researched: {len(self.memory.pages)}*",
            "",
            "---",
            "",
            "## Overview",
            "",
        ]

        # Use the first page's full extract as the overview section
        if self.memory.pages:
            first = self.memory.pages[0]
            overview = first["extract"]
            # Trim to ~600 chars for readability
            if len(overview) > 600:
                cutoff = overview.rfind(". ", 0, 600)
                overview = overview[: cutoff + 1] if cutoff > 0 else overview[:600] + "..."
            lines.append(overview)
            lines.append("")

        # One section per researched page
        lines += ["---", "", "## Key Findings by Topic", ""]
        for page in self.memory.pages:
            title = page["title"]
            desc = page.get("description", "")
            url = page.get("url", "")
            facts = self.memory.key_facts.get(title, [])

            lines.append(f"### {title}")
            if desc:
                lines.append(f"*{desc}*")
            lines.append("")
            for fact in facts:
                # Wrap long sentences neatly
                wrapped = textwrap.fill(fact, width=92)
                lines.append(f"- {wrapped}")
            lines.append("")
            if url:
                lines.append(f"🔗 [Read more on Wikipedia]({url})")
            lines.append("")

        # Agent trace at the bottom
        topics = ", ".join(p["title"] for p in self.memory.pages)
        total_facts = sum(len(f) for f in self.memory.key_facts.values())
        lines += [
            "---",
            "",
            "## Agent Execution Trace",
            "",
            f"| Parameter      | Value |",
            f"|----------------|-------|",
            f"| Query          | {query} |",
            f"| Pages fetched  | {len(self.memory.pages)} |",
            f"| Facts extracted| {total_facts} |",
            f"| Topics covered | {topics} |",
            "",
            "> *Report compiled autonomously by ResearchAgent using the Wikipedia REST API.*  ",
            "> *To enhance with GPT-4o-mini synthesis: set `OPENAI_API_KEY` and run with `--llm`.*",
        ]

        return "\n".join(lines)


# ── Optional LLM Synthesis ────────────────────────────────────────────────────


def llm_synthesize(report: str, query: str) -> str:
    """
    Optional enhancement: use OpenAI GPT-4o-mini to rewrite the synthesized
    report into a polished narrative. Requires OPENAI_API_KEY in environment.

    Falls back gracefully to the original report if the key is missing.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set — skipping LLM synthesis.", file=sys.stderr)
        return report

    try:
        import openai  # type: ignore
        client = openai.OpenAI(api_key=api_key)
        prompt = (
            f"You are a research analyst. Rewrite the following structured research notes "
            f"about '{query}' into a clear, engaging 3-paragraph narrative summary. "
            f"Preserve all factual information. Use flowing prose, no bullet points.\n\n"
            f"--- RESEARCH NOTES ---\n{report}"
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=900,
        )
        narrative = response.choices[0].message.content.strip()
        return (
            f"# Research Summary: {query}\n\n"
            f"*GPT-4o-mini narrative synthesis*\n\n"
            f"{narrative}\n\n"
            f"---\n\n"
            f"## Raw Research Notes\n\n{report}"
        )
    except Exception as e:
        print(f"⚠️  LLM synthesis failed: {e}", file=sys.stderr)
        return report


# ── CLI Entry Point ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="ResearchAgent — autonomous multi-step Wikipedia research tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python agent.py "quantum computing"
          python agent.py "CRISPR gene editing" --depth 2 --pages 8
          python agent.py "climate change" --output report.md
          python agent.py "black holes" --llm --output report.md
          python agent.py "machine learning" --quiet --output ml_report.md
        """),
    )
    parser.add_argument("query", help="Research topic or question")
    parser.add_argument(
        "--depth", type=int, default=1,
        help="Max link-follow depth (default: 1, higher = broader research)"
    )
    parser.add_argument(
        "--pages", type=int, default=5,
        help="Max Wikipedia pages to fetch (default: 5)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save the report to this Markdown file (default: print to stdout)"
    )
    parser.add_argument(
        "--llm", action="store_true",
        help="Use OpenAI GPT-4o-mini to write a narrative synthesis (requires OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress agent trace output (only print final report)"
    )
    args = parser.parse_args()

    # Run the agent
    agent = ResearchAgent(
        max_pages=args.pages,
        max_depth=args.depth,
        verbose=not args.quiet,
    )
    report = agent.run(args.query)

    # Optionally enhance with LLM
    if args.llm:
        print("\n🤖 Enhancing report with GPT-4o-mini...")
        report = llm_synthesize(report, args.query)

    # Output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n📄 Report saved to: {args.output}")
    else:
        print("\n" + "═" * 60)
        print(report)


if __name__ == "__main__":
    main()
