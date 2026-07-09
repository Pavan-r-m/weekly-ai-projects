"""
agents.py -- Agent definitions for the Multi-Agent Debate Simulator.

Implements three cooperating/competing AI agents that argue a topic:
    - ProposerAgent : argues FOR the motion
    - CriticAgent   : argues AGAINST the motion
    - JudgeAgent    : impartially scores each round and declares a winner

Each agent wraps an LLM "brain". By default this is a deterministic,
rule-based MockLLM so the whole simulator runs offline with zero
dependencies and zero API cost. If OPENAI_API_KEY or ANTHROPIC_API_KEY
is set in your environment, get_llm_backend() shows exactly where to
plug in a real model while keeping the same interface.
"""

import os
import random
import re
from collections import Counter
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# LLM Backends
# ---------------------------------------------------------------------------

class MockLLM:
    """
    A deterministic, offline stand-in for a real language model.

    It doesn't "understand" the topic the way a real LLM would, but it
    behaves like one at the interface level: given a role, a topic, and
    conversation history, it produces a plausible, on-topic argument by
    combining rhetorical templates with keyword extraction from the
    opponent's last statement -- so replies read like genuine rebuttals
    rather than canned, unrelated text.
    """

    OPENING_TEMPLATES = [
        "Consider that {point}.",
        "The evidence strongly suggests {point}.",
        "It's worth emphasizing that {point}.",
        "From a practical standpoint, {point}.",
        "History and data both point to one conclusion: {point}.",
    ]

    REBUTTAL_TEMPLATES = [
        "My opponent claims '{claim}', but this overlooks {counter}.",
        "While it's true that '{claim}', the stronger consideration is {counter}.",
        "The argument that '{claim}' doesn't hold up once we account for {counter}.",
        "'{claim}' sounds compelling, yet {counter} tells a different story.",
    ]

    CLOSING_TEMPLATES = [
        "In summary, {point}, and that is why this position holds.",
        "Taken together, these points show that {point}.",
        "The case rests on this: {point}.",
    ]

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def _extract_keywords(self, text: str, k: int = 4) -> List[str]:
        """Tiny keyword extractor: drop stopwords, keep the most frequent terms."""
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "of", "to", "in",
            "on", "for", "and", "or", "but", "that", "this", "it", "its",
            "as", "at", "by", "with", "be", "have", "has", "not", "no",
            "so", "if", "than", "then", "we", "our", "their", "they",
        }
        words = re.findall(r"[a-zA-Z']+", text.lower())
        words = [w for w in words if w not in stopwords and len(w) > 3]
        common = [w for w, _ in Counter(words).most_common(k)]
        return common

    def generate_opening(self, topic: str, stance: str, argument_bank: List[str]) -> str:
        point = self.rng.choice(argument_bank)
        template = self.rng.choice(self.OPENING_TEMPLATES)
        return template.format(point=point)

    def _clean_claim_snippet(self, text: str, max_words: int = 14) -> str:
        """
        Produce a short, quote-safe paraphrase snippet of the opponent's
        last statement. Strips any rebuttal scaffolding from a *previous*
        round (so quotes don't nest deeper and deeper turn after turn)
        and truncates to a fixed word count.
        """
        stripped = text.strip()
        # Peel off nested rebuttal framing like "My opponent claims '...'"
        # so we always quote the underlying substantive claim, not a
        # quote-of-a-quote-of-a-quote.
        match = re.search(r"'([^']+)'", stripped)
        if match and len(match.group(1)) > 15:
            stripped = match.group(1)
        quote_chars = chr(39) + chr(34) + " "
        stripped = stripped.strip(quote_chars)
        words = stripped.split()
        snippet = " ".join(words[:max_words])
        if len(words) > max_words:
            snippet += "..."
        return snippet

    def generate_rebuttal(self, topic: str, stance: str, opponent_text: str,
                           argument_bank: List[str]) -> str:
        keywords = self._extract_keywords(opponent_text)
        claim_snippet = self._clean_claim_snippet(opponent_text)
        counter_point = self.rng.choice(argument_bank)
        template = self.rng.choice(self.REBUTTAL_TEMPLATES)
        counter = counter_point
        if keywords:
            counter = f"{counter_point} (especially regarding {', '.join(keywords[:2])})"
        return template.format(claim=claim_snippet, counter=counter)

    def generate_closing(self, topic: str, stance: str, argument_bank: List[str]) -> str:
        point = self.rng.choice(argument_bank)
        template = self.rng.choice(self.CLOSING_TEMPLATES)
        return template.format(point=point)

    def score_round(self, proposer_text: str, critic_text: str) -> Dict[str, float]:
        """
        Heuristic scoring: rewards specificity (length within reason),
        rebuttal quality (keyword overlap with opponent's prior turn),
        and lexical diversity (avoids repetition).
        """
        def score_one(text: str, opponent_text: str) -> float:
            words = re.findall(r"[a-zA-Z']+", text.lower())
            if not words:
                return 0.0
            diversity = len(set(words)) / len(words)
            length_score = min(len(words) / 25.0, 1.0)
            opp_words = set(re.findall(r"[a-zA-Z']+", opponent_text.lower()))
            overlap = len(set(words) & opp_words) / max(len(opp_words), 1)
            rebuttal_score = min(overlap * 2, 1.0)
            return round((diversity * 0.3 + length_score * 0.4 + rebuttal_score * 0.3) * 10, 2)

        return {
            "proposer": score_one(proposer_text, critic_text),
            "critic": score_one(critic_text, proposer_text),
        }


def get_llm_backend(seed: Optional[int] = None):
    """
    Returns an LLM backend object. Defaults to MockLLM (no API key needed).
    If a real API key is present, this function is the extension point for
    wiring up a genuine model call while keeping the exact same interface
    (generate_opening / generate_rebuttal / generate_closing / score_round).
    """
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"):
        print("[info] API key detected, but this demo ships with MockLLM only. "
              "See README for how to wire up a real LLM backend.")
    return MockLLM(seed=seed)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class DebateAgent:
    """Base class for a debating agent with a persistent role and memory."""

    def __init__(self, name: str, stance: str, argument_bank: List[str], llm: MockLLM):
        self.name = name
        self.stance = stance          # "for" or "against"
        self.argument_bank = argument_bank
        self.llm = llm
        self.memory: List[str] = []   # everything this agent has said

    def open(self, topic: str) -> str:
        text = self.llm.generate_opening(topic, self.stance, self.argument_bank)
        self.memory.append(text)
        return text

    def rebut(self, topic: str, opponent_text: str) -> str:
        text = self.llm.generate_rebuttal(topic, self.stance, opponent_text, self.argument_bank)
        self.memory.append(text)
        return text

    def close(self, topic: str) -> str:
        text = self.llm.generate_closing(topic, self.stance, self.argument_bank)
        self.memory.append(text)
        return text


class ProposerAgent(DebateAgent):
    def __init__(self, argument_bank: List[str], llm: MockLLM):
        super().__init__("Proposer", "for", argument_bank, llm)


class CriticAgent(DebateAgent):
    def __init__(self, argument_bank: List[str], llm: MockLLM):
        super().__init__("Critic", "against", argument_bank, llm)


class JudgeAgent:
    """Impartial agent that scores each round and tallies an overall winner."""

    def __init__(self, llm: MockLLM):
        self.llm = llm
        self.round_scores: List[Dict[str, float]] = []

    def score(self, proposer_text: str, critic_text: str) -> Dict[str, float]:
        scores = self.llm.score_round(proposer_text, critic_text)
        self.round_scores.append(scores)
        return scores

    def verdict(self) -> Dict[str, object]:
        total_proposer = sum(r["proposer"] for r in self.round_scores)
        total_critic = sum(r["critic"] for r in self.round_scores)
        if total_proposer > total_critic:
            winner = "Proposer"
        elif total_critic > total_proposer:
            winner = "Critic"
        else:
            winner = "Tie"
        return {
            "winner": winner,
            "proposer_total": round(total_proposer, 2),
            "critic_total": round(total_critic, 2),
            "rounds_judged": len(self.round_scores),
        }
