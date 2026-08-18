"""
PII Text Redactor & Anonymizer
================================
A self-contained NLP tool that scans free-form text for personally
identifiable information (PII), redacts it, and produces a machine-readable
audit report with a weighted privacy "risk score".

No external NER model downloads are required: structured PII (emails,
phone numbers, SSNs, credit cards, IP addresses, URLs, dates of birth,
street addresses) is caught with carefully-tuned regular expressions, and
unstructured PII (person names) is caught with a lightweight statistical
heuristic that combines a bundled common-name gazetteer with contextual
honorific/pattern cues ("Mr. Smith", "My name is ...", "contact John Doe").

This mirrors how many production PII scrubbers (e.g. AWS Comprehend's
regex-based detectors, Microsoft Presidio's pattern recognizers) start
before layering on heavier transformer-based NER.
"""

import argparse
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict


# ---------------------------------------------------------------------------
# 1. Gazetteers used by the lightweight name detector
# ---------------------------------------------------------------------------

COMMON_FIRST_NAMES = {
    "james", "mary", "john", "patricia", "robert", "jennifer", "michael",
    "linda", "william", "elizabeth", "david", "barbara", "richard", "susan",
    "joseph", "jessica", "thomas", "sarah", "charles", "karen", "christopher",
    "nancy", "daniel", "lisa", "matthew", "betty", "anthony", "margaret",
    "mark", "sandra", "donald", "ashley", "steven", "kimberly", "andrew",
    "emily", "joshua", "donna", "kevin", "michelle", "brian", "priya",
    "raj", "wei", "hiroshi", "fatima", "amir", "carlos", "sofia", "elena",
    "noah", "olivia", "liam", "emma", "ava", "sophia", "mia", "isabella",
}

COMMON_LAST_NAMES = {
    "smith", "johnson", "williams", "brown", "jones", "garcia", "miller",
    "davis", "rodriguez", "martinez", "hernandez", "lopez", "gonzalez",
    "wilson", "anderson", "thomas", "taylor", "moore", "jackson", "martin",
    "lee", "perez", "thompson", "white", "harris", "sanchez", "clark",
    "ramirez", "lewis", "robinson", "walker", "young", "allen", "king",
    "wright", "scott", "torres", "nguyen", "hill", "flores", "green",
    "patel", "khan", "chen", "kim", "singh", "kumar", "sato", "muller",
    "doe", "carter", "mitchell", "roberts", "campbell", "parker", "evans",
}

HONORIFICS = r"(?:Mr|Mrs|Ms|Miss|Dr|Prof|Sir|Madam)\."
NAME_CONTEXT_CUES = [
    r"my name is",
    r"i am",
    r"i'm",
    r"this is",
    r"contact",
    r"attn:?",
    r"attention:?",
    r"regards,",
    r"sincerely,",
    r"signed,",
    r"reach out to",
    r"speak with",
    r"spoke with",
    r"thanks,",
    r"thank you,",
    r"best,",
    r"cheers,",
]

# ---------------------------------------------------------------------------
# 2. Regex library for structured PII
#    (ordering matters: more specific patterns are checked before generic
#     ones so e.g. a credit card number isn't also flagged as a phone number)
# ---------------------------------------------------------------------------

PATTERNS: Dict[str, "re.Pattern"] = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "IP_ADDRESS": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
    ),
    "PHONE": re.compile(
        r"(?:(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4})"
    ),
    "URL": re.compile(r"\bhttps?://[^\s]+"),
    "DATE_OF_BIRTH": re.compile(
        r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b"
    ),
    "STREET_ADDRESS": re.compile(
        r"\b\d{1,5}\s+([A-Z][a-z]+\s){1,3}"
        r"(Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Lane|Ln|Drive|Dr|Court|Ct|"
        r"Way|Place|Pl|Terrace|Ter|Parkway|Pkwy|Circle|Cir|Highway|Hwy|Trail)\b\.?",
    ),
}

# Sensitivity weights feed the overall risk score (higher = more sensitive).
SENSITIVITY_WEIGHTS = {
    "SSN": 10,
    "CREDIT_CARD": 9,
    "DATE_OF_BIRTH": 6,
    "STREET_ADDRESS": 5,
    "PERSON_NAME": 4,
    "PHONE": 4,
    "EMAIL": 3,
    "IP_ADDRESS": 2,
    "URL": 1,
}

PLACEHOLDER = {
    "EMAIL": "[REDACTED_EMAIL]",
    "SSN": "[REDACTED_SSN]",
    "CREDIT_CARD": "[REDACTED_CARD]",
    "IP_ADDRESS": "[REDACTED_IP]",
    "PHONE": "[REDACTED_PHONE]",
    "URL": "[REDACTED_URL]",
    "DATE_OF_BIRTH": "[REDACTED_DOB]",
    "STREET_ADDRESS": "[REDACTED_ADDRESS]",
    "PERSON_NAME": "[REDACTED_NAME]",
}


@dataclass
class Finding:
    entity_type: str
    text: str
    start: int
    end: int
    confidence: float


@dataclass
class RedactionReport:
    findings: List[Finding] = field(default_factory=list)

    def counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for f in self.findings:
            counts[f.entity_type] = counts.get(f.entity_type, 0) + 1
        return counts

    def risk_score(self) -> float:
        """Weighted, saturating risk score in [0, 100]."""
        raw = sum(SENSITIVITY_WEIGHTS.get(f.entity_type, 1) for f in self.findings)
        # Saturating curve so a handful of high-severity hits already
        # pushes the score into "Critical" without requiring dozens of hits.
        score = 100 * (1 - pow(2.71828, -raw / 20))
        return round(score, 1)

    def risk_level(self) -> str:
        score = self.risk_score()
        if score >= 75:
            return "CRITICAL"
        if score >= 45:
            return "HIGH"
        if score >= 15:
            return "MEDIUM"
        if score > 0:
            return "LOW"
        return "NONE"

    def to_dict(self) -> dict:
        return {
            "total_findings": len(self.findings),
            "counts_by_type": self.counts(),
            "risk_score": self.risk_score(),
            "risk_level": self.risk_level(),
            "findings": [
                {
                    "type": f.entity_type,
                    "text": f.text,
                    "start": f.start,
                    "end": f.end,
                    "confidence": f.confidence,
                }
                for f in self.findings
            ],
        }


class PIIRedactor:
    """Detects and redacts PII in free-form text."""

    def __init__(self, min_name_confidence: float = 0.55):
        self.min_name_confidence = min_name_confidence

    # -- structured detectors -------------------------------------------------
    def _find_structured(self, text: str) -> List[Finding]:
        findings: List[Finding] = []
        claimed_spans: List[tuple] = []

        def overlaps(start, end):
            return any(s < end and start < e for s, e in claimed_spans)

        # Order dictates priority when spans overlap (e.g. SSN before phone).
        for entity_type in [
            "SSN", "CREDIT_CARD", "IP_ADDRESS", "EMAIL", "URL",
            "DATE_OF_BIRTH", "PHONE", "STREET_ADDRESS",
        ]:
            pattern = PATTERNS[entity_type]
            for m in pattern.finditer(text):
                start, end = m.start(), m.end()
                if overlaps(start, end):
                    continue
                value = m.group().strip()

                # Credit-card regex is greedy over 13-19 digit runs; sanity
                # check with the Luhn algorithm to cut false positives.
                if entity_type == "CREDIT_CARD":
                    digits = re.sub(r"\D", "", value)
                    if not (13 <= len(digits) <= 16) or not self._luhn_valid(digits):
                        continue

                findings.append(Finding(entity_type, value, start, end, confidence=0.95))
                claimed_spans.append((start, end))

        return findings, claimed_spans

    @staticmethod
    def _luhn_valid(digits: str) -> bool:
        total = 0
        reverse_digits = digits[::-1]
        for i, d in enumerate(reverse_digits):
            n = int(d)
            if i % 2 == 1:
                n *= 2
                if n > 9:
                    n -= 9
            total += n
        return total % 10 == 0

    # -- name heuristic ---------------------------------------------------
    def _find_names(self, text: str, claimed_spans: List[tuple]) -> List[Finding]:
        findings: List[Finding] = []

        def overlaps(start, end):
            return any(s < end and start < e for s, e in claimed_spans)

        # Candidate: one or two consecutive capitalized words (with optional
        # honorific prefix), e.g. "Dr. Jane Smith", "John Doe".
        candidate_pattern = re.compile(
            rf"\b(?:{HONORIFICS}\s*)?([A-Z][a-z]+)(?:\s+([A-Z][a-z]+))?\b"
        )

        lowered = text.lower()

        for m in candidate_pattern.finditer(text):
            start, end = m.start(), m.end()
            if overlaps(start, end):
                continue
            first, last = m.group(1), m.group(2)
            has_honorific = bool(re.match(rf"^{HONORIFICS}", m.group()))

            confidence = 0.0
            if first and first.lower() in COMMON_FIRST_NAMES:
                confidence += 0.4
            if last and last.lower() in COMMON_LAST_NAMES:
                confidence += 0.4
            if has_honorific:
                confidence += 0.5
            # Context cue in the ~40 chars preceding the match.
            window_start = max(0, start - 40)
            preceding = lowered[window_start:start]
            if any(re.search(cue, preceding) for cue in NAME_CONTEXT_CUES):
                confidence += 0.3
            # A bare single capitalized word with no supporting evidence is
            # too likely to be a sentence-initial word -- require *some*
            # signal beyond capitalization.
            if not last and not has_honorific and confidence < 0.4:
                continue

            confidence = min(confidence, 0.99)
            if confidence >= self.min_name_confidence:
                findings.append(
                    Finding("PERSON_NAME", m.group().strip(), start, end, round(confidence, 2))
                )
                claimed_spans.append((start, end))

        return findings

    # -- public API ---------------------------------------------------------
    def analyze(self, text: str) -> RedactionReport:
        structured, claimed_spans = self._find_structured(text)
        names = self._find_names(text, claimed_spans)
        all_findings = sorted(structured + names, key=lambda f: f.start)
        return RedactionReport(all_findings)

    def redact(self, text: str) -> "tuple[str, RedactionReport]":
        report = self.analyze(text)
        # Replace from the end so earlier offsets stay valid.
        redacted = text
        for f in sorted(report.findings, key=lambda f: f.start, reverse=True):
            placeholder = PLACEHOLDER.get(f.entity_type, "[REDACTED]")
            redacted = redacted[: f.start] + placeholder + redacted[f.end :]
        return redacted, report


# ---------------------------------------------------------------------------
# 3. Demo / CLI
# ---------------------------------------------------------------------------

SAMPLE_TEXTS = {
    "support_ticket": (
        "Hi, my name is John Doe and I need help with my account. "
        "You can reach me at john.doe1985@gmail.com or call (415) 555-2671. "
        "My date of birth is 03/14/1985 and I live at 742 Evergreen Terrace. "
        "For verification my SSN is 123-45-6789. Thanks, John Doe."
    ),
    "hr_email": (
        "Attn: Ms. Sarah Johnson - Please update payroll records. "
        "New address: 1600 Amphitheatre Parkway. Emergency contact: "
        "Dr. Michael Chen, reachable at m.chen@company.org or 650-253-0000. "
        "Employee card on file ends in 4532015112830366."
    ),
    "server_log": (
        "Failed login attempt from IP 192.168.1.45 at endpoint "
        "https://portal.example.com/login for user robert.brown@example.net. "
        "Retry from 10.0.0.7 succeeded."
    ),
    "clean_text": (
        "The quarterly report shows a 12% increase in revenue driven by "
        "strong performance in the Asia-Pacific region. No follow-up "
        "action is required at this time."
    ),
}


def print_report(label: str, original: str, redactor: PIIRedactor) -> None:
    redacted, report = redactor.redact(original)
    print("=" * 78)
    print(f"SAMPLE: {label}")
    print("-" * 78)
    print("ORIGINAL :", original)
    print("REDACTED :", redacted)
    print(
        f"RISK     : {report.risk_level()} (score={report.risk_score()}/100), "
        f"{len(report.findings)} finding(s) -> {report.counts()}"
    )
    print()


def main():
    parser = argparse.ArgumentParser(description="Detect and redact PII in text.")
    parser.add_argument("--text", type=str, help="Raw text to scan.")
    parser.add_argument("--file", type=str, help="Path to a text file to scan.")
    parser.add_argument(
        "--json", action="store_true", help="Print the full JSON report."
    )
    parser.add_argument(
        "--demo", action="store_true", help="Run against bundled sample texts."
    )
    args = parser.parse_args()

    redactor = PIIRedactor()

    if args.text or args.file:
        if args.file:
            with open(args.file, "r", encoding="utf-8") as fh:
                text = fh.read()
        else:
            text = args.text
        redacted, report = redactor.redact(text)
        print("REDACTED TEXT:\n", redacted, "\n")
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(f"Risk: {report.risk_level()} ({report.risk_score()}/100)")
            print(f"Findings by type: {report.counts()}")
        return

    # Default / --demo: run the bundled samples so the tool is runnable
    # out of the box with zero setup.
    for label, text in SAMPLE_TEXTS.items():
        print_report(label, text, redactor)


if __name__ == "__main__":
    main()
