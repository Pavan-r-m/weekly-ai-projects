# PII Text Redactor & Anonymizer

A zero-dependency NLP tool that scans free-form text for personally
identifiable information (PII), redacts it in place, and produces a
machine-readable audit report with a weighted **privacy risk score**.

## Why it's interesting

Every support ticket, HR email, or server log is a potential compliance
liability if it contains emails, SSNs, credit card numbers, or names that
shouldn't be logged, shared, or fed into an LLM prompt. This tool shows how
a production-grade PII scrubber (think AWS Comprehend or Microsoft
Presidio) is actually built in practice: a layered pipeline of high-precision
regex detectors for **structured** PII (emails, phone numbers, SSNs, credit
cards, IP addresses, URLs, dates of birth, street addresses), plus a
lightweight **statistical heuristic** for **unstructured** PII (person
names) that combines a name gazetteer, honorific detection, and contextual
cue phrases -- all without downloading a multi-hundred-megabyte NER model.

It also demonstrates the **Luhn algorithm** for validating candidate credit
card numbers (cutting false positives from arbitrary 16-digit strings) and
a saturating exponential curve for turning raw "hit counts" into a bounded
0-100 risk score.

## Tech stack & key concepts

- **Python 3 standard library only** (`re`, `dataclasses`, `argparse`, `json`)
- Regex-based structured entity recognition (emails, SSNs, credit cards,
  IPs, URLs, dates, street addresses)
- Luhn checksum validation for credit card candidates
- Heuristic statistical NER for person names (gazetteer + honorifics +
  context cues + confidence scoring)
- Span-overlap resolution so multiple detectors don't double-redact the
  same text
- Weighted, saturating risk scoring (`LOW` / `MEDIUM` / `HIGH` / `CRITICAL`)

## Installation

No third-party packages are required to run the tool itself:

```bash
pip install -r requirements.txt   # only needed if you plan to add pytest tests
```

## How to run it

**Demo mode** -- runs the redactor against four bundled sample texts
(a support ticket, an HR email, a server log, and a clean control text):

```bash
python3 redactor.py --demo
# or simply:
python3 redactor.py
```

**Scan your own text:**

```bash
python3 redactor.py --text "Contact John Doe at john@example.com or 555-123-4567"
```

**Scan a file and print the full JSON report:**

```bash
python3 redactor.py --file ticket.txt --json
```

## Example output

```
SAMPLE: support_ticket
------------------------------------------------------------------------------
ORIGINAL : Hi, my name is John Doe and I need help with my account. You can
           reach me at john.doe1985@gmail.com or call (415) 555-2671. My date
           of birth is 03/14/1985 and I live at 742 Evergreen Terrace. For
           verification my SSN is 123-45-6789. Thanks, John Doe.
REDACTED : Hi, my name is [REDACTED_NAME] and I need help with my account.
           You can reach me at [REDACTED_EMAIL] or call [REDACTED_PHONE]. My
           date of birth is [REDACTED_DOB] and I live at [REDACTED_ADDRESS]
           For verification my SSN is [REDACTED_SSN]. Thanks, [REDACTED_NAME].
RISK     : CRITICAL (score=83.5/100), 7 finding(s) ->
           {'PERSON_NAME': 2, 'EMAIL': 1, 'PHONE': 1, 'DATE_OF_BIRTH': 1,
            'STREET_ADDRESS': 1, 'SSN': 1}
```

JSON report shape (`--json`):

```json
{
  "total_findings": 2,
  "counts_by_type": { "PHONE": 1, "EMAIL": 1 },
  "risk_score": 29.5,
  "risk_level": "MEDIUM",
  "findings": [
    { "type": "PHONE", "text": "512-444-9821", "start": 13, "end": 25, "confidence": 0.95 },
    { "type": "EMAIL", "text": "jane@site.io", "start": 29, "end": 41, "confidence": 0.95 }
  ]
}
```

## How it works

1. **Structured detection** -- a prioritized list of regex patterns
   (SSN → credit card → IP → email → URL → date of birth → phone → street
   address) is applied to the text. Matches are tracked as claimed
   character spans so lower-priority patterns can't re-claim text already
   matched by a higher-priority one (e.g. a 9-digit SSN never also gets
   flagged as a phone number).
2. **Credit card validation** -- any 13-16 digit run caught by the credit
   card pattern is checked against the **Luhn algorithm** before being
   accepted, which is the same checksum real card issuers use and
   eliminates most false positives from arbitrary long numbers.
3. **Name detection** -- candidate spans are one or two consecutive
   capitalized words, optionally preceded by an honorific ("Dr.", "Ms.").
   Each candidate accumulates confidence from four independent signals:
   first name in a common-name gazetteer (+0.4), last name in the
   gazetteer (+0.4), a leading honorific (+0.5), and a nearby context cue
   like "my name is" or "Attn:" (+0.3). Only candidates crossing a
   confidence threshold (default 0.55) are redacted, which keeps ordinary
   capitalized words (e.g. sentence-initial words, place names) from being
   misflagged.
4. **Redaction** -- all findings are sorted by position and replaced with
   type-specific placeholders (`[REDACTED_EMAIL]`, `[REDACTED_SSN]`, etc.),
   working from the end of the string backwards so earlier character
   offsets stay valid.
5. **Risk scoring** -- each finding type has a hand-tuned sensitivity
   weight (SSN=10, credit card=9, date of birth=6, address=5, name=4,
   phone=4, email=3, IP=2, URL=1). The weights are summed and passed
   through a saturating exponential (`100 * (1 - e^(-raw/20))`) so a
   single SSN or credit card already pushes the score into `HIGH`/`CRITICAL`
   territory, while a lone email address stays `LOW`/`MEDIUM` -- mirroring
   how real privacy-risk scoring should not scale purely linearly with hit
   count.

## Limitations & extension ideas

- The name detector is a heuristic, not a trained NER model -- it will
  miss uncommon names and can occasionally misfire on capitalized non-name
  phrases. Swapping in `spaCy`'s `en_core_web_sm` NER pipeline (or a
  transformer-based model) would improve recall at the cost of a larger
  dependency footprint.
- Regex-based detectors are locale-specific (US phone/SSN/address formats).
  Extending `PATTERNS` with international formats is straightforward.
- No API key or external service is required anywhere in this project --
  it runs fully offline.
