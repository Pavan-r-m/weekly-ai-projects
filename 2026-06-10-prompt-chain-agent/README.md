# 🔗 Self-Correcting Prompt Chain Agent

**Category:** AI Agent / Automation  
**Date:** 2026-06-10  
**Stack:** Python (stdlib only) — no API key required

---

## What It Does

This project implements a **self-correcting prompt chain agent** — an AI orchestration pattern where a complex goal is broken into sequential reasoning steps, each step is executed via an LLM, and low-quality outputs are automatically detected and retried with refined prompts.

Key behaviours:
- **Goal decomposition** — any topic or question is split into 5 analytical steps
- **Quality evaluation** — each LLM output is scored using heuristics (length, structure, sentence variety, error markers)
- **Automatic retry with prompt refinement** — if a step scores below the quality threshold, the prompt is enhanced and retried (up to 3 attempts)
- **Escalating refinement strategy** — Attempt 2 adds format constraints; Attempt 3 adds chain-of-thought scaffolding
- **Output synthesis** — all step outputs are merged into a structured analysis report

The demo uses a **local mock LLM** (no API key needed), but replacing one function with a real API call (OpenAI, Anthropic, etc.) gives you a production-ready agent.

---

## Why It's Interesting

Prompt chaining is a foundational pattern in LLM engineering — used in tools like LangChain, LlamaIndex, and AutoGPT. But most tutorials skip the self-correction layer. This project shows:

1. **Why quality gates matter** — LLMs sometimes return empty, short, or off-topic responses
2. **How to detect failures programmatically** — without relying on the model to self-report errors
3. **How prompt refinement escalates** — from format hints to full chain-of-thought scaffolding
4. **How to synthesize multi-step outputs** — merging partial results into a coherent final report

---

## Installation

```bash
# No pip install needed — uses Python standard library only
python --version   # Python 3.8+ required
```

---

## How to Run

```bash
# Default goal (neural networks)
python prompt_chain_agent.py

# Custom goal
python prompt_chain_agent.py --goal "Explain how blockchain works"

# Verbose mode — see every LLM call attempt
python prompt_chain_agent.py --goal "How does CRISPR gene editing work" --verbose

# Disable fixed seed for varied results
python prompt_chain_agent.py --seed 0

# Combine flags
python prompt_chain_agent.py --goal "What is quantum computing" --verbose --seed 0
```

---

## Example Output

```
Goal decomposed into 5 steps.

==============================================================
  PROMPT CHAIN AGENT
  Goal: Explain how neural networks learn from data
==============================================================

[Step 1/5] Core Understanding
  ✓ Final score: 0.95 | Attempts: 1 | Passed

[Step 2/5] Component Analysis
  ↺ Retry 2: prev_score=0.00, prompt expanded +109 chars
  ✓ Final score: 0.95 | Attempts: 2 | Passed

[Step 3/5] Relationship Mapping
  ✓ Final score: 0.90 | Attempts: 1 | Passed

[Step 4/5] Concrete Examples
  ✓ Final score: 0.95 | Attempts: 1 | Passed

[Step 5/5] Synthesis & Insights
  ✓ Final score: 0.90 | Attempts: 1 | Passed

==============================================================
  EXECUTION SUMMARY
==============================================================
  Goal:           Explain how neural networks learn from data
  Total steps:    5
  Total attempts: 6
  Success rate:   100%  (5/5 steps passed)
  Elapsed time:   0.91s

  ID   Step Name                Score    Tries   Retried
  -------------------------------------------------------
  ✓1   Core Understanding       0.95     1       No
  ✓2   Component Analysis       0.95     2       Yes
  ✓3   Relationship Mapping     0.90     1       No
  ✓4   Concrete Examples        0.95     1       No
  ✓5   Synthesis & Insights     0.90     1       No
```

---

## How It Works

### 1. Goal Decomposition
Every user goal is split into a fixed 5-step chain:
`Core Understanding → Component Analysis → Relationship Mapping → Concrete Examples → Synthesis`

### 2. Quality Scoring
Each LLM response is scored 0–1 by checking:
- **Length** (50+ words = 0.35 points)
- **Structure** (numbered lists, colons, line breaks)
- **Sentence variety** (3+ complete sentences)
- **Error penalty** (phrases like "I cannot" deduct points)

### 3. Retry & Refinement
If `score < threshold (0.6)`:
- **Attempt 2**: Adds `[FORMAT INSTRUCTION]` to the prompt
- **Attempt 3**: Adds a full `[CHAIN-OF-THOUGHT]` scaffold

The agent always keeps the best output across all attempts.

### 4. Synthesis
All 5 step outputs are merged into a structured Markdown report with quality labels.

---

## Connecting a Real LLM

Replace the `mock_llm_call` function body with a real API call:

```python
# OpenAI
import openai
def mock_llm_call(prompt, step_type="default", simulate_failure=False):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Anthropic
import anthropic
def mock_llm_call(prompt, step_type="default", simulate_failure=False):
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text
```

Set your API key as an environment variable:
```bash
export OPENAI_API_KEY="sk-..."
# or
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Project Structure

```
2026-06-10-prompt-chain-agent/
├── prompt_chain_agent.py   # Main agent (400+ lines, fully commented)
├── requirements.txt        # No deps needed; optional real LLM packages listed
└── README.md               # This file
```

Output file `chain_output.txt` is written to your working directory after each run.
