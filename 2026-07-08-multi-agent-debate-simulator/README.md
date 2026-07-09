# Multi-Agent Debate Simulator

Three AI agents -- a **Proposer**, a **Critic**, and a **Judge** -- debate a
motion over several structured rounds, with the Judge scoring each exchange
and declaring a winner at the end.

> **No API key required.** Ships with a deterministic, offline `MockLLM` so
> the entire multi-agent loop runs with zero cost and zero setup. A clearly
> marked extension point shows exactly where to plug in a real model.

---

## Why This Is Interesting

Most single-agent demos show an LLM answering a question. This project
demonstrates the next step up: **multi-agent orchestration**, where several
agents with distinct roles, distinct incentives, and shared context interact
over multiple turns to produce something none of them could produce alone.

This is the same underlying pattern used in production systems such as:
- **Debate-based evaluation** -- having two models argue opposing sides to
  surface a stronger, more scrutinized answer than a single pass would.
- **Red-team / blue-team review** -- one agent attacks a design or piece of
  code, another defends and patches it.
- **Editor / writer pipelines** -- one agent drafts, another critiques,
  and a third (or the orchestrator itself) decides when the result is good
  enough.

The `DebateOrchestrator` class is the reusable piece: it manages turn-taking,
passes each agent's output as context to the next agent, and hands the full
transcript to a scoring agent at the end -- independent of what the agents
actually say.

## Tech Stack and Key Concepts

- **Pure Python standard library** (`argparse`, `random`, `re`, `json`,
  `collections.Counter`, `datetime`) -- no pip installs needed to run it.
- **Role-based agents** (`ProposerAgent`, `CriticAgent`, `JudgeAgent`) built
  on a shared `DebateAgent` base class, each with its own stance, argument
  bank, and memory of what it has said.
- **Mock LLM interface** (`MockLLM` / `get_llm_backend`) -- a drop-in
  stand-in for a real language model that implements the same method
  signatures (`generate_opening`, `generate_rebuttal`, `generate_closing`,
  `score_round`) a real API-backed class would need, so swapping in OpenAI
  or Anthropic later means writing one new class, not rewriting the
  orchestrator.
- **Keyword-based rebuttal grounding** -- each rebuttal extracts keywords
  and a cleaned claim snippet from the opponent's previous turn so replies
  read as responses to what was actually said, not generic canned text.
- **Heuristic scoring** -- the Judge scores lexical diversity, argument
  length/specificity, and rebuttal relevance (keyword overlap with the
  opponent's prior statement) to produce a per-round, per-side score.
- **Orchestration loop** -- opening statements, N rebuttal rounds, closing
  statements, then a final verdict tallied across every scored exchange.

## Installation

```bash
pip install -r requirements.txt   # no-op: stdlib only
```

Requires Python 3.8+.

## How to Run It

Random topic, 3 rebuttal rounds (default):
```bash
python debate_simulator.py
```

List all available motions:
```bash
python debate_simulator.py --list-topics
```

Pick a specific topic and round count, with a reproducible seed:
```bash
python debate_simulator.py --topic 0 --rounds 4 --seed 42
```

Save the transcript as Markdown and/or the full structured result as JSON:
```bash
python debate_simulator.py --topic 3 --rounds 3 --save transcript.md --json result.json
```

Suppress live printout (useful when only the saved files matter):
```bash
python debate_simulator.py --topic 1 --quiet --save transcript.md
```

## Example Output

```
Moderator: Motion: "Should AI-generated code be reviewed by humans before merging?"

[Round 0] Proposer: The evidence strongly suggests AI models can confidently produce subtly incorrect logic.

[Round 0] Critic: Consider that mandatory review encourages rubber-stamping rather than real scrutiny.

[Round 0] Judge: Round 0 scores -- Proposer: 4.92, Critic: 4.76

[Round 1] Proposer: My opponent claims 'Consider that mandatory review encourages rubber-stamping rather than real scrutiny.', but this overlooks human reviewers catch context and intent that automated tools miss (especially regarding consider, mandatory).

...

Judge: VERDICT: Proposer wins (40.31 vs 39.43 over 5 scored exchanges)
```

## How It Works

1. **Setup** -- `DebateOrchestrator` picks a topic from `topics.py`, which
   stores a bank of "for" and "against" talking points per motion. Each
   agent gets the argument bank matching its stance.
2. **Opening statements** -- Proposer and Critic each pick a point from
   their bank and wrap it in a rhetorical opening template.
3. **Rebuttal rounds** -- each side receives the *other* side's last
   statement, extracts keywords and a cleaned claim snippet from it via
   `MockLLM._extract_keywords` / `_clean_claim_snippet`, and generates a
   rebuttal that explicitly references what the opponent just said.
4. **Judging** -- after every exchange, `JudgeAgent.score()` computes a
   heuristic score per side based on lexical diversity, length/specificity,
   and how directly the rebuttal engaged with the opponent's prior point.
5. **Closing statements** -- each side restates its strongest point.
6. **Verdict** -- `JudgeAgent.verdict()` sums every round's scores and
   declares a winner (or a tie).
7. **Export** -- `save_markdown()` and the `--json` flag turn the in-memory
   transcript into shareable artifacts for later review.

### Extending to a Real LLM

`agents.get_llm_backend()` is the single seam to change. Implement a class
with the same four methods as `MockLLM` (`generate_opening`,
`generate_rebuttal`, `generate_closing`, `score_round`), wire it up to
`openai` or `anthropic` inside that function when an API key is present,
and the rest of the orchestrator, CLI, and export logic works unchanged.
