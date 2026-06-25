# ReAct Agent Simulator

A clean, runnable Python implementation of the **ReAct (Reasoning + Acting)** pattern — one of the foundational ideas behind modern AI agents.

> **No API key required.** Uses a rule-based mock LLM so the full loop runs offline.

---

## What It Does

The agent receives a natural-language query and solves it by looping through three phases until it has an answer:

```
Thought  →  the agent reasons about what to do next
Action   →  the agent calls a tool with specific arguments
Observation →  the tool returns a real result
… repeat …
Final Answer
```

This mirrors how production agents (LangChain, AutoGPT, Claude tool-use) actually work — breaking a complex question into a series of grounded tool calls.

### Built-in Tools

| Tool | What it does |
|---|---|
| `calculator` | Safely evaluates math: `2**16`, `sqrt(144)`, `sin(pi/2)` |
| `unit_converter` | Converts km↔miles, kg↔lbs, °C↔°F, m↔ft, L↔gal |
| `sentiment_analyzer` | Rule-based Positive / Negative / Neutral with confidence % |
| `date_calculator` | Days between dates, days until a date, today's date |
| `word_stats` | Word / character / sentence count + uniqueness ratio |

---

## Tech Stack

- **Language:** Python 3.10+ — standard library only
- **Pattern:** ReAct (Yao et al., 2022) — [arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)
- **Design:** dataclass-based trace, registry-driven tools, swappable LLM backend

---

## Installation

```bash
git clone https://github.com/Pavan-r-m/weekly-ai-projects.git
cd weekly-ai-projects/2026-06-24-react-agent-simulator
pip install -r requirements.txt   # nothing to install — stdlib only
```

---

## How to Run

```bash
# Run all 6 built-in demo queries
python agent.py

# List available demo queries
python agent.py --list

# Run a specific query
python agent.py --query "What is today's date, and how many days until 2027?"

# List all available tools
python agent.py --tools

# Run without step delays (faster)
python agent.py --no-delay
```

---

## Example Output

```
════════════════════════════════════════════════════════════
  🤖 ReAct Agent
════════════════════════════════════════════════════════════
  Query: What is 2 to the power of 16, and how many miles is that many kilometres?

────────────────────────────────────────────────────────────
  Step 1
────────────────────────────────────────────────────────────
  💭 Thought:
     The user wants two things: 2^16, then convert that number of
     km to miles. I'll start by calculating 2^16.

  🔧 Action:  calculator(2 ** 16)

  👁  Observation:
     65536

────────────────────────────────────────────────────────────
  Step 2
────────────────────────────────────────────────────────────
  💭 Thought:
     2^16 = 65536. Now I need to convert 65536 km to miles.

  🔧 Action:  unit_converter(65536 km to miles)

  👁  Observation:
     65536 km = 40722.1699 miles

────────────────────────────────────────────────────────────
  Step 3
────────────────────────────────────────────────────────────
  💭 Thought:
     I have both answers now. I can compose the final response.

  ✅ Final Answer:
     Here are the results:
       • 65536
       • 65536 km = 40722.1699 miles
```

---

## How It Works

### The ReAct Pattern

Standard LLMs answer in one shot. ReAct agents **interleave** reasoning with tool calls:

```
Query
  └─ Thought₁ → Action₁ → Observation₁
       └─ Thought₂ → Action₂ → Observation₂
            └─ Thought₃ → Final Answer
```

Each Observation is fed back into the next Thought, so the agent adapts — it sees real results before deciding what to do next.

### Code Architecture

```
agent.py    ReActAgent class, mock LLM traces, CLI entry point
tools.py    Five tool functions + TOOLS registry dict
```

**`ReActAgent.run(query)`**:
1. Calls `mock_llm_get_trace(query)` — in production, this is an LLM API call
2. For each step, parses `tool_name(args)` from the Action string
3. Calls the matching function from the `TOOLS` registry
4. Stores the Observation and passes context to the next Thought
5. On the final step, composes a natural-language answer from all observations

### Extending with a Real LLM

Replace `mock_llm_get_trace()` with an actual LLM call:

```python
import openai

SYSTEM_PROMPT = """You are a ReAct agent. On each turn output exactly:
Thought: <your reasoning>
Action: tool_name(args)

On the last turn:
Thought: <reasoning>
Final Answer: <answer>

Available tools: calculator, unit_converter, sentiment_analyzer, date_calculator, word_stats"""

def llm_think(query: str, history: list) -> str:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
    )
    return response.choices[0].message.content
```

### Safe Calculator

The `calculator` tool uses a **whitelist regex + restricted `eval()`** to prevent code injection. Only math symbols and whitelisted function names (`sqrt`, `sin`, `cos`, `log`, `pi`, `e`, etc.) are allowed — no builtins, no imports, no arbitrary Python.

---

## Project Structure

```
2026-06-24-react-agent-simulator/
├── agent.py          # Agent loop, mock LLM traces, CLI
├── tools.py          # Tool implementations + TOOLS registry
├── requirements.txt  # No external dependencies
└── README.md         # This file
```
