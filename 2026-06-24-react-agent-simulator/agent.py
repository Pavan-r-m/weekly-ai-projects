"""
agent.py — ReAct Agent Simulator
=================================
Implements the ReAct (Reasoning + Acting) pattern introduced in:
  "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022)

The loop:
  Thought     → The agent reasons about what to do next
  Action      → The agent calls a tool with specific arguments
  Observation → The tool returns a result
  ... repeat until the agent produces a Final Answer ...

This demo uses a rule-based "mock LLM" so it runs 100% offline without any
API key. Each demo query is pre-wired with a reasoning trace that shows
exactly how a real LLM-powered ReAct agent would behave.

Usage:
  python agent.py                  # run all demo queries
  python agent.py --list           # list available demo queries
  python agent.py --tools          # list available tools
  python agent.py --query "..."    # run a specific demo query
  python agent.py --no-delay       # skip step delays
"""

import re
import sys
import time
import textwrap
import argparse
from dataclasses import dataclass, field
from typing import Optional
from tools import TOOLS


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Step:
    """One step in the agent's reasoning trace."""
    thought: str
    action: Optional[str] = None        # "tool_name(args)"
    observation: Optional[str] = None   # tool result
    final_answer: Optional[str] = None  # set on last step


@dataclass
class AgentTrace:
    """Complete trace for a single query."""
    query: str
    steps: list = field(default_factory=list)

    def add_step(self, **kwargs) -> Step:
        step = Step(**kwargs)
        self.steps.append(step)
        return step


# ---------------------------------------------------------------------------
# Mock LLM — pre-defined reasoning traces for demo queries
# ---------------------------------------------------------------------------

DEMO_TRACES = {
    "What is 2 to the power of 16, and how many miles is that many kilometres?": [
        {
            "thought": "The user wants two things: 2^16, then convert that number of km to miles. "
                       "I'll start by calculating 2^16.",
            "action": "calculator(2 ** 16)",
        },
        {
            "thought": "2^16 = 65536. Now I need to convert 65536 km to miles.",
            "action": "unit_converter(65536 km to miles)",
        },
        {
            "thought": "I have both answers now. I can compose the final response.",
            "final": True,
        },
    ],
    "How many days are there between New Year's Day 2026 and Christmas 2026?": [
        {
            "thought": "I need to find the number of days between 2026-01-01 and 2026-12-25.",
            "action": "date_calculator(days between 2026-01-01 and 2026-12-25)",
        },
        {
            "thought": "The date calculator gave me the answer directly.",
            "final": True,
        },
    ],
    "Analyse this review: 'The camera is excellent and the battery life is amazing, "
    "but the software crashes constantly and the support was not helpful at all.'": [
        {
            "thought": "This is a mixed product review. I'll run sentiment analysis first, "
                       "then get word stats to characterise its length.",
            "action": "sentiment_analyzer(The camera is excellent and the battery life is amazing, "
                      "but the software crashes constantly and the support was not helpful at all.)",
        },
        {
            "thought": "Interesting mixed result. Let me also get word statistics.",
            "action": "word_stats(The camera is excellent and the battery life is amazing, "
                      "but the software crashes constantly and the support was not helpful at all.)",
        },
        {
            "thought": "I now have both sentiment and length data. Ready for final answer.",
            "final": True,
        },
    ],
    "Convert 37 degrees Celsius to Fahrenheit, then tell me the square root of that Fahrenheit value.": [
        {
            "thought": "Two-step task: first convert 37 C to F, then take sqrt of the F result.",
            "action": "unit_converter(37 celsius to fahrenheit)",
        },
        {
            "thought": "37 C = 98.6 F. Now I'll compute sqrt(98.6).",
            "action": "calculator(sqrt(98.6))",
        },
        {
            "thought": "sqrt(98.6) is approximately 9.93. I have all the information needed.",
            "final": True,
        },
    ],
    "What is today's date, and how many days until 2027?": [
        {
            "thought": "I need two pieces of info: today's date, and days until 2027-01-01.",
            "action": "date_calculator(today)",
        },
        {
            "thought": "Good, I know today's date. Now let me find days until January 1, 2027.",
            "action": "date_calculator(days until 2027-01-01)",
        },
        {
            "thought": "I have both answers now.",
            "final": True,
        },
    ],
    "Count the words in 'To be or not to be, that is the question.' "
    "and calculate how many unique words there are as a percentage.": [
        {
            "thought": "First I'll use word_stats to get total and unique word counts.",
            "action": "word_stats(To be or not to be, that is the question.)",
        },
        {
            "thought": "The stats show 10 words and 7 unique. "
                       "I'll calculate the uniqueness percentage: 7/10 * 100.",
            "action": "calculator(round(7 / 10 * 100, 1))",
        },
        {
            "thought": "70.0% of words are unique. I can now give a complete answer.",
            "final": True,
        },
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mock_llm_get_trace(query: str):
    """Return the pre-defined trace for a query, or None if not recognised."""
    if query in DEMO_TRACES:
        return DEMO_TRACES[query]
    q_lower = query.lower()
    for key, trace in DEMO_TRACES.items():
        if key.lower() in q_lower or q_lower in key.lower():
            return trace
    return None


def generate_final_answer(steps: list) -> str:
    """Compose a natural-language answer from all observations collected."""
    observations = [s.observation for s in steps if s.observation]
    if not observations:
        return "I was unable to find an answer with the available tools."
    if len(observations) == 1:
        return f"Based on my calculation: {observations[0]}."
    parts = "\n".join(f"  • {obs}" for obs in observations)
    return f"Here are the results from each step:\n{parts}"


def parse_action(action_str: str):
    """Parse 'tool_name(args)' → (tool_name, args_string) or (None, None)."""
    match = re.match(r'(\w+)\((.+)\)$', action_str.strip(), flags=re.DOTALL)
    if match:
        return match.group(1), match.group(2)
    return None, None


def execute_tool(tool_name: str, args: str) -> str:
    """Look up and call the named tool, returning the result as a string."""
    if tool_name not in TOOLS:
        return f"Unknown tool '{tool_name}'. Available: {', '.join(TOOLS.keys())}"
    try:
        return TOOLS[tool_name]["fn"](args)
    except Exception as ex:
        return f"Tool error: {ex}"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ReActAgent:
    """
    Runs the ReAct loop: Thought → Action → Observation → … → Final Answer.

    Parameters
    ----------
    max_steps : int   Safety limit on reasoning steps.
    delay     : float Seconds to pause between steps (aids readability).
    verbose   : bool  Print each step as it executes.
    """

    SEP  = "─" * 60
    HEAD = "═" * 60

    def __init__(self, max_steps: int = 10, delay: float = 0.3, verbose: bool = True):
        self.max_steps = max_steps
        self.delay = delay
        self.verbose = verbose

    def _print_step(self, num: int, step: Step):
        wrap = lambda t: textwrap.fill(t, width=68, initial_indent="     ", subsequent_indent="     ")
        print(f"\n{self.SEP}")
        print(f"  Step {num}")
        print(self.SEP)
        print(f"  💭 Thought:\n{wrap(step.thought)}")
        if step.action:
            print(f"\n  🔧 Action:  {step.action}")
        if step.observation:
            print(f"\n  👁  Observation:\n{wrap(step.observation)}")
        if step.final_answer:
            print(f"\n  ✅ Final Answer:\n{wrap(step.final_answer)}")

    def run(self, query: str) -> str:
        """Execute the ReAct loop for a query. Returns the final answer."""
        print(f"\n{self.HEAD}")
        print(f"  🤖 ReAct Agent")
        print(self.HEAD)
        print(f"  Query: {query}")

        raw_steps = mock_llm_get_trace(query)
        if raw_steps is None:
            msg = (
                "No pre-defined reasoning trace for that query. "
                "Run `python agent.py --list` to see supported queries."
            )
            print(f"\n  ⚠️  {msg}")
            return msg

        trace = AgentTrace(query=query)

        for i, raw in enumerate(raw_steps):
            if i >= self.max_steps:
                print(f"\n  ⚠️  Max steps ({self.max_steps}) reached.")
                break

            is_final   = raw.get("final", False)
            action_str = raw.get("action")

            # --- Execute tool (if any) ---
            observation = None
            if action_str:
                tool_name, args = parse_action(action_str)
                observation = (
                    execute_tool(tool_name, args) if tool_name
                    else f"Could not parse action: {action_str}"
                )

            # --- Build final answer on last step ---
            final_answer = None
            if is_final:
                # Include latest observation in the answer if it exists
                all_obs = [s.observation for s in trace.steps if s.observation]
                if observation:
                    all_obs.append(observation)
                if len(all_obs) == 1:
                    final_answer = f"Based on my calculation: {all_obs[0]}."
                elif all_obs:
                    parts = "\n".join(f"  • {o}" for o in all_obs)
                    final_answer = f"Here are the results:\n{parts}"
                else:
                    final_answer = "No tool results were needed for this answer."

            step = trace.add_step(
                thought=raw["thought"],
                action=action_str,
                observation=observation,
                final_answer=final_answer,
            )

            if self.verbose:
                self._print_step(i + 1, step)
                time.sleep(self.delay)

            if is_final:
                return final_answer or ""

        # Fallback
        final = generate_final_answer(trace.steps)
        if self.verbose:
            print(f"\n  ✅ Final Answer: {final}")
        return final


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def list_demo_queries():
    print("\nAvailable demo queries:")
    for i, q in enumerate(DEMO_TRACES.keys(), 1):
        short = q if len(q) <= 80 else q[:77] + "..."
        print(f"  [{i}] {short}")


def list_tools():
    print("\nAvailable tools:")
    for name, info in TOOLS.items():
        print(f"  • {name}")
        print(f"      {info['description']}")
        print(f"      Example: {info['example']}")


def main():
    parser = argparse.ArgumentParser(
        description="ReAct Agent Simulator — demonstrates the Reasoning+Acting loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python agent.py                   # run all demo queries
          python agent.py --list            # list demo queries
          python agent.py --tools           # list available tools
          python agent.py --query "..."     # run a specific query
          python agent.py --no-delay        # no step delays
        """),
    )
    parser.add_argument("--query", "-q", help="Run a specific query")
    parser.add_argument("--list",  "-l", action="store_true", help="List demo queries")
    parser.add_argument("--tools", "-t", action="store_true", help="List available tools")
    parser.add_argument("--no-delay", action="store_true", help="Disable step delays")
    args = parser.parse_args()

    delay = 0.0 if args.no_delay else 0.3

    if args.list:
        list_demo_queries()
        return
    if args.tools:
        list_tools()
        return

    agent = ReActAgent(max_steps=10, delay=delay, verbose=True)

    if args.query:
        agent.run(args.query)
    else:
        print("\n" + "█" * 60)
        print("  ReAct Agent Simulator — Running all demo queries")
        print("█" * 60)
        for query in DEMO_TRACES.keys():
            agent.run(query)
            print()


if __name__ == "__main__":
    main()
