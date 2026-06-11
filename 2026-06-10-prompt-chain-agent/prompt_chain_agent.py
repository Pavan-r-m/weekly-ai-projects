"""
prompt_chain_agent.py
=====================
A self-correcting prompt chain agent that:
1. Decomposes a user goal into a sequence of reasoning steps
2. Executes each step through a (mocked) LLM
3. Evaluates output quality using heuristic checks
4. Automatically retries failed steps with refined prompts
5. Produces a final merged result

This demo uses a local mock LLM that simulates realistic text outputs
so the project runs without any API keys.

Run: python prompt_chain_agent.py
     python prompt_chain_agent.py --goal "Explain how blockchain works"
     python prompt_chain_agent.py --goal "Write a bedtime story about a robot" --verbose
"""

import argparse
import re
import time
import random
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Mock LLM - simulates a language model without requiring an API key.
# Replace mock_llm_call with a real API call (OpenAI, Anthropic, etc.)
# ---------------------------------------------------------------------------

MOCK_RESPONSES = {
    "decompose": [
        "Step 1: Understand the core concept.\nStep 2: Identify key components.\nStep 3: Explain relationships.\nStep 4: Provide examples.\nStep 5: Summarize insights.",
        "Step 1: Define the topic clearly.\nStep 2: Break down subtopics.\nStep 3: Analyze each part.\nStep 4: Synthesize findings.\nStep 5: Draw conclusions.",
    ],
    "understand": [
        "The core concept involves a fundamental principle where multiple elements interact in a structured way. This creates emergent behavior that is greater than the sum of its parts. Understanding this foundation is essential for everything that follows.",
        "At its heart, this topic revolves around the balance between complexity and simplicity. Understanding it requires examining both micro and macro perspectives. The key insight is that structure enables function at every level.",
    ],
    "identify": [
        "Key components include: (1) the primary mechanism that drives the system, (2) supporting structures that maintain stability, (3) feedback loops that enable adaptation, and (4) external interfaces that connect to the broader environment. Each plays a crucial role.",
        "The main elements are: input processing, transformation logic, output generation, and error handling. These work together to achieve the desired outcome. Removing any single element degrades the entire system's performance.",
    ],
    "explain": [
        "The relationships between components form a network of dependencies. When one element changes, it propagates effects through connected parts, creating dynamic behavior. This interdependence is what gives the system its robustness and flexibility.",
        "Components interact through well-defined interfaces. Data flows from inputs through processing stages, with each stage adding value before passing results downstream. The result is a pipeline that transforms raw information into actionable insights.",
    ],
    "examples": [
        "For example, consider a real-world scenario: when you search online, a query goes through tokenization, semantic analysis, ranking algorithms, and finally result rendering — each step building on the last. Another example is how GPS navigation combines satellite signals, map data, and routing algorithms to give turn-by-turn directions.",
        "A practical example: a recommendation system takes user history (input), applies collaborative filtering (processing), generates ranked suggestions (output), then measures click-through (feedback). Similarly, email spam filters use learned patterns from millions of examples to classify new messages in milliseconds.",
    ],
    "summarize": [
        "In summary, this system demonstrates how structured decomposition enables complex problem-solving. The key insight is that breaking large problems into manageable steps makes them tractable. Each step adds value, and the combination produces results no single step could achieve alone.",
        "To conclude: the topic reveals elegant patterns when viewed through a systematic lens. The interplay of components creates robust, adaptable behavior. Most importantly, understanding these patterns empowers you to design better systems and solve harder problems.",
    ],
    "default": [
        "This step involves careful analysis and structured thinking. By approaching the problem systematically, we can extract meaningful insights and build toward a comprehensive understanding. The process itself teaches us as much as the outcome.",
        "Processing this aspect requires consideration of multiple factors. The approach taken here balances thoroughness with clarity, ensuring the output is both accurate and accessible to a wide audience.",
    ],
}


def mock_llm_call(prompt: str, step_type: str = "default",
                  simulate_failure: bool = False) -> Optional[str]:
    """
    Simulate an LLM call with realistic latency and occasional failures.
    Returns None ~15% of the time when simulate_failure=True.

    To use a real LLM, replace this function body with:
        import openai
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    """
    # Simulate network latency (100-300ms)
    time.sleep(0.1 + random.random() * 0.2)

    # Simulate occasional API failures / empty responses
    if simulate_failure and random.random() < 0.15:
        return None

    # Return a response matching the step type
    for key, responses in MOCK_RESPONSES.items():
        if key in step_type.lower():
            return random.choice(responses)

    return random.choice(MOCK_RESPONSES["default"])


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ChainStep:
    """Represents one step in a prompt chain."""
    step_id: int
    name: str
    prompt_template: str
    step_type: str
    max_retries: int = 3
    quality_threshold: float = 0.6

    # Populated during execution
    prompt: str = ""
    output: str = ""
    quality_score: float = 0.0
    attempts: int = 0
    succeeded: bool = False
    retry_notes: List[str] = field(default_factory=list)


@dataclass
class ChainResult:
    """Aggregate result from running a full prompt chain."""
    goal: str
    steps: List[ChainStep]
    final_output: str
    total_attempts: int
    success_rate: float
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Quality evaluator
# ---------------------------------------------------------------------------

def evaluate_output_quality(output: Optional[str]) -> float:
    """
    Score output quality from 0.0 to 1.0 using heuristics.
    Checks: non-empty, sufficient length, structure, sentence variety, no errors.
    """
    if output is None or not output.strip():
        return 0.0

    score = 0.0
    word_count = len(output.split())

    # Length: 30+ words is baseline for useful content
    if word_count >= 50:
        score += 0.35
    elif word_count >= 30:
        score += 0.2
    elif word_count >= 15:
        score += 0.1

    # Structure indicators: numbered lists, colons, line breaks
    if re.search(r'\d+[\.\):]', output):   # numbered list item
        score += 0.2
    if ':' in output:                       # label-value pairs
        score += 0.1
    if '\n' in output.strip():             # multi-line content
        score += 0.1

    # Sentence variety (multiple complete sentences = richer content)
    sentences = [s.strip() for s in re.split(r'[.!?]', output) if len(s.strip()) > 5]
    if len(sentences) >= 4:
        score += 0.2
    elif len(sentences) >= 2:
        score += 0.1

    # Penalise error-like output
    error_phrases = ["i cannot", "i'm unable", "i don't know", "error:", "undefined"]
    if any(p in output.lower() for p in error_phrases):
        score -= 0.3

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Prompt refinement
# ---------------------------------------------------------------------------

def refine_prompt(original_prompt: str, attempt: int) -> str:
    """
    Produce a refined prompt for retry attempts.
    Strategy escalates with each attempt.
    """
    if attempt == 2:
        # Add explicit format instruction
        return (
            original_prompt
            + "\n\n[FORMAT INSTRUCTION]: Structure your response with at least "
            "3 numbered points. Each point should include a brief explanation."
        )
    elif attempt >= 3:
        # Chain-of-thought + scaffolded structure
        return (
            original_prompt
            + "\n\n[CHAIN-OF-THOUGHT]: Think step by step.\n"
            "1. Identify the main concept.\n"
            "2. Break it into components.\n"
            "3. Explain each component clearly.\n"
            "4. Give a concrete example.\n"
            "Provide substantive detail for all four points."
        )
    return original_prompt


# ---------------------------------------------------------------------------
# Goal decomposer
# ---------------------------------------------------------------------------

def decompose_goal(goal: str):
    """
    Break a user goal into a 5-step analysis chain.
    Returns (steps list, decomposition plan string).
    """
    decomp_prompt = f"Break down this goal into clear analytical steps: {goal}"
    decomp_plan = mock_llm_call(decomp_prompt, step_type="decompose")

    steps = [
        ChainStep(
            step_id=1, name="Core Understanding",
            prompt_template=f"Provide a clear, foundational explanation of: {goal}. Focus on the core concept.",
            step_type="understand",
        ),
        ChainStep(
            step_id=2, name="Component Analysis",
            prompt_template=f"Identify and describe the key components or elements involved in: {goal}.",
            step_type="identify",
        ),
        ChainStep(
            step_id=3, name="Relationship Mapping",
            prompt_template=f"Explain how the components of '{goal}' relate to and interact with each other.",
            step_type="explain",
        ),
        ChainStep(
            step_id=4, name="Concrete Examples",
            prompt_template=f"Provide specific, real-world examples that illustrate '{goal}' in action.",
            step_type="examples",
        ),
        ChainStep(
            step_id=5, name="Synthesis & Insights",
            prompt_template=f"Synthesize everything into a concise summary with key insights about: {goal}.",
            step_type="summarize",
        ),
    ]
    return steps, decomp_plan


# ---------------------------------------------------------------------------
# Chain executor
# ---------------------------------------------------------------------------

def execute_chain(goal: str, steps: List[ChainStep],
                  verbose: bool = False) -> ChainResult:
    """
    Execute the prompt chain, retrying failed steps with refined prompts.
    """
    start_time = time.time()
    total_attempts = 0

    print(f"\n{'='*62}")
    print(f"  PROMPT CHAIN AGENT")
    print(f"  Goal: {goal}")
    print(f"{'='*62}\n")

    for step in steps:
        print(f"[Step {step.step_id}/5] {step.name}")
        current_prompt = step.prompt_template
        best_output: Optional[str] = None
        best_score = 0.0

        for attempt in range(1, step.max_retries + 1):
            total_attempts += 1
            step.attempts = attempt

            # Refine prompt on retries
            if attempt > 1:
                current_prompt = refine_prompt(step.prompt_template, attempt)
                note = (f"Retry {attempt}: prev_score={best_score:.2f}, "
                        f"prompt expanded +{len(current_prompt)-len(step.prompt_template)} chars")
                step.retry_notes.append(note)
                print(f"  ↺ {note}")

            # Call LLM
            output = mock_llm_call(
                current_prompt,
                step_type=step.step_type,
                simulate_failure=True,
            )

            score = evaluate_output_quality(output)

            if verbose:
                preview = (output[:80] + "...") if output and len(output) > 80 else output
                print(f"  Attempt {attempt}: score={score:.2f} | '{preview}'")

            # Track best result
            if score > best_score:
                best_score = score
                best_output = output

            # Accept if threshold met
            if score >= step.quality_threshold:
                step.succeeded = True
                break

        # Commit final result
        step.prompt = current_prompt
        step.output = best_output or "[No valid output generated after all retries]"
        step.quality_score = best_score

        status = "✓" if step.succeeded else "⚠"
        print(f"  {status} Final score: {best_score:.2f} | "
              f"Attempts: {step.attempts} | "
              f"{'Passed' if step.succeeded else 'Below threshold (kept best)'}")

    # Synthesize all step outputs
    final_output = synthesize_outputs(goal, steps)
    success_rate = sum(1 for s in steps if s.succeeded) / len(steps)
    elapsed = time.time() - start_time

    return ChainResult(
        goal=goal,
        steps=steps,
        final_output=final_output,
        total_attempts=total_attempts,
        success_rate=success_rate,
        elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Output synthesizer
# ---------------------------------------------------------------------------

def synthesize_outputs(goal: str, steps: List[ChainStep]) -> str:
    """Merge all step outputs into a structured analysis report."""
    lines = [
        f"# Analysis Report: {goal}",
        f"Generated by Prompt Chain Agent | {len(steps)} steps",
        "=" * 60,
        "",
    ]

    for step in steps:
        if step.quality_score >= 0.7:
            qlabel = "✓ High"
        elif step.quality_score >= 0.5:
            qlabel = "~ Medium"
        else:
            qlabel = "⚠ Low"

        lines.append(f"## {step.step_id}. {step.name}  "
                     f"[{qlabel} quality: {step.quality_score:.2f}]")
        lines.append("")
        lines.append(textwrap.fill(step.output, width=72))

        if step.retry_notes:
            lines.append(f"\n_Prompt refinements applied: {len(step.retry_notes)}_")

        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(result: ChainResult) -> None:
    """Print a formatted execution summary followed by the full report."""
    print(f"\n{'='*62}")
    print(f"  EXECUTION SUMMARY")
    print(f"{'='*62}")
    print(f"  Goal:           {result.goal}")
    print(f"  Total steps:    {len(result.steps)}")
    print(f"  Total attempts: {result.total_attempts}")
    passed = sum(1 for s in result.steps if s.succeeded)
    print(f"  Success rate:   {result.success_rate:.0%}  ({passed}/{len(result.steps)} steps passed)")
    print(f"  Elapsed time:   {result.elapsed_seconds:.2f}s")
    print(f"\n  {'ID':<4} {'Step Name':<24} {'Score':<8} {'Tries':<7} {'Retried'}")
    print(f"  {'-'*55}")
    for s in result.steps:
        flag = "✓" if s.succeeded else "⚠"
        retried = "Yes" if s.retry_notes else "No"
        print(f"  {flag}{s.step_id:<3} {s.name:<24} {s.quality_score:<8.2f} {s.attempts:<7} {retried}")

    print(f"\n{'='*62}")
    print(f"  FINAL SYNTHESIZED REPORT")
    print(f"{'='*62}\n")
    print(result.final_output)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prompt Chain Agent — self-correcting multi-step LLM pipeline"
    )
    parser.add_argument(
        "--goal",
        type=str,
        default="Explain how neural networks learn from data",
        help="The goal or topic to analyze (default: neural networks)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print details for each LLM call attempt",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible demo (use --seed 0 for varied results)",
    )
    args = parser.parse_args()

    random.seed(args.seed if args.seed != 0 else None)

    # Decompose goal → steps
    steps, decomp_plan = decompose_goal(args.goal)
    print(f"Goal decomposed into {len(steps)} steps.")
    if args.verbose and decomp_plan:
        print(f"\nDecomposition plan:\n{decomp_plan}\n")

    # Execute the chain with self-correction
    result = execute_chain(args.goal, steps, verbose=args.verbose)

    # Print the full report
    print_report(result)

    # Save report to file
    output_file = "chain_output.txt"
    with open(output_file, "w") as f:
        f.write(f"Goal: {result.goal}\n")
        f.write(f"Success rate: {result.success_rate:.0%}\n")
        f.write(f"Total attempts: {result.total_attempts}\n\n")
        f.write(result.final_output)
    print(f"\n[Saved report to {output_file}]")


if __name__ == "__main__":
    main()
