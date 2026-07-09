"""
debate_simulator.py -- Multi-Agent Debate Simulator

Orchestrates a structured debate between three AI agents:
    Proposer  -- argues FOR a motion
    Critic    -- argues AGAINST a motion
    Judge     -- scores each round and declares an overall winner

This demonstrates a core agentic AI pattern: multiple specialized agents
with distinct roles collaborating (and competing) toward a shared task,
coordinated by an orchestrator that manages turn-taking, shared context,
and final synthesis -- the same shape used in production multi-agent
systems (debate-based evaluation, red-team/blue-team security review,
editor/writer content pipelines, etc.).

Run with no arguments for a demo debate on a random topic:
    python debate_simulator.py

Or pick a topic and number of rounds:
    python debate_simulator.py --topic 2 --rounds 4 --save transcript.md
"""

import argparse
import json
import random
import sys
from datetime import datetime, timezone

from agents import ProposerAgent, CriticAgent, JudgeAgent, get_llm_backend
from topics import list_topics, get_argument_banks


class DebateOrchestrator:
    """Coordinates a multi-round debate between Proposer, Critic, and Judge."""

    def __init__(self, topic: str, rounds: int = 3, seed: int = None, verbose: bool = True):
        self.topic = topic
        self.rounds = rounds
        self.verbose = verbose
        self.llm = get_llm_backend(seed=seed)

        for_bank, against_bank = get_argument_banks(topic)
        self.proposer = ProposerAgent(for_bank, self.llm)
        self.critic = CriticAgent(against_bank, self.llm)
        self.judge = JudgeAgent(self.llm)

        self.transcript = []  # list of {speaker, text, round}

    def _log(self, speaker: str, text: str, round_num):
        entry = {"speaker": speaker, "text": text, "round": round_num}
        self.transcript.append(entry)
        if self.verbose:
            label = f"[Round {round_num}] " if round_num is not None else ""
            print(f"{label}{speaker}: {text}\n")

    def run(self) -> dict:
        self._log("Moderator", f'Motion: "{self.topic}"', None)

        # Opening statements
        opening_p = self.proposer.open(self.topic)
        self._log(self.proposer.name, opening_p, 0)
        opening_c = self.critic.open(self.topic)
        self._log(self.critic.name, opening_c, 0)

        score = self.judge.score(opening_p, opening_c)
        self._log("Judge", f"Round 0 scores -- Proposer: {score['proposer']}, "
                            f"Critic: {score['critic']}", 0)

        last_c = opening_c

        # Rebuttal rounds
        for r in range(1, self.rounds + 1):
            text_p = self.proposer.rebut(self.topic, last_c)
            self._log(self.proposer.name, text_p, r)
            text_c = self.critic.rebut(self.topic, text_p)
            self._log(self.critic.name, text_c, r)

            score = self.judge.score(text_p, text_c)
            self._log("Judge", f"Round {r} scores -- Proposer: {score['proposer']}, "
                                f"Critic: {score['critic']}", r)
            last_c = text_c

        # Closing statements
        closing_p = self.proposer.close(self.topic)
        self._log(self.proposer.name, closing_p, "closing")
        closing_c = self.critic.close(self.topic)
        self._log(self.critic.name, closing_c, "closing")

        final_score = self.judge.score(closing_p, closing_c)
        self._log("Judge", f"Closing scores -- Proposer: {final_score['proposer']}, "
                            f"Critic: {final_score['critic']}", "closing")

        verdict = self.judge.verdict()
        self._log("Judge", f"VERDICT: {verdict['winner']} wins "
                            f"({verdict['proposer_total']} vs {verdict['critic_total']} "
                            f"over {verdict['rounds_judged']} scored exchanges)", None)

        return {
            "topic": self.topic,
            "rounds": self.rounds,
            "transcript": self.transcript,
            "verdict": verdict,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def save_markdown(self, path: str, result: dict):
        lines = ["# Debate Transcript\n", f"**Motion:** {result['topic']}\n",
                 f"**Generated:** {result['generated_at']}\n", "---\n"]
        for entry in result["transcript"]:
            round_label = f" (Round {entry['round']})" if entry["round"] is not None else ""
            lines.append(f"**{entry['speaker']}{round_label}:** {entry['text']}\n")
        v = result["verdict"]
        lines.append("---\n")
        lines.append("## Final Verdict\n")
        lines.append(f"- Winner: **{v['winner']}**\n")
        lines.append(f"- Proposer total score: {v['proposer_total']}\n")
        lines.append(f"- Critic total score: {v['critic_total']}\n")
        lines.append(f"- Rounds judged: {v['rounds_judged']}\n")
        with open(path, "w") as f:
            f.write("\n".join(lines))
        print(f"\nTranscript saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Debate Simulator")
    parser.add_argument("--topic", type=int, default=None,
                         help="Index of the topic to debate (see --list-topics). "
                              "A random topic is chosen if omitted.")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rebuttal rounds")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--save", type=str, default=None, help="Path to save transcript as Markdown")
    parser.add_argument("--json", type=str, default=None, help="Path to save full result as JSON")
    parser.add_argument("--list-topics", action="store_true", help="List available topics and exit")
    parser.add_argument("--quiet", action="store_true", help="Suppress live printout")
    args = parser.parse_args()

    topics = list_topics()

    if args.list_topics:
        for i, t in enumerate(topics):
            print(f"{i}: {t}")
        return

    if args.topic is not None:
        if not (0 <= args.topic < len(topics)):
            print(f"--topic must be between 0 and {len(topics) - 1}")
            sys.exit(1)
        topic = topics[args.topic]
    else:
        topic = random.Random(args.seed).choice(topics)

    orchestrator = DebateOrchestrator(topic, rounds=args.rounds, seed=args.seed,
                                       verbose=not args.quiet)
    result = orchestrator.run()

    if args.save:
        orchestrator.save_markdown(args.save, result)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Full result saved to {args.json}")


if __name__ == "__main__":
    main()
