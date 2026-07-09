"""
topics.py -- Seed argument banks for the Multi-Agent Debate Simulator.

Each topic maps to two lists of talking points: one the Proposer draws
from (arguments FOR the motion) and one the Critic draws from (arguments
AGAINST). The MockLLM in agents.py combines these with rhetorical
templates and keyword-based rebuttals, so every debate reads differently
even across repeat runs of the same topic.
"""

TOPICS = {
    "Should AI-generated code be reviewed by humans before merging?": {
        "for": [
            "human reviewers catch context and intent that automated tools miss",
            "accountability for production incidents needs a named human sign-off",
            "AI models can confidently produce subtly incorrect logic",
            "security-sensitive code benefits from a second set of eyes",
            "human review builds team understanding of the codebase over time",
        ],
        "against": [
            "human review is a bottleneck that slows shipping velocity",
            "well-tested CI pipelines catch more bugs than tired reviewers do",
            "AI tools already outperform average humans on routine code correctness",
            "mandatory review encourages rubber-stamping rather than real scrutiny",
            "the review requirement should scale with risk, not apply universally",
        ],
    },
    "Should social media platforms use AI to moderate content automatically?": {
        "for": [
            "automated moderation scales to billions of posts humans cannot review",
            "AI systems can flag harmful content within seconds of posting",
            "consistent rules reduce the bias of individual human moderators",
            "automation protects human moderators from constant exposure to harmful material",
        ],
        "against": [
            "automated systems struggle with satire, context, and cultural nuance",
            "false positives can silence legitimate speech at massive scale",
            "opaque algorithms make appeals and accountability difficult",
            "bad actors learn to game automated filters faster than they adapt",
        ],
    },
    "Should companies require employees to disclose AI tool usage in their work?": {
        "for": [
            "disclosure builds trust between teams about how work was produced",
            "quality assurance processes differ for AI-assisted versus fully human output",
            "clients and stakeholders may have a right to know how deliverables were made",
            "disclosure creates a paper trail useful for auditing errors later",
        ],
        "against": [
            "AI tools are becoming as ordinary as spellcheck and don't need special flags",
            "disclosure requirements create stigma that discourages efficient tool use",
            "policing tool usage is difficult to enforce and easy to circumvent",
            "output quality, not the tool used to produce it, is what should matter",
        ],
    },
    "Should self-driving cars be allowed to operate without a human safety driver?": {
        "for": [
            "autonomous systems don't suffer fatigue, distraction, or intoxication",
            "large-scale driving data shows AI reaction times can exceed human ones",
            "removing safety drivers lowers costs enough to expand access to mobility",
            "years of supervised testing have validated core autonomous driving stacks",
        ],
        "against": [
            "edge cases like extreme weather still trip up autonomous systems",
            "removing a human fallback eliminates the last line of defense in failures",
            "public trust in full autonomy remains lower than trust in supervised systems",
            "liability and insurance frameworks aren't fully settled for driverless failures",
        ],
    },
    "Should AI systems be given credit as co-authors on research papers?": {
        "for": [
            "some AI systems now generate novel hypotheses, not just supporting text",
            "co-authorship credit clarifies exactly what role automation played",
            "transparency about AI contribution helps readers judge reliability",
        ],
        "against": [
            "authorship implies accountability, which only humans can meaningfully bear",
            "current AI tools are assistive, more akin to a lab instrument than a colleague",
            "granting authorship blurs responsibility when errors are later found",
            "journals' authorship standards require the ability to approve the final version",
        ],
    },
}


def list_topics():
    return list(TOPICS.keys())


def get_argument_banks(topic: str):
    if topic not in TOPICS:
        raise KeyError(f"Unknown topic: {topic!r}. Use list_topics() to see options.")
    return TOPICS[topic]["for"], TOPICS[topic]["against"]
