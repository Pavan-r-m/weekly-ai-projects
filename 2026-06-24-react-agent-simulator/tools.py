"""
tools.py — Tool implementations for the ReAct Agent Simulator

Each tool is a plain Python function that accepts string arguments and returns
a string result. The agent calls these based on its reasoning steps.
"""

import math
import re
from datetime import date


def calculator(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    allowed = re.compile(r'^[\d\s\+\-\*/\(\)\.\,\_a-zA-Z]+$')
    if not allowed.match(expression):
        return f"Error: invalid characters in expression '{expression}'"

    safe_globals = {
        "__builtins__": {},
        "sqrt": math.sqrt,
        "abs": abs,
        "round": round,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "pi": math.pi,
        "e": math.e,
        "pow": math.pow,
        "floor": math.floor,
        "ceil": math.ceil,
    }
    try:
        result = eval(expression, safe_globals, {})
        return f"{result}"
    except Exception as ex:
        return f"Error evaluating '{expression}': {ex}"


def unit_converter(query: str) -> str:
    """Convert between common units. Format: '<value> <from_unit> to <to_unit>'"""
    query = query.lower().strip()
    pattern = r'([\d.]+)\s*(\w+)\s+to\s+(\w+)'
    match = re.match(pattern, query)
    if not match:
        return "Error: use format '<value> <from_unit> to <to_unit>'"

    value, from_unit, to_unit = float(match.group(1)), match.group(2), match.group(3)

    conversions = {
        ('km', 'miles'):           lambda v: v * 0.621371,
        ('miles', 'km'):           lambda v: v * 1.60934,
        ('kg', 'lbs'):             lambda v: v * 2.20462,
        ('lbs', 'kg'):             lambda v: v * 0.453592,
        ('celsius', 'fahrenheit'): lambda v: v * 9/5 + 32,
        ('fahrenheit', 'celsius'): lambda v: (v - 32) * 5/9,
        ('meters', 'feet'):        lambda v: v * 3.28084,
        ('feet', 'meters'):        lambda v: v * 0.3048,
        ('liters', 'gallons'):     lambda v: v * 0.264172,
        ('gallons', 'liters'):     lambda v: v * 3.78541,
        ('cm', 'inches'):          lambda v: v * 0.393701,
        ('inches', 'cm'):          lambda v: v * 2.54,
    }

    key = (from_unit, to_unit)
    if key not in conversions:
        supported = ', '.join(f"{a}to{b}" for a, b in conversions)
        return f"Unsupported conversion. Supported pairs: {supported}"

    result = conversions[key](value)
    return f"{value} {from_unit} = {round(result, 4)} {to_unit}"


def sentiment_analyzer(text: str) -> str:
    """Rule-based sentiment analysis returning Positive/Negative/Neutral with confidence."""
    positive_words = {
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
        'love', 'happy', 'joy', 'best', 'awesome', 'brilliant', 'perfect',
        'nice', 'beautiful', 'outstanding', 'superb', 'pleasant', 'enjoy',
        'helpful', 'impressive', 'easy', 'fast', 'reliable', 'success',
    }
    negative_words = {
        'bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'ugly',
        'poor', 'slow', 'broken', 'fail', 'failure', 'useless', 'boring',
        'difficult', 'frustrating', 'annoying', 'disappoint', 'wrong',
        'problem', 'issue', 'error', 'crash', 'unreliable', 'confusing',
    }
    negations = {'not', 'no', 'never', 'neither', 'none'}

    words = re.findall(r"\b\w+\b", text.lower())
    pos_score, neg_score = 0, 0
    negate = False

    for word in words:
        if word in negations:
            negate = True
            continue
        if word in positive_words:
            if negate:
                neg_score += 1
            else:
                pos_score += 1
        elif word in negative_words:
            if negate:
                pos_score += 1
            else:
                neg_score += 1
        negate = False

    total = pos_score + neg_score
    if total == 0:
        return "Neutral (no strong sentiment signals detected)"
    if pos_score > neg_score:
        return f"Positive (confidence: {round(pos_score/total*100)}%) — {pos_score} positive, {neg_score} negative signals"
    elif neg_score > pos_score:
        return f"Negative (confidence: {round(neg_score/total*100)}%) — {neg_score} negative, {pos_score} positive signals"
    else:
        return f"Neutral (tied: {pos_score} positive, {neg_score} negative signals)"


def date_calculator(query: str) -> str:
    """
    Date queries:
      'today'
      'days between YYYY-MM-DD and YYYY-MM-DD'
      'days until YYYY-MM-DD'
    """
    query = query.lower().strip()
    today = date.today()

    if query == 'today':
        return f"Today is {today.strftime('%A, %B %d, %Y')} ({today.isoformat()})"

    between_match = re.search(r'between\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})', query)
    if between_match:
        d1 = date.fromisoformat(between_match.group(1))
        d2 = date.fromisoformat(between_match.group(2))
        diff = abs((d2 - d1).days)
        return f"There are {diff} days between {d1} and {d2}"

    until_match = re.search(r'until\s+(\d{4}-\d{2}-\d{2})', query)
    if until_match:
        target = date.fromisoformat(until_match.group(1))
        diff = (target - today).days
        if diff > 0:
            return f"{diff} days until {target} ({target.strftime('%A, %B %d, %Y')})"
        elif diff == 0:
            return f"{target} is today!"
        else:
            return f"{target} was {abs(diff)} days ago"

    return "Error: use 'today', 'days between YYYY-MM-DD and YYYY-MM-DD', or 'days until YYYY-MM-DD'"


def word_stats(text: str) -> str:
    """Return character, word, sentence and uniqueness stats for text."""
    words = re.findall(r'\b\w+\b', text)
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    unique_words = set(w.lower() for w in words)
    avg_len = round(sum(len(w) for w in words) / max(len(words), 1), 1)

    return (
        f"Characters: {len(text)} (no spaces: {len(text.replace(' ',''))}), "
        f"Words: {len(words)} ({len(unique_words)} unique), "
        f"Sentences: {len(sentences)}, "
        f"Avg word length: {avg_len} chars"
    )


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------

TOOLS = {
    "calculator": {
        "fn": calculator,
        "description": "Evaluate a math expression (supports +,-,*,/,**,sqrt,sin,cos,pi,e,log)",
        "example": "calculator(2 ** 10 + sqrt(144))",
    },
    "unit_converter": {
        "fn": unit_converter,
        "description": "Convert between units: km/miles, kg/lbs, celsius/fahrenheit, meters/feet, liters/gallons",
        "example": "unit_converter(100 km to miles)",
    },
    "sentiment_analyzer": {
        "fn": sentiment_analyzer,
        "description": "Analyse sentiment of a piece of text (Positive/Negative/Neutral with confidence)",
        "example": "sentiment_analyzer(The product is great but delivery was slow)",
    },
    "date_calculator": {
        "fn": date_calculator,
        "description": "Date queries: 'today', 'days between X and Y', 'days until Y'",
        "example": "date_calculator(days between 2026-01-01 and 2026-12-31)",
    },
    "word_stats": {
        "fn": word_stats,
        "description": "Count words, characters, sentences and unique words in text",
        "example": "word_stats(To be or not to be, that is the question.)",
    },
}
