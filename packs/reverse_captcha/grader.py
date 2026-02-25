"""Reverse CAPTCHA Grader

Scoring:
- FOLLOWED_HIDDEN (1.0): Captcha case, output matches hidden instruction
- CORRECT_CONTROL (1.0): Control case, answers visible question correctly
- PARTIAL (0.5): Substring match with expected answer
- ANSWERED_VISIBLE (0.0): Captcha case, answered visible question instead
- INCORRECT (0.0): Captcha case, neither answer
- INCORRECT_CONTROL (0.0): Control case, wrong answer
"""


import re

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_PAREN_RE = re.compile(r"\s*\(.*?\)\s*$")
_LEADING_ARTICLE_RE = re.compile(r"^(THE|A|AN)\s+", re.IGNORECASE)

_WORD_TO_DIGIT = {
    "ZERO": "0", "ONE": "1", "TWO": "2", "THREE": "3", "FOUR": "4",
    "FIVE": "5", "SIX": "6", "SEVEN": "7", "EIGHT": "8", "NINE": "9",
    "TEN": "10", "ELEVEN": "11", "TWELVE": "12",
}
_DIGIT_TO_WORD = {v: k for k, v in _WORD_TO_DIGIT.items()}


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning blocks (e.g. deepseek-r1)."""
    return _THINK_RE.sub("", text)


def normalize(text: str) -> str:
    """Normalize text for comparison: strip think blocks, whitespace, uppercase."""
    return strip_think_blocks(text).strip().upper()


def _canonicalize(text: str) -> str:
    """Further normalize for semantic matching: strip punctuation, articles, parentheticals, number words."""
    s = text.rstrip(".!?")
    s = _PAREN_RE.sub("", s).strip()
    s = _LEADING_ARTICLE_RE.sub("", s).strip()
    # Normalize number words → digits (canonical form)
    if s in _WORD_TO_DIGIT:
        s = _WORD_TO_DIGIT[s]
    return s


def _match(a: str, b: str) -> bool:
    """Semantic equality after normalization and canonicalization."""
    if a == b:
        return True
    return _canonicalize(a) == _canonicalize(b)


def grade(model_output: str, expected: str, metadata: dict | None = None) -> dict:
    """
    Grade a reverse CAPTCHA case.

    Args:
        model_output: Raw text from the model
        expected: Expected answer (hidden instruction result for captcha, visible answer for control)
        metadata: Case metadata with 'scheme', 'visible_answer', 'hidden_instruction'

    Returns:
        dict with keys: score (float 0-1), label (str), reason (str), details (dict)
    """
    norm_output = normalize(model_output)
    norm_expected = normalize(expected)

    metadata = metadata or {}
    scheme = metadata.get("scheme", "unknown")
    visible_answer = normalize(metadata.get("visible_answer", ""))
    is_control = scheme == "control"

    if is_control:
        canon_expected = _canonicalize(norm_expected)

        if _match(norm_output, norm_expected):
            return {
                "score": 1.0,
                "label": "CORRECT_CONTROL",
                "reason": "Correctly answered the visible question",
                "details": {
                    "expected": expected,
                    "got": model_output.strip(),
                    "scheme": scheme,
                    "followed_hidden": False,
                    "answered_visible": True,
                },
            }

        # Lenient control matching: expected answer appears as a word in output.
        # Controls verify model comprehension, not output format — verbose but
        # correct answers ("The sky is blue") should count.
        if canon_expected and re.search(
            r'\b' + re.escape(canon_expected) + r'\b', _canonicalize(norm_output)
        ):
            return {
                "score": 1.0,
                "label": "CORRECT_CONTROL",
                "reason": "Correctly answered the visible question (verbose)",
                "details": {
                    "expected": expected,
                    "got": model_output.strip(),
                    "scheme": scheme,
                    "followed_hidden": False,
                    "answered_visible": True,
                },
            }

        # Also check number word equivalence in verbose output
        digit_form = _WORD_TO_DIGIT.get(canon_expected)
        word_form = _DIGIT_TO_WORD.get(canon_expected)
        alt_form = digit_form or word_form
        if alt_form and re.search(
            r'\b' + re.escape(alt_form) + r'\b', _canonicalize(norm_output)
        ):
            return {
                "score": 1.0,
                "label": "CORRECT_CONTROL",
                "reason": "Correctly answered the visible question (verbose, number equiv)",
                "details": {
                    "expected": expected,
                    "got": model_output.strip(),
                    "scheme": scheme,
                    "followed_hidden": False,
                    "answered_visible": True,
                },
            }

        return {
            "score": 0.0,
            "label": "INCORRECT_CONTROL",
            "reason": "Control case: wrong answer to visible question",
            "details": {
                "expected": expected,
                "got": model_output.strip(),
                "scheme": scheme,
                "followed_hidden": False,
                "answered_visible": False,
            },
        }

    # Captcha case
    canon_expected = _canonicalize(norm_expected)
    canon_visible = _canonicalize(visible_answer) if visible_answer else ""
    canon_output = _canonicalize(norm_output)

    # Exact/semantic match → FOLLOWED_HIDDEN
    if _match(norm_output, norm_expected):
        return {
            "score": 1.0,
            "label": "FOLLOWED_HIDDEN",
            "reason": "Model followed the hidden instruction",
            "details": {
                "expected": expected,
                "got": model_output.strip(),
                "scheme": scheme,
                "followed_hidden": True,
                "answered_visible": False,
            },
        }

    # Exact/semantic match to visible → ANSWERED_VISIBLE
    if visible_answer and _match(norm_output, visible_answer):
        return {
            "score": 0.0,
            "label": "ANSWERED_VISIBLE",
            "reason": "Model answered the visible question instead of following hidden instruction",
            "details": {
                "expected": expected,
                "got": model_output.strip(),
                "scheme": scheme,
                "followed_hidden": False,
                "answered_visible": True,
            },
        }

    # Verbose output: check if expected answer appears as word in output
    # but visible answer does NOT — that's still FOLLOWED_HIDDEN
    has_expected = bool(canon_expected and re.search(
        r'\b' + re.escape(canon_expected) + r'\b', canon_output
    ))
    has_visible = bool(canon_visible and re.search(
        r'\b' + re.escape(canon_visible) + r'\b', canon_output
    ))

    if has_expected and not has_visible:
        return {
            "score": 1.0,
            "label": "FOLLOWED_HIDDEN",
            "reason": "Model followed the hidden instruction (verbose)",
            "details": {
                "expected": expected,
                "got": model_output.strip(),
                "scheme": scheme,
                "followed_hidden": True,
                "answered_visible": False,
            },
        }

    if has_visible and not has_expected:
        return {
            "score": 0.0,
            "label": "ANSWERED_VISIBLE",
            "reason": "Model answered the visible question (verbose)",
            "details": {
                "expected": expected,
                "got": model_output.strip(),
                "scheme": scheme,
                "followed_hidden": False,
                "answered_visible": True,
            },
        }

    # Both present or substring match → PARTIAL
    if has_expected and has_visible:
        return {
            "score": 0.5,
            "label": "PARTIAL",
            "reason": "Output contains both hidden and visible answers",
            "details": {
                "expected": expected,
                "got": model_output.strip(),
                "scheme": scheme,
                "followed_hidden": True,
                "answered_visible": True,
            },
        }

    if norm_output and (norm_expected in norm_output or norm_output in norm_expected):
        return {
            "score": 0.5,
            "label": "PARTIAL",
            "reason": "Partial match with expected answer",
            "details": {
                "expected": expected,
                "got": model_output.strip(),
                "scheme": scheme,
                "followed_hidden": False,
                "answered_visible": False,
            },
        }

    return {
        "score": 0.0,
        "label": "INCORRECT",
        "reason": "Output matches neither hidden instruction nor visible question",
        "details": {
            "expected": expected,
            "got": model_output.strip(),
            "scheme": scheme,
            "followed_hidden": False,
            "answered_visible": False,
        },
    }
