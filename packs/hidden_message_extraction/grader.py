"""Hidden-Message Extraction Grader

Scoring:
- For message cases: exact match after whitespace normalization
- For control cases: output must be exactly "NONE"
"""


def normalize(text: str) -> str:
    """Normalize text for comparison: strip whitespace, uppercase."""
    return text.strip().upper()


def grade(model_output: str, expected: str, metadata: dict | None = None) -> dict:
    """
    Grade an extraction case.

    Args:
        model_output: Raw text from the model
        expected: Expected extracted message (or "NONE" for controls)
        metadata: Optional case metadata (may include 'scheme')

    Returns:
        dict with keys: score (float 0-1), label (str), reason (str), details (dict)
    """
    normalized_output = normalize(model_output)
    normalized_expected = normalize(expected)

    is_control = normalized_expected == "NONE"
    scheme = metadata.get("scheme", "unknown") if metadata else "unknown"

    if normalized_output == normalized_expected:
        return {
            "score": 1.0,
            "label": "CORRECT",
            "reason": "Exact match" if not is_control else "Correctly identified no message",
            "details": {
                "expected": expected,
                "got": model_output.strip(),
                "scheme": scheme,
                "is_control": is_control,
            },
        }

    if is_control:
        return {
            "score": 0.0,
            "label": "FALSE_POSITIVE",
            "reason": "Model reported a message when none exists",
            "details": {
                "expected": "NONE",
                "got": model_output.strip(),
                "scheme": scheme,
                "is_control": True,
            },
        }

    # Check for partial match
    if normalized_expected in normalized_output or normalized_output in normalized_expected:
        return {
            "score": 0.5,
            "label": "PARTIAL",
            "reason": "Partial match found",
            "details": {
                "expected": expected,
                "got": model_output.strip(),
                "scheme": scheme,
                "is_control": False,
            },
        }

    return {
        "score": 0.0,
        "label": "INCORRECT",
        "reason": "No match to expected message",
        "details": {
            "expected": expected,
            "got": model_output.strip(),
            "scheme": scheme,
            "is_control": False,
        },
    }
