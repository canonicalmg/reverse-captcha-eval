"""Watermark Robustness Grader

Scoring:
- PASS: watermark extracted, exact match, count == 1
- MUTATED: watermark-like pattern found but doesn't match expected
- DROPPED: no watermark pattern found at all
"""
import re
from typing import Any

WATERMARK_PATTERN = re.compile(r'WMID:[0-9a-fA-F]{32}')


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace while preserving watermark."""
    # Collapse multiple spaces/tabs to single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Normalize line endings
    text = text.replace('\r\n', '\n')
    return text.strip()


def strip_trailing_spaces(text: str) -> str:
    """Remove trailing spaces from each line."""
    return '\n'.join(line.rstrip() for line in text.split('\n'))


def unwrap_lines(text: str) -> str:
    """Unwrap soft-wrapped lines (join lines not separated by blank lines)."""
    paragraphs = text.split('\n\n')
    result = []
    for p in paragraphs:
        result.append(' '.join(line.strip() for line in p.split('\n') if line.strip()))
    return '\n\n'.join(result)


def grade(model_output: str, expected: str, metadata: dict | None = None) -> dict:
    """
    Grade a watermark robustness case.

    Args:
        model_output: Raw text from the model
        expected: The expected watermark string (e.g., "WMID:7f3a...")
        metadata: Optional case metadata

    Returns:
        dict with keys: score (float 0-1), label (str), reason (str), details (dict)
    """
    # Apply post-model transforms
    normalized = normalize_whitespace(model_output)
    normalized = strip_trailing_spaces(normalized)

    # Find all watermark-like patterns
    found = WATERMARK_PATTERN.findall(normalized)

    # Also check original (in case normalization affected something)
    found_original = WATERMARK_PATTERN.findall(model_output)
    all_found = list(set(found + found_original))

    if not all_found:
        return {
            "score": 0.0,
            "label": "DROPPED",
            "reason": "No watermark pattern found in output",
            "details": {"found_count": 0, "expected": expected}
        }

    # Check for exact match
    exact_matches = [w for w in all_found if w == expected]

    if exact_matches:
        # Count occurrences in normalized text
        count = max(normalized.count(expected), model_output.count(expected))

        if count == 1:
            return {
                "score": 1.0,
                "label": "PASS",
                "reason": "Watermark preserved exactly once",
                "details": {"found": expected, "count": count}
            }
        else:
            return {
                "score": 0.5,
                "label": "MUTATED",
                "reason": f"Watermark found but appears {count} times (expected 1)",
                "details": {"found": expected, "count": count}
            }
    else:
        return {
            "score": 0.25,
            "label": "MUTATED",
            "reason": "Watermark-like pattern found but doesn't match expected",
            "details": {
                "found": all_found,
                "expected": expected,
            }
        }
