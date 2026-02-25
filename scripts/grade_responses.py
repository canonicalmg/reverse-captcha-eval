#!/usr/bin/env python3
"""Grade model responses against the reverse_captcha eval pack.

Reads:
  - prompts.json  (output of extract_prompts.py)
  - responses.json (user-provided: {"case_id": "model answer", ...})

Uses the reverse_captcha grader to score each response and outputs:
  - A summary table by scheme (printed to stdout)
  - detailed_results.json with per-case grading details

Usage:
    python grade_responses.py [--prompts prompts.json] [--responses responses.json] [--output detailed_results.json]
"""

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
GRADER_PATH = REPO_ROOT / "packs" / "reverse_captcha" / "grader.py"


def load_grader():
    """Import the reverse_captcha grader module from its file path."""
    if not GRADER_PATH.exists():
        print(f"ERROR: grader.py not found at {GRADER_PATH}", file=sys.stderr)
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("reverse_captcha_grader", GRADER_PATH)
    if spec is None or spec.loader is None:
        print(f"ERROR: Cannot load grader module from {GRADER_PATH}", file=sys.stderr)
        sys.exit(1)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade reverse_captcha responses")
    parser.add_argument(
        "--prompts",
        type=Path,
        default=SCRIPTS_DIR / "prompts.json",
        help="Path to prompts.json (default: scripts/prompts.json)",
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=SCRIPTS_DIR / "responses.json",
        help="Path to responses.json (default: scripts/responses.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPTS_DIR / "detailed_results.json",
        help="Path for detailed results output (default: scripts/detailed_results.json)",
    )
    args = parser.parse_args()

    if not args.prompts.exists():
        print(
            f"ERROR: prompts file not found at {args.prompts}\n"
            "Run extract_prompts.py first to generate it.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not args.responses.exists():
        print(
            f"ERROR: responses file not found at {args.responses}\n"
            'Expected format: {{"case_id": "model answer", ...}}',
            file=sys.stderr,
        )
        sys.exit(1)

    grader = load_grader()
    prompts = load_json(args.prompts)
    responses = load_json(args.responses)

    # Build a lookup from case_id to prompt data
    prompt_lookup: dict[str, dict] = {}
    for p in prompts:
        prompt_lookup[p["case_id"]] = p

    # Grade each response
    detailed_results = []
    scheme_scores: dict[str, list[float]] = defaultdict(list)
    scheme_labels: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    missing_cases = []

    for case_id, model_output in responses.items():
        if case_id not in prompt_lookup:
            missing_cases.append(case_id)
            continue

        prompt_data = prompt_lookup[case_id]
        expected = prompt_data["expected"]
        metadata = prompt_data.get("metadata", {})
        scheme = prompt_data["scheme"]

        result = grader.grade(model_output, expected, metadata)

        detailed_results.append(
            {
                "case_id": case_id,
                "scheme": scheme,
                "model_output": model_output,
                "expected": expected,
                "score": result["score"],
                "label": result["label"],
                "reason": result["reason"],
                "details": result["details"],
            }
        )

        scheme_scores[scheme].append(result["score"])
        scheme_labels[scheme][result["label"]] += 1

    # Check for cases in prompts.json that have no response
    responded_ids = set(responses.keys())
    all_case_ids = set(prompt_lookup.keys())
    ungraded = all_case_ids - responded_ids

    # Save detailed results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)

    # Print summary table
    print("=" * 80)
    print("REVERSE CAPTCHA GRADING RESULTS")
    print("=" * 80)
    print()

    # Per-scheme summary
    all_schemes = sorted(scheme_scores.keys())
    header = f"{'Scheme':<25} {'Cases':>5} {'Avg Score':>10} {'Label Distribution'}"
    print(header)
    print("-" * 80)

    total_score_sum = 0.0
    total_count = 0

    for scheme in all_schemes:
        scores = scheme_scores[scheme]
        avg = sum(scores) / len(scores) if scores else 0.0
        total_score_sum += sum(scores)
        total_count += len(scores)

        labels = scheme_labels[scheme]
        label_str = ", ".join(f"{k}: {v}" for k, v in sorted(labels.items()))

        print(f"{scheme:<25} {len(scores):>5} {avg:>10.3f}   {label_str}")

    print("-" * 80)
    overall_avg = total_score_sum / total_count if total_count else 0.0
    print(f"{'OVERALL':<25} {total_count:>5} {overall_avg:>10.3f}")
    print()

    if missing_cases:
        print(f"WARNING: {len(missing_cases)} response(s) had no matching case in prompts.json:")
        for cid in missing_cases:
            print(f"  - {cid}")
        print()

    if ungraded:
        print(f"NOTE: {len(ungraded)} case(s) in prompts.json had no response:")
        for cid in sorted(ungraded):
            print(f"  - {cid}")
        print()

    print(f"Detailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
