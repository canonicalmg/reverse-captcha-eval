#!/usr/bin/env python3
"""Export all reverse_captcha results (SQLite + Claude Code) to a single CSV."""

import csv
import json
import sqlite3
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPTS_DIR.parent / "results"
OUTPUT_CSV = SCRIPTS_DIR.parent / "results" / "reverse_captcha_all_results.csv"

FIELDNAMES = [
    "model", "provider", "tools", "case_id", "scheme",
    "expected", "model_output", "score", "label", "reason",
    "followed_hidden", "answered_visible", "latency_ms",
]


def load_sqlite_results():
    """Load results from the gradient SQLite database."""
    db_path = RESULTS_DIR / "results_captcha_gradient.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Pick the best run per model (latest with 50 outputs, skip the broken deepseek runs)
    runs = conn.execute("""
        SELECT r.run_id, m.model_id, m.model_id as mid,
               (SELECT COUNT(*) FROM outputs o WHERE o.run_id = r.run_id) as cnt,
               r.params_json
        FROM runs r JOIN models m ON r.model_id = m.model_id
        ORDER BY r.created_at DESC
    """).fetchall()

    # Pick best run per model: prefer 50 outputs, latest
    # Skip deepseek-r1:8b entirely — all runs are unreliable
    # (tools not supported → empty responses, no-tools run crashed at 32/50)
    best_runs = {}
    for r in runs:
        mid = r["model_id"]
        if "deepseek" in mid:
            continue
        cnt = r["cnt"]
        if mid in best_runs:
            if best_runs[mid]["cnt"] >= cnt:
                continue
        best_runs[mid] = {"run_id": r["run_id"], "cnt": cnt, "params_json": r["params_json"]}

    rows = []
    for model_id, info in best_runs.items():
        run_id = info["run_id"]
        params = json.loads(info["params_json"]) if info["params_json"] else {}
        tools = bool(params.get("tools_enabled"))

        provider = model_id.split(":")[0]
        model_name = model_id

        results = conn.execute("""
            SELECT c.case_id, o.raw_text, o.latency_ms,
                   s.score, s.label, s.reason, s.details_json,
                   c.expected, c.scheme
            FROM outputs o
            JOIN cases c ON o.case_id = c.case_id
            LEFT JOIN scores s ON o.output_id = s.output_id
            WHERE o.run_id = ?
            ORDER BY c.case_id
        """, (run_id,)).fetchall()

        for r in results:
            details = json.loads(r["details_json"]) if r["details_json"] else {}
            rows.append({
                "model": model_name,
                "provider": provider,
                "tools": tools,
                "case_id": r["case_id"],
                "scheme": r["scheme"],
                "expected": r["expected"],
                "model_output": (r["raw_text"] or "")[:200],
                "score": r["score"],
                "label": r["label"],
                "reason": r["reason"],
                "followed_hidden": details.get("followed_hidden", ""),
                "answered_visible": details.get("answered_visible", ""),
                "latency_ms": r["latency_ms"],
            })

    conn.close()
    return rows


def load_claude_results():
    """Load results from Claude Code detailed_results JSON files."""
    rows = []
    claude_models = [
        ("claude-opus-4.6", "responses_claude.json", "detailed_results_claude.json"),
        ("claude-sonnet-4.6", "responses_sonnet.json", "detailed_results_sonnet.json"),
        ("claude-haiku-4.5", "responses_haiku.json", "detailed_results_haiku.json"),
    ]

    for model_name, resp_file, detail_file in claude_models:
        detail_path = SCRIPTS_DIR / detail_file
        if not detail_path.exists():
            continue

        with open(detail_path) as f:
            results = json.load(f)

        for r in results:
            details = r.get("details", {})
            rows.append({
                "model": model_name,
                "provider": "anthropic",
                "tools": False,
                "case_id": r["case_id"],
                "scheme": r["scheme"],
                "expected": r["expected"],
                "model_output": r.get("model_output", "")[:200],
                "score": r["score"],
                "label": r["label"],
                "reason": r["reason"],
                "followed_hidden": details.get("followed_hidden", ""),
                "answered_visible": details.get("answered_visible", ""),
                "latency_ms": "",
            })

    return rows


def main():
    all_rows = load_sqlite_results() + load_claude_results()

    # Sort by model then case_id
    all_rows.sort(key=lambda r: (r["model"], r["case_id"]))

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    # Summary
    from collections import defaultdict
    summary = defaultdict(lambda: defaultdict(list))
    for r in all_rows:
        summary[r["model"]][r["scheme"]].append(r["score"])

    print(f"Wrote {len(all_rows)} rows to {OUTPUT_CSV}\n")
    print(f"{'Model':<30s} {'Scheme':<25s} {'Avg':>6s} {'N':>4s}")
    print("-" * 70)
    for model in sorted(summary):
        for scheme in sorted(summary[model]):
            scores = summary[model][scheme]
            avg = sum(scores) / len(scores)
            print(f"{model:<30s} {scheme:<25s} {avg:>6.2f} {len(scores):>4d}")
        print()


if __name__ == "__main__":
    main()
