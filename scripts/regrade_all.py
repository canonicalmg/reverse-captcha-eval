#!/usr/bin/env python3
"""Re-grade all existing reverse_captcha results with the fixed grader.

Processes:
  1. Claude Code JSON response files (responses_*.json + prompts.json)
  2. SQLite databases (reads outputs.raw_text, recomputes scores)

Outputs:
  - Updated detailed_results_*.json files
  - Updated scores in SQLite databases
  - Re-exported combined CSV via export_all_results.py
"""

import importlib.util
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
GRADER_PATH = REPO_ROOT / "packs" / "reverse_captcha" / "grader.py"


def load_grader():
    spec = importlib.util.spec_from_file_location("reverse_captcha_grader", GRADER_PATH)
    if spec is None or spec.loader is None:
        print(f"ERROR: Cannot load grader from {GRADER_PATH}", file=sys.stderr)
        sys.exit(1)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def regrade_json_responses(grader) -> None:
    """Re-grade all Claude Code JSON response files."""
    prompts_path = SCRIPTS_DIR / "prompts.json"
    if not prompts_path.exists():
        print("  Skipping JSON re-grade: prompts.json not found")
        return

    with open(prompts_path) as f:
        prompts = json.load(f)

    prompt_lookup = {p["case_id"]: p for p in prompts}

    response_files = [
        ("responses_claude.json", "detailed_results_claude.json"),
        ("responses_sonnet.json", "detailed_results_sonnet.json"),
        ("responses_haiku.json", "detailed_results_haiku.json"),
        ("responses.json", "detailed_results.json"),
    ]

    for resp_file, detail_file in response_files:
        resp_path = SCRIPTS_DIR / resp_file
        if not resp_path.exists():
            continue

        with open(resp_path) as f:
            responses = json.load(f)

        detailed_results = []
        scheme_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        changed = 0
        total = 0

        # Load old results for comparison
        old_detail_path = SCRIPTS_DIR / detail_file
        old_labels = {}
        if old_detail_path.exists():
            with open(old_detail_path) as f:
                for r in json.load(f):
                    old_labels[r["case_id"]] = r["label"]

        for case_id, model_output in responses.items():
            if case_id not in prompt_lookup:
                continue

            total += 1
            prompt_data = prompt_lookup[case_id]
            expected = prompt_data["expected"]
            metadata = prompt_data.get("metadata", {})
            scheme = prompt_data["scheme"]

            result = grader.grade(model_output, expected, metadata)

            detailed_results.append({
                "case_id": case_id,
                "scheme": scheme,
                "model_output": model_output,
                "expected": expected,
                "score": result["score"],
                "label": result["label"],
                "reason": result["reason"],
                "details": result["details"],
            })

            scheme_stats[scheme][result["label"]] += 1

            if case_id in old_labels and old_labels[case_id] != result["label"]:
                changed += 1
                print(f"    {case_id}: {old_labels[case_id]} -> {result['label']}")

        out_path = SCRIPTS_DIR / detail_file
        with open(out_path, "w") as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)

        print(f"  {resp_file}: {total} cases, {changed} labels changed -> {detail_file}")


def regrade_sqlite(grader) -> None:
    """Re-grade scores in SQLite databases containing reverse_captcha data."""
    db_files = sorted(RESULTS_DIR.glob("*.sqlite"))

    for db_path in db_files:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # Check if this DB has reverse_captcha runs
        try:
            runs = conn.execute(
                "SELECT run_id FROM runs WHERE pack_id = 'reverse_captcha'"
            ).fetchall()
        except sqlite3.OperationalError:
            conn.close()
            continue

        if not runs:
            conn.close()
            continue

        run_ids = [r["run_id"] for r in runs]
        changed = 0
        total = 0

        for run_id in run_ids:
            rows = conn.execute("""
                SELECT o.output_id, o.raw_text, c.expected, c.metadata_json, c.scheme,
                       s.label as old_label
                FROM outputs o
                JOIN cases c ON o.case_id = c.case_id
                LEFT JOIN scores s ON o.output_id = s.output_id
                WHERE o.run_id = ?
            """, (run_id,)).fetchall()

            for row in rows:
                total += 1
                raw_text = row["raw_text"] or ""
                expected = row["expected"] or ""
                metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}

                result = grader.grade(raw_text, expected, metadata)

                new_label = result["label"]
                old_label = row["old_label"]

                if old_label != new_label:
                    changed += 1

                # Update the score row
                conn.execute("""
                    UPDATE scores SET
                        score = ?,
                        label = ?,
                        reason = ?,
                        details_json = ?
                    WHERE output_id = ?
                """, (
                    result["score"],
                    result["label"],
                    result["reason"],
                    json.dumps(result["details"]),
                    row["output_id"],
                ))

        conn.commit()
        conn.close()

        if total > 0:
            print(f"  {db_path.name}: {total} outputs, {changed} labels changed")


def main():
    grader = load_grader()

    print("Re-grading JSON response files...")
    regrade_json_responses(grader)

    print("\nRe-grading SQLite databases...")
    regrade_sqlite(grader)

    # Re-export combined CSV
    print("\nRe-exporting combined CSV...")
    export_script = SCRIPTS_DIR / "export_all_results.py"
    if export_script.exists():
        import subprocess
        subprocess.run([sys.executable, str(export_script)], cwd=str(REPO_ROOT))
    else:
        print("  Skipping: export_all_results.py not found")

    print("\nDone.")


if __name__ == "__main__":
    main()
