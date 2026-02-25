#!/usr/bin/env python3
"""Merge valid pilot data (n=1) into the eval database (n=2) to get n=3 total.

Copies valid pilot runs into eval.sqlite:
  - From pilot.sqlite: all runs EXCEPT Sonnet-tools and Opus-tools (credit-depleted)
  - From pilot2.sqlite: Sonnet-tools and Opus-tools (valid re-runs)

Usage:
    python scripts/merge_pilot_into_eval.py [--eval results/journal/eval.sqlite]
"""

import argparse
import json
import sqlite3
from pathlib import Path


def get_run_info(conn):
    """Get run metadata."""
    rows = conn.execute("""
        SELECT run_id, model_id, params_json FROM runs
    """).fetchall()
    results = []
    for row in rows:
        params = json.loads(row[2]) if row[2] else {}
        tools = bool(params.get("tools_enabled"))
        results.append({
            "run_id": row[0],
            "model_id": row[1],
            "tools": tools,
        })
    return results


def copy_run(src_conn, dst_conn, run_id):
    """Copy a single run and all its outputs/scores from src to dst."""
    # Copy run row
    run_row = src_conn.execute(
        "SELECT * FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if not run_row:
        return 0

    run_cols = [desc[0] for desc in src_conn.execute("SELECT * FROM runs LIMIT 0").description]
    placeholders = ", ".join(["?"] * len(run_cols))
    dst_conn.execute(
        f"INSERT OR IGNORE INTO runs ({', '.join(run_cols)}) VALUES ({placeholders})",
        run_row,
    )

    # Copy model if needed
    model_id = run_row[run_cols.index("model_id")]
    model_row = src_conn.execute(
        "SELECT * FROM models WHERE model_id = ?", (model_id,)
    ).fetchone()
    if model_row:
        model_cols = [desc[0] for desc in src_conn.execute("SELECT * FROM models LIMIT 0").description]
        placeholders = ", ".join(["?"] * len(model_cols))
        dst_conn.execute(
            f"INSERT OR IGNORE INTO models ({', '.join(model_cols)}) VALUES ({placeholders})",
            model_row,
        )

    # Copy cases (may already exist)
    cases = src_conn.execute("""
        SELECT DISTINCT c.* FROM cases c
        JOIN outputs o ON c.case_id = o.case_id
        WHERE o.run_id = ?
    """, (run_id,)).fetchall()
    case_cols = [desc[0] for desc in src_conn.execute("SELECT * FROM cases LIMIT 0").description]
    placeholders = ", ".join(["?"] * len(case_cols))
    for case_row in cases:
        dst_conn.execute(
            f"INSERT OR IGNORE INTO cases ({', '.join(case_cols)}) VALUES ({placeholders})",
            case_row,
        )

    # Copy outputs
    outputs = src_conn.execute(
        "SELECT * FROM outputs WHERE run_id = ?", (run_id,)
    ).fetchall()
    out_cols = [desc[0] for desc in src_conn.execute("SELECT * FROM outputs LIMIT 0").description]
    placeholders = ", ".join(["?"] * len(out_cols))
    count = 0
    for out_row in outputs:
        dst_conn.execute(
            f"INSERT OR IGNORE INTO outputs ({', '.join(out_cols)}) VALUES ({placeholders})",
            out_row,
        )
        count += 1

        # Copy corresponding score
        output_id = out_row[out_cols.index("output_id")]
        score_row = src_conn.execute(
            "SELECT * FROM scores WHERE output_id = ?", (output_id,)
        ).fetchone()
        if score_row:
            score_cols = [desc[0] for desc in src_conn.execute("SELECT * FROM scores LIMIT 0").description]
            placeholders_s = ", ".join(["?"] * len(score_cols))
            dst_conn.execute(
                f"INSERT OR IGNORE INTO scores ({', '.join(score_cols)}) VALUES ({placeholders_s})",
                score_row,
            )

    return count


def main():
    parser = argparse.ArgumentParser(description="Merge pilot data into eval DB")
    parser.add_argument("--eval", default="results/journal/eval.sqlite", help="Eval DB path")
    parser.add_argument("--pilot1", default="results/journal/pilot.sqlite", help="Pilot DB 1")
    parser.add_argument("--pilot2", default="results/journal/pilot2.sqlite", help="Pilot DB 2")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be copied")
    args = parser.parse_args()

    if not Path(args.eval).exists():
        print(f"Eval DB not found: {args.eval}")
        print("Run the n=2 evaluation first, then merge pilot data.")
        return

    # Open connections
    dst_conn = sqlite3.connect(args.eval)
    p1_conn = sqlite3.connect(args.pilot1)
    p2_conn = sqlite3.connect(args.pilot2)

    # Identify valid runs from pilot.sqlite
    # Skip: Sonnet-tools and Opus-tools (credit-depleted, contaminated)
    p1_runs = get_run_info(p1_conn)
    p1_valid = [
        r for r in p1_runs
        if not (r["tools"] and "sonnet" in r["model_id"].lower())
        and not (r["tools"] and "opus" in r["model_id"].lower())
    ]

    # All runs from pilot2.sqlite are valid (Sonnet-tools + Opus-tools re-runs)
    p2_runs = get_run_info(p2_conn)

    print("=== Merge Plan ===")
    print(f"\nFrom {args.pilot1} ({len(p1_valid)}/{len(p1_runs)} runs):")
    for r in p1_valid:
        tools_str = "tools" if r["tools"] else "no-tools"
        print(f"  {r['model_id']} ({tools_str}) — run {r['run_id'][:8]}")

    print(f"\nFrom {args.pilot2} ({len(p2_runs)} runs):")
    for r in p2_runs:
        tools_str = "tools" if r["tools"] else "no-tools"
        print(f"  {r['model_id']} ({tools_str}) — run {r['run_id'][:8]}")

    skipped = [r for r in p1_runs if r not in p1_valid]
    if skipped:
        print(f"\nSkipping from {args.pilot1} ({len(skipped)} contaminated runs):")
        for r in skipped:
            tools_str = "tools" if r["tools"] else "no-tools"
            print(f"  {r['model_id']} ({tools_str}) — run {r['run_id'][:8]}")

    if args.dry_run:
        print("\n[DRY RUN] No data copied.")
        return

    # Copy
    total_copied = 0
    for r in p1_valid:
        n = copy_run(p1_conn, dst_conn, r["run_id"])
        total_copied += n
        tools_str = "tools" if r["tools"] else "no-tools"
        print(f"  Copied {n} outputs: {r['model_id']} ({tools_str})")

    for r in p2_runs:
        n = copy_run(p2_conn, dst_conn, r["run_id"])
        total_copied += n
        tools_str = "tools" if r["tools"] else "no-tools"
        print(f"  Copied {n} outputs: {r['model_id']} ({tools_str})")

    dst_conn.commit()
    print(f"\nTotal: {total_copied} outputs merged into {args.eval}")

    # Verify final counts
    row = dst_conn.execute("SELECT COUNT(*) FROM outputs").fetchone()
    print(f"Eval DB now has {row[0]} total outputs")

    p1_conn.close()
    p2_conn.close()
    dst_conn.close()


if __name__ == "__main__":
    main()
