"""Table generation for eval reports."""
import json
import sqlite3

import pandas as pd
from tabulate import tabulate


def _query_df(db_path: str, sql: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


def _to_markdown(df: pd.DataFrame) -> str:
    return tabulate(df, headers="keys", tablefmt="pipe", showindex=False, floatfmt=".1f")


# ── watermark tables ──────────────────────────────────────────────

def watermark_summary_table(db_path: str) -> str:
    """Markdown table: model | retention% | mutation% | drop% | total cases."""
    df = _query_df(db_path, """
        SELECT m.name AS model, s.label
        FROM scores s
        JOIN outputs o ON o.output_id = s.output_id
        JOIN runs r    ON r.run_id    = o.run_id
        JOIN models m  ON m.model_id  = r.model_id
        WHERE r.pack_id = 'watermark_robustness'
    """)
    if df.empty:
        return ""

    rows = []
    for model, grp in df.groupby("model"):
        total = len(grp)
        pct = lambda lbl: (grp["label"] == lbl).sum() / total * 100
        rows.append({
            "Model": model,
            "Retention %": pct("PASS"),
            "Mutation %": pct("MUTATED"),
            "Drop %": pct("DROPPED"),
            "Total Cases": total,
        })
    return _to_markdown(pd.DataFrame(rows))


def watermark_by_task_table(db_path: str) -> str:
    """Markdown table: model | task_family | retention% | n."""
    df = _query_df(db_path, """
        SELECT m.name AS model,
               json_extract(c.metadata_json, '$.task_family') AS task_family,
               s.label
        FROM scores s
        JOIN outputs o ON o.output_id = s.output_id
        JOIN runs r    ON r.run_id    = o.run_id
        JOIN models m  ON m.model_id  = r.model_id
        JOIN cases c   ON c.case_id   = o.case_id
        WHERE r.pack_id = 'watermark_robustness'
          AND json_extract(c.metadata_json, '$.task_family') IS NOT NULL
    """)
    if df.empty:
        return ""

    rows = []
    for (model, tf), grp in df.groupby(["model", "task_family"]):
        rows.append({
            "Model": model,
            "Task Family": tf,
            "Retention %": (grp["label"] == "PASS").sum() / len(grp) * 100,
            "n": len(grp),
        })
    return _to_markdown(pd.DataFrame(rows))


# ── extraction tables ─────────────────────────────────────────────

def extraction_summary_table(db_path: str) -> str:
    """Markdown table: model | accuracy% | false_positive% | total cases."""
    df = _query_df(db_path, """
        SELECT m.name AS model, s.label, c.scheme
        FROM scores s
        JOIN outputs o ON o.output_id = s.output_id
        JOIN runs r    ON r.run_id    = o.run_id
        JOIN models m  ON m.model_id  = r.model_id
        JOIN cases c   ON c.case_id   = o.case_id
        WHERE r.pack_id = 'hidden_message_extraction'
    """)
    if df.empty:
        return ""

    rows = []
    for model, grp in df.groupby("model"):
        total = len(grp)
        accuracy = (grp["label"] == "CORRECT").sum() / total * 100
        control = grp[grp["scheme"] == "no_message_control"]
        fp_rate = (
            (control["label"] == "FALSE_POSITIVE").sum() / len(control) * 100
            if len(control) > 0
            else 0.0
        )
        rows.append({
            "Model": model,
            "Accuracy %": accuracy,
            "False Positive %": fp_rate,
            "Total Cases": total,
        })
    return _to_markdown(pd.DataFrame(rows))


def extraction_by_scheme_table(db_path: str) -> str:
    """Markdown table: model | scheme | accuracy% | n."""
    df = _query_df(db_path, """
        SELECT m.name AS model, c.scheme, s.label
        FROM scores s
        JOIN outputs o ON o.output_id = s.output_id
        JOIN runs r    ON r.run_id    = o.run_id
        JOIN models m  ON m.model_id  = r.model_id
        JOIN cases c   ON c.case_id   = o.case_id
        WHERE r.pack_id = 'hidden_message_extraction'
          AND c.scheme IS NOT NULL
    """)
    if df.empty:
        return ""

    rows = []
    for (model, scheme), grp in df.groupby(["model", "scheme"]):
        rows.append({
            "Model": model,
            "Scheme": scheme,
            "Accuracy %": (grp["label"] == "CORRECT").sum() / len(grp) * 100,
            "n": len(grp),
        })
    return _to_markdown(pd.DataFrame(rows))
