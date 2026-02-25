"""Chart generation for eval reports."""
import json
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")

COLORS = {
    "PASS": "#2ecc71",
    "MUTATED": "#f39c12",
    "DROPPED": "#e74c3c",
    "CORRECT": "#2ecc71",
    "PARTIAL": "#f39c12",
    "INCORRECT": "#e74c3c",
    "FALSE_POSITIVE": "#9b59b6",
}

_WATERMARK_LABELS = ["PASS", "MUTATED", "DROPPED"]
_EXTRACTION_LABELS = ["CORRECT", "PARTIAL", "INCORRECT", "FALSE_POSITIVE"]


def _query_df(db_path: str, sql: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


def _save(fig: plt.Figure, output_dir: str, name: str) -> str:
    path = str(Path(output_dir) / name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── watermark charts ──────────────────────────────────────────────

def watermark_stacked_bar(db_path: str, output_dir: str) -> str:
    """Stacked bar chart: retention/mutation/drop rates by model."""
    df = _query_df(db_path, """
        SELECT m.name AS model, s.label, COUNT(*) AS cnt
        FROM scores s
        JOIN outputs o ON o.output_id = s.output_id
        JOIN runs r    ON r.run_id    = o.run_id
        JOIN models m  ON m.model_id  = r.model_id
        JOIN cases c   ON c.case_id   = o.case_id
        WHERE r.pack_id = 'watermark_robustness'
        GROUP BY m.name, s.label
    """)
    if df.empty:
        return ""

    pivot = df.pivot_table(index="model", columns="label", values="cnt", fill_value=0)
    totals = pivot.sum(axis=1)
    pct = pivot.div(totals, axis=0) * 100

    # Ensure column order
    cols = [l for l in _WATERMARK_LABELS if l in pct.columns]
    pct = pct[cols]

    fig, ax = plt.subplots(figsize=(10, 6))
    pct.plot.bar(stacked=True, ax=ax, color=[COLORS[c] for c in cols], edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Watermark Robustness: Outcome Distribution by Model")
    ax.set_xlabel("")
    ax.set_ylim(0, 100)
    ax.legend(title="Outcome", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=30)

    return _save(fig, output_dir, "watermark_stacked_bar.png")


def watermark_by_task_type(db_path: str, output_dir: str) -> str:
    """Grouped bar chart: retention rate by task family for each model."""
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

    grouped = df.groupby(["model", "task_family"]).apply(
        lambda g: (g["label"] == "PASS").sum() / len(g) * 100, include_groups=False
    ).reset_index(name="retention_pct")

    pivot = grouped.pivot_table(index="task_family", columns="model", values="retention_pct", fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot.bar(ax=ax, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Retention Rate (%)")
    ax.set_title("Watermark Retention Rate by Task Family")
    ax.set_xlabel("Task Family")
    ax.set_ylim(0, 105)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=30)

    return _save(fig, output_dir, "watermark_by_task_type.png")


# ── extraction charts ─────────────────────────────────────────────

def extraction_accuracy_by_scheme(db_path: str, output_dir: str) -> str:
    """Grouped bar chart: accuracy by scheme type for each model."""
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

    grouped = df.groupby(["model", "scheme"]).apply(
        lambda g: (g["label"] == "CORRECT").sum() / len(g) * 100, include_groups=False
    ).reset_index(name="accuracy_pct")

    pivot = grouped.pivot_table(index="scheme", columns="model", values="accuracy_pct", fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot.bar(ax=ax, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Hidden-Message Extraction: Accuracy by Scheme")
    ax.set_xlabel("Scheme")
    ax.set_ylim(0, 105)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=30)

    return _save(fig, output_dir, "extraction_accuracy_by_scheme.png")


def extraction_false_positive_rate(db_path: str, output_dir: str) -> str:
    """Bar chart: false positive rate by model."""
    df = _query_df(db_path, """
        SELECT m.name AS model, s.label
        FROM scores s
        JOIN outputs o ON o.output_id = s.output_id
        JOIN runs r    ON r.run_id    = o.run_id
        JOIN models m  ON m.model_id  = r.model_id
        JOIN cases c   ON c.case_id   = o.case_id
        WHERE r.pack_id = 'hidden_message_extraction'
          AND c.scheme = 'no_message_control'
    """)
    if df.empty:
        return ""

    rates = df.groupby("model").apply(
        lambda g: (g["label"] == "FALSE_POSITIVE").sum() / len(g) * 100, include_groups=False
    ).reset_index(name="fp_rate")
    rates.columns = ["model", "fp_rate"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(rates["model"], rates["fp_rate"], color=COLORS["FALSE_POSITIVE"], edgecolor="white", linewidth=0.5)
    ax.set_ylabel("False Positive Rate (%)")
    ax.set_title("Hidden-Message Extraction: False Positive Rate by Model")
    ax.set_xlabel("")
    ax.set_ylim(0, max(rates["fp_rate"].max() * 1.2, 10))
    ax.tick_params(axis="x", rotation=30)

    for bar, val in zip(bars, rates["fp_rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    return _save(fig, output_dir, "extraction_false_positive_rate.png")


# ── shared charts ─────────────────────────────────────────────────

def latency_boxplot(db_path: str, output_dir: str) -> str:
    """Box plot: latency distribution by model."""
    df = _query_df(db_path, """
        SELECT m.name AS model, o.latency_ms
        FROM outputs o
        JOIN runs r   ON r.run_id   = o.run_id
        JOIN models m ON m.model_id = r.model_id
        WHERE o.latency_ms IS NOT NULL
    """)
    if df.empty:
        return ""

    models = sorted(df["model"].unique())
    data = [df[df["model"] == m]["latency_ms"].values for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, labels=models, patch_artist=True, medianprops=dict(color="black"))

    palette = plt.cm.tab10.colors
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(palette[i % len(palette)])
        patch.set_alpha(0.7)

    ax.set_ylabel("Latency (ms)")
    ax.set_title("Response Latency Distribution by Model")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)

    return _save(fig, output_dir, "latency_boxplot.png")


# ── orchestration ─────────────────────────────────────────────────

def generate_all_charts(db_path: str, output_dir: str) -> list[str]:
    """Generate all applicable charts. Returns list of chart file paths."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    generators = [
        watermark_stacked_bar,
        watermark_by_task_type,
        extraction_accuracy_by_scheme,
        extraction_false_positive_rate,
        latency_boxplot,
    ]
    paths: list[str] = []
    for gen in generators:
        try:
            p = gen(db_path, output_dir)
            if p:
                paths.append(p)
        except Exception:
            # Skip charts that fail (e.g. missing data for a pack)
            pass
    return paths
