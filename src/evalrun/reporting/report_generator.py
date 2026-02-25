"""Report generator that combines charts and tables into a markdown report."""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .charts import generate_all_charts
from .tables import (
    extraction_by_scheme_table,
    extraction_summary_table,
    watermark_by_task_table,
    watermark_summary_table,
)


def _run_metadata(db_path: str) -> dict:
    """Gather high-level metadata about the runs in the database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        models = [
            dict(r)
            for r in conn.execute(
                "SELECT DISTINCT m.name, m.provider FROM models m "
                "JOIN runs r ON r.model_id = m.model_id"
            ).fetchall()
        ]
        packs = [
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT pack_id FROM runs"
            ).fetchall()
        ]
        sha_row = conn.execute(
            "SELECT git_sha FROM runs WHERE git_sha IS NOT NULL LIMIT 1"
        ).fetchone()
        git_sha = sha_row["git_sha"] if sha_row else None
        return {"models": models, "packs": packs, "git_sha": git_sha}
    finally:
        conn.close()


def _has_pack_data(db_path: str, pack_id: str) -> bool:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM runs WHERE pack_id = ?", (pack_id,)
        ).fetchone()
        return row[0] > 0
    finally:
        conn.close()


def generate_report(db_path: str, output_dir: str) -> str:
    """Generate a full report from eval results.

    Creates chart PNGs in *output_dir* and writes ``summary.md`` with
    embedded image references and markdown tables.

    Returns the path to ``summary.md``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    chart_paths = generate_all_charts(db_path, output_dir)

    meta = _run_metadata(db_path)
    model_list = ", ".join(f"{m['name']} ({m['provider']})" for m in meta["models"]) or "none"

    sections: list[str] = []

    # ── header ────────────────────────────────────────────────────
    sections.append("# Evaluation Report\n")
    sections.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")
    sections.append(f"**Models:** {model_list}\n")
    if meta["git_sha"]:
        sections.append(f"**Git SHA:** `{meta['git_sha']}`\n")
    sections.append("")

    # ── watermark robustness ──────────────────────────────────────
    if _has_pack_data(db_path, "watermark_robustness"):
        sections.append("## Watermark Robustness\n")

        tbl = watermark_summary_table(db_path)
        if tbl:
            sections.append("### Summary\n")
            sections.append(tbl)
            sections.append("")

        tbl = watermark_by_task_table(db_path)
        if tbl:
            sections.append("### Retention by Task Family\n")
            sections.append(tbl)
            sections.append("")

        for cp in chart_paths:
            name = Path(cp).name
            if name.startswith("watermark_"):
                sections.append(f"![{name}]({name})\n")

    # ── hidden-message extraction ─────────────────────────────────
    if _has_pack_data(db_path, "hidden_message_extraction"):
        sections.append("## Hidden-Message Extraction\n")

        tbl = extraction_summary_table(db_path)
        if tbl:
            sections.append("### Summary\n")
            sections.append(tbl)
            sections.append("")

        tbl = extraction_by_scheme_table(db_path)
        if tbl:
            sections.append("### Accuracy by Scheme\n")
            sections.append(tbl)
            sections.append("")

        for cp in chart_paths:
            name = Path(cp).name
            if name.startswith("extraction_"):
                sections.append(f"![{name}]({name})\n")

    # ── latency ───────────────────────────────────────────────────
    latency_chart = [p for p in chart_paths if Path(p).name == "latency_boxplot.png"]
    if latency_chart:
        sections.append("## Latency\n")
        sections.append("![latency_boxplot.png](latency_boxplot.png)\n")

    md_path = str(out / "summary.md")
    Path(md_path).write_text("\n".join(sections))
    return md_path
