#!/usr/bin/env python3
"""Generate publication-quality figures for the reverse CAPTCHA paper.

Figures:
  1. Heatmap (model × scheme): compliance rate with CI annotations
  2. Bar chart with error bars: compliance by hint level, grouped by model
  3. Tool use ablation: paired bar chart (tools ON vs OFF) per model
  4. Encoding comparison: Binary ZW vs Unicode Tags compliance rates

Exports PDF (vector) for the paper and PNG for the blog.

Usage:
    python scripts/generate_figures.py [--db results/journal/eval.sqlite] [--out results/journal/figures/]
"""

import argparse
import json
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

# Consistent style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# Color palette
MODEL_COLORS = {
    "openai:gpt-5.2": "#1f77b4",
    "openai:gpt-4o-mini": "#aec7e8",
    "anthropic:claude-opus-4-20250514": "#d62728",
    "anthropic:claude-sonnet-4-20250514": "#ff7f0e",
    "anthropic:claude-haiku-4-5-20251001": "#ffbb78",
}

# Short model names for display
MODEL_SHORT = {
    "openai:gpt-5.2": "GPT-5.2",
    "openai:gpt-4o-mini": "GPT-4o-mini",
    "anthropic:claude-opus-4-20250514": "Claude Opus",
    "anthropic:claude-sonnet-4-20250514": "Claude Sonnet",
    "anthropic:claude-haiku-4-5-20251001": "Claude Haiku",
}

HINT_ORDER = [
    "zw_unhinted", "zw_hint_codepoints", "zw_hint_full", "zw_hint_full_injection",
    "tag_unhinted", "tag_hint_codepoints", "tag_hint_full", "tag_hint_full_injection",
]

HINT_SHORT = {
    "zw_unhinted": "ZW: None",
    "zw_hint_codepoints": "ZW: Codepoints",
    "zw_hint_full": "ZW: Full",
    "zw_hint_full_injection": "ZW: Full+Inject",
    "tag_unhinted": "Tag: None",
    "tag_hint_codepoints": "Tag: Codepoints",
    "tag_hint_full": "Tag: Full",
    "tag_hint_full_injection": "Tag: Full+Inject",
}


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    lo = max(0.0, center - spread)
    hi = min(1.0, center + spread)
    # Clamp so p-lo and hi-p are never negative (floating point edge cases)
    return p, min(lo, p), max(hi, p)


def load_results(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT
            m.model_id AS model, r.params_json,
            c.scheme, s.label, s.score
        FROM outputs o
        JOIN runs r ON o.run_id = r.run_id
        JOIN models m ON r.model_id = m.model_id
        JOIN cases c ON o.case_id = c.case_id
        LEFT JOIN scores s ON o.output_id = s.output_id
        WHERE r.pack_id = 'reverse_captcha'
    """, conn)
    conn.close()

    def has_tools(p):
        if not p or not isinstance(p, str):
            return False
        try:
            return bool(json.loads(p).get("tools_enabled"))
        except (json.JSONDecodeError, AttributeError, TypeError):
            return False

    df["tools"] = df["params_json"].apply(has_tools)
    df["complied"] = (df["label"] == "FOLLOWED_HIDDEN").astype(int)

    def enc(scheme):
        if scheme and scheme.startswith("tag_"):
            return "unicode_tags"
        elif scheme and scheme.startswith("zw_"):
            return "zero_width"
        return "control"

    df["encoding"] = df["scheme"].apply(enc)

    def payload_type(scheme):
        if scheme and scheme.endswith("_injection"):
            return "injection"
        return "benign"

    df["payload_type"] = df["scheme"].apply(payload_type)
    return df


def short_name(model):
    return MODEL_SHORT.get(model, model.split(":")[-1])


def model_color(model):
    return MODEL_COLORS.get(model, "#999999")


# ---------------------------------------------------------------------------
# Figure 1: Heatmap
# ---------------------------------------------------------------------------

def _make_heatmap(df: pd.DataFrame, out_dir: Path, tools: bool, suffix: str):
    """Shared heatmap logic for tools ON/OFF variants."""
    captcha = df[(df["scheme"] != "control") & (df["tools"] == tools)]
    schemes = [s for s in HINT_ORDER if s in captcha["scheme"].unique()]
    models = sorted(captcha["model"].unique(), key=lambda m: short_name(m))

    if not models or not schemes:
        print(f"  heatmap_{suffix}: no data, skipping")
        return

    matrix = np.zeros((len(models), len(schemes)))
    annot = []

    for i, model in enumerate(models):
        row_annot = []
        for j, scheme in enumerate(schemes):
            cell = captcha[(captcha["model"] == model) & (captcha["scheme"] == scheme)]
            n = len(cell)
            k = cell["complied"].sum()
            p, lo, hi = wilson_ci(k, n)
            matrix[i, j] = p
            if n > 0:
                row_annot.append(f"{p:.0%}\n({lo:.0%}-{hi:.0%})")
            else:
                row_annot.append("--")
        annot.append(row_annot)

    fig, ax = plt.subplots(figsize=(12, max(4, len(models) * 0.7 + 1)))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(schemes)))
    ax.set_xticklabels([HINT_SHORT.get(s, s) for s in schemes], rotation=45, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([short_name(m) for m in models])

    for i in range(len(models)):
        for j in range(len(schemes)):
            color = "white" if matrix[i, j] > 0.5 else "black"
            ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=7, color=color)

    label = "ON" if tools else "OFF"
    ax.set_title(f"Hidden Instruction Compliance Rate (tools {label})")
    fig.colorbar(im, ax=ax, label="Compliance Rate", shrink=0.8)

    fig.savefig(out_dir / f"heatmap_{suffix}.pdf")
    fig.savefig(out_dir / f"heatmap_{suffix}.png")
    plt.close(fig)
    print(f"  heatmap_{suffix}.pdf / heatmap_{suffix}.png")


def fig_heatmap(df: pd.DataFrame, out_dir: Path):
    _make_heatmap(df, out_dir, tools=False, suffix="tools_off")
    _make_heatmap(df, out_dir, tools=True, suffix="tools_on")


# ---------------------------------------------------------------------------
# Figure 2: Bar chart by hint level
# ---------------------------------------------------------------------------

def fig_hint_bars(df: pd.DataFrame, out_dir: Path):
    captcha = df[(df["scheme"] != "control") & (~df["tools"])]
    models = sorted(captcha["model"].unique(), key=lambda m: short_name(m))
    schemes = [s for s in HINT_ORDER if s in captcha["scheme"].unique()]

    n_models = len(models)
    n_schemes = len(schemes)
    x = np.arange(n_schemes)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(14, 5))

    for i, model in enumerate(models):
        rates = []
        errors_lo = []
        errors_hi = []
        for scheme in schemes:
            cell = captcha[(captcha["model"] == model) & (captcha["scheme"] == scheme)]
            k = cell["complied"].sum()
            n = len(cell)
            p, lo, hi = wilson_ci(k, n)
            rates.append(p)
            errors_lo.append(p - lo)
            errors_hi.append(hi - p)

        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(
            x + offset, rates, width,
            yerr=[errors_lo, errors_hi],
            label=short_name(model),
            color=model_color(model),
            capsize=2,
            error_kw={"linewidth": 0.8},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([HINT_SHORT.get(s, s) for s in schemes], rotation=45, ha="right")
    ax.set_ylabel("Compliance Rate")
    ax.set_title("Compliance by Hint Level and Model (tools OFF)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.axhline(y=0, color="black", linewidth=0.5)

    fig.savefig(out_dir / "hint_bars.pdf")
    fig.savefig(out_dir / "hint_bars.png")
    plt.close(fig)
    print("  hint_bars.pdf / hint_bars.png")


# ---------------------------------------------------------------------------
# Figure 3: Tool use ablation
# ---------------------------------------------------------------------------

def fig_tools_ablation(df: pd.DataFrame, out_dir: Path):
    captcha = df[df["scheme"] != "control"]
    models = sorted(captcha["model"].unique(), key=lambda m: short_name(m))

    rates_on = []
    rates_off = []
    err_on = []
    err_off = []
    labels = []

    for model in models:
        for tools, rates_list, err_list in [(True, rates_on, err_on), (False, rates_off, err_off)]:
            cell = captcha[(captcha["model"] == model) & (captcha["tools"] == tools)]
            k = cell["complied"].sum()
            n = len(cell)
            p, lo, hi = wilson_ci(k, n)
            rates_list.append(p)
            err_list.append([p - lo, hi - p])
        labels.append(short_name(model))

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    err_off_arr = np.array(err_off).T
    err_on_arr = np.array(err_on).T

    ax.bar(x - width / 2, rates_off, width, yerr=err_off_arr, label="Tools OFF",
           color="#4e79a7", capsize=3, error_kw={"linewidth": 0.8})
    ax.bar(x + width / 2, rates_on, width, yerr=err_on_arr, label="Tools ON",
           color="#e15759", capsize=3, error_kw={"linewidth": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Compliance Rate")
    ax.set_title("Tool Use Ablation: Impact on Hidden Instruction Compliance")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylim(0, 1.05)
    ax.legend()

    fig.savefig(out_dir / "tools_ablation.pdf")
    fig.savefig(out_dir / "tools_ablation.png")
    plt.close(fig)
    print("  tools_ablation.pdf / tools_ablation.png")


# ---------------------------------------------------------------------------
# Figure 4: Encoding comparison
# ---------------------------------------------------------------------------

def fig_encoding_comparison(df: pd.DataFrame, out_dir: Path):
    captcha = df[(df["scheme"] != "control") & (df["tools"])]
    models = sorted(captcha["model"].unique(), key=lambda m: short_name(m))

    rates_zw = []
    rates_tag = []
    err_zw = []
    err_tag = []
    labels = []

    for model in models:
        for enc, rates_list, err_list in [
            ("zero_width", rates_zw, err_zw),
            ("unicode_tags", rates_tag, err_tag),
        ]:
            cell = captcha[(captcha["model"] == model) & (captcha["encoding"] == enc)]
            k = cell["complied"].sum()
            n = len(cell)
            p, lo, hi = wilson_ci(k, n)
            rates_list.append(p)
            err_list.append([p - lo, hi - p])
        labels.append(short_name(model))

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    err_zw_arr = np.array(err_zw).T
    err_tag_arr = np.array(err_tag).T

    ax.bar(x - width / 2, rates_zw, width, yerr=err_zw_arr, label="Zero-Width Binary",
           color="#59a14f", capsize=3, error_kw={"linewidth": 0.8})
    ax.bar(x + width / 2, rates_tag, width, yerr=err_tag_arr, label="Unicode Tags",
           color="#b07aa1", capsize=3, error_kw={"linewidth": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Compliance Rate")
    ax.set_title("Encoding Comparison: Zero-Width Binary vs Unicode Tags (tools ON)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylim(0, 1.05)
    ax.legend()

    fig.savefig(out_dir / "encoding_comparison.pdf")
    fig.savefig(out_dir / "encoding_comparison.png")
    plt.close(fig)
    print("  encoding_comparison.pdf / encoding_comparison.png")


# ---------------------------------------------------------------------------
# Figure 5: Payload comparison (benign vs injection)
# ---------------------------------------------------------------------------

def fig_payload_comparison(df: pd.DataFrame, out_dir: Path):
    """Grouped bar chart: benign hint_full vs injection hint_full compliance per model, per encoding."""
    captcha = df[(df["scheme"] != "control") & (df["tools"])]
    models = sorted(captcha["model"].unique(), key=lambda m: short_name(m))

    # Compare hint_full (benign) vs hint_full_injection for each encoding
    encodings = [
        ("zero_width", "zw_hint_full", "zw_hint_full_injection", "Zero-Width"),
        ("unicode_tags", "tag_hint_full", "tag_hint_full_injection", "Unicode Tags"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, (enc, benign_scheme, inject_scheme, enc_label) in zip(axes, encodings):
        rates_benign = []
        rates_inject = []
        err_benign = []
        err_inject = []
        labels = []

        for model in models:
            # Benign (hint_full)
            cell_b = captcha[(captcha["model"] == model) & (captcha["scheme"] == benign_scheme)]
            k_b = cell_b["complied"].sum()
            n_b = len(cell_b)
            p_b, lo_b, hi_b = wilson_ci(k_b, n_b)
            rates_benign.append(p_b)
            err_benign.append([p_b - lo_b, hi_b - p_b])

            # Injection (hint_full_injection)
            cell_i = captcha[(captcha["model"] == model) & (captcha["scheme"] == inject_scheme)]
            k_i = cell_i["complied"].sum()
            n_i = len(cell_i)
            p_i, lo_i, hi_i = wilson_ci(k_i, n_i)
            rates_inject.append(p_i)
            err_inject.append([p_i - lo_i, hi_i - p_i])

            labels.append(short_name(model))

        x = np.arange(len(models))
        width = 0.35

        err_benign_arr = np.array(err_benign).T
        err_inject_arr = np.array(err_inject).T

        ax.bar(x - width / 2, rates_benign, width, yerr=err_benign_arr,
               label="Benign (hint_full)", color="#4e79a7", capsize=3,
               error_kw={"linewidth": 0.8})
        ax.bar(x + width / 2, rates_inject, width, yerr=err_inject_arr,
               label="Injection (hint_full)", color="#e15759", capsize=3,
               error_kw={"linewidth": 0.8})

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(f"{enc_label} Encoding")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Compliance Rate")
    fig.suptitle("Payload Comparison: Benign vs Injection Framing (tools ON)", fontsize=12)
    fig.tight_layout()

    fig.savefig(out_dir / "payload_comparison.pdf")
    fig.savefig(out_dir / "payload_comparison.png")
    plt.close(fig)
    print("  payload_comparison.pdf / payload_comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate publication figures for reverse CAPTCHA paper")
    parser.add_argument("--db", default="results/journal/eval.sqlite", help="SQLite database path")
    parser.add_argument("--out", default="results/journal/figures/", help="Output directory for figures")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.db).exists():
        print(f"Database not found: {args.db}")
        print("Run the journal evaluation first: ./scripts/run_journal_eval.sh")
        return

    print(f"Loading results from {args.db}...")
    df = load_results(args.db)
    print(f"Loaded {len(df)} rows")

    print("\nGenerating figures...")
    fig_heatmap(df, out_dir)
    fig_hint_bars(df, out_dir)
    fig_tools_ablation(df, out_dir)
    fig_encoding_comparison(df, out_dir)
    fig_payload_comparison(df, out_dir)

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
