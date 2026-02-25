#!/usr/bin/env python3
"""Statistical analysis for the reverse CAPTCHA journal evaluation.

Computes per (model, scheme, tools) cell:
  - Compliance rate (FOLLOWED_HIDDEN / total) with Wilson score 95% CI
  - Control accuracy (CORRECT_CONTROL / control total)

Statistical tests:
  - Fisher's exact test: pairwise model comparison on binary compliance
  - Chi-squared test: scheme × compliance interaction per model
  - Bonferroni correction for multiple comparisons
  - Cohen's h for effect sizes on proportion differences

Usage:
    python scripts/analyze_journal.py [--db results/journal/eval.sqlite] [--out results/journal/]
"""

import argparse
import json
import sqlite3
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Wilson score confidence interval
# ---------------------------------------------------------------------------

def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score 95% CI for a binomial proportion.

    Returns (proportion, lower, upper).
    """
    if total == 0:
        return 0.0, 0.0, 0.0
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return p, max(0, center - spread), min(1, center + spread)


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(db_path: str) -> pd.DataFrame:
    """Load all reverse_captcha results from the journal SQLite database."""
    conn = sqlite3.connect(db_path)

    df = pd.read_sql_query("""
        SELECT
            r.run_id,
            m.model_id AS model,
            m.provider,
            r.params_json,
            c.case_id,
            c.scheme,
            c.expected,
            o.raw_text AS model_output,
            o.latency_ms,
            o.tokens_in,
            o.tokens_out,
            s.score,
            s.label,
            s.reason,
            s.details_json
        FROM outputs o
        JOIN runs r ON o.run_id = r.run_id
        JOIN models m ON r.model_id = m.model_id
        JOIN cases c ON o.case_id = c.case_id
        LEFT JOIN scores s ON o.output_id = s.output_id
        WHERE r.pack_id = 'reverse_captcha'
        ORDER BY m.model_id, c.scheme, c.case_id
    """, conn)

    conn.close()

    # Extract tools_enabled from params_json
    def has_tools(params_json):
        if not params_json or not isinstance(params_json, str):
            return False
        try:
            return bool(json.loads(params_json).get("tools_enabled"))
        except (json.JSONDecodeError, AttributeError, TypeError):
            return False

    df["tools"] = df["params_json"].apply(has_tools)

    # Derive encoding type from scheme
    def encoding_type(scheme):
        if scheme and scheme.startswith("tag_"):
            return "unicode_tags"
        elif scheme and scheme.startswith("zw_"):
            return "zero_width"
        elif scheme == "control":
            return "control"
        return "unknown"

    df["encoding"] = df["scheme"].apply(encoding_type)

    # Derive payload type: schemes ending in "_injection" are injection, others benign
    def payload_type(scheme):
        if scheme and scheme.endswith("_injection"):
            return "injection"
        return "benign"

    df["payload_type"] = df["scheme"].apply(payload_type)

    # Binary compliance flag
    df["complied"] = (df["label"] == "FOLLOWED_HIDDEN").astype(int)
    df["is_control"] = (df["scheme"] == "control").astype(int)

    return df


# ---------------------------------------------------------------------------
# Per-cell statistics
# ---------------------------------------------------------------------------

def compute_cell_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute compliance rate and Wilson CI per (model, scheme, tools) cell."""
    rows = []

    # For captcha cases (non-control)
    captcha_df = df[df["scheme"] != "control"]

    for (model, scheme, tools), group in captcha_df.groupby(["model", "scheme", "tools"]):
        n = len(group)
        k = group["complied"].sum()
        p, lo, hi = wilson_ci(k, n)
        rows.append({
            "model": model,
            "scheme": scheme,
            "tools": tools,
            "n": n,
            "complied": k,
            "rate": p,
            "ci_lo": lo,
            "ci_hi": hi,
        })

    # For control cases
    control_df = df[df["scheme"] == "control"]
    for (model, tools), group in control_df.groupby(["model", "tools"]):
        n = len(group)
        k = (group["label"] == "CORRECT_CONTROL").sum()
        p, lo, hi = wilson_ci(k, n)
        rows.append({
            "model": model,
            "scheme": "control",
            "tools": tools,
            "n": n,
            "complied": k,
            "rate": p,
            "ci_lo": lo,
            "ci_hi": hi,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def pairwise_fisher(df: pd.DataFrame, tools_filter: bool | None = None) -> pd.DataFrame:
    """Fisher's exact test for pairwise model comparison on compliance.

    Compares overall compliance rates between all model pairs.
    """
    captcha = df[(df["scheme"] != "control")]
    if tools_filter is not None:
        captcha = captcha[captcha["tools"] == tools_filter]

    model_stats = {}
    for model, group in captcha.groupby("model"):
        k = group["complied"].sum()
        n = len(group)
        model_stats[model] = (k, n - k)

    models = sorted(model_stats.keys())
    rows = []

    for m1, m2 in combinations(models, 2):
        k1, f1 = model_stats[m1]
        k2, f2 = model_stats[m2]
        table = [[k1, f1], [k2, f2]]
        odds_ratio, p_val = stats.fisher_exact(table)
        p1 = k1 / (k1 + f1) if (k1 + f1) > 0 else 0
        p2 = k2 / (k2 + f2) if (k2 + f2) > 0 else 0
        h = cohens_h(p1, p2)
        rows.append({
            "model_1": m1,
            "model_2": m2,
            "rate_1": p1,
            "rate_2": p2,
            "odds_ratio": odds_ratio,
            "p_value": p_val,
            "cohens_h": h,
        })

    result = pd.DataFrame(rows)

    # Bonferroni correction
    if len(result) > 0:
        result["p_bonferroni"] = np.minimum(result["p_value"] * len(result), 1.0)
        result["significant"] = result["p_bonferroni"] < 0.05

    return result


def scheme_chi_squared(df: pd.DataFrame) -> pd.DataFrame:
    """Chi-squared test for scheme × compliance interaction per model.

    Tests whether hint level significantly affects compliance rate.
    """
    captcha = df[df["scheme"] != "control"]
    rows = []

    for model, group in captcha.groupby("model"):
        contingency = pd.crosstab(group["scheme"], group["complied"])

        # Need at least 2 schemes with data
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            rows.append({
                "model": model,
                "chi2": np.nan,
                "p_value": np.nan,
                "dof": np.nan,
                "significant": False,
                "n_schemes": contingency.shape[0],
            })
            continue

        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
        rows.append({
            "model": model,
            "chi2": chi2,
            "p_value": p_val,
            "dof": dof,
            "significant": p_val < 0.05,
            "n_schemes": contingency.shape[0],
        })

    return pd.DataFrame(rows)


def tools_ablation(df: pd.DataFrame) -> pd.DataFrame:
    """Compare tools ON vs OFF compliance rates per model."""
    captcha = df[df["scheme"] != "control"]
    rows = []

    for model, group in captcha.groupby("model"):
        tools_on = group[group["tools"]]
        tools_off = group[~group["tools"]]

        if len(tools_on) == 0 or len(tools_off) == 0:
            continue

        k_on = tools_on["complied"].sum()
        n_on = len(tools_on)
        k_off = tools_off["complied"].sum()
        n_off = len(tools_off)

        p_on = k_on / n_on
        p_off = k_off / n_off

        # Fisher's exact for this model
        table = [[k_on, n_on - k_on], [k_off, n_off - k_off]]
        odds_ratio, p_val = stats.fisher_exact(table)
        h = cohens_h(p_on, p_off)

        rows.append({
            "model": model,
            "rate_tools_on": p_on,
            "rate_tools_off": p_off,
            "n_on": n_on,
            "n_off": n_off,
            "odds_ratio": odds_ratio,
            "p_value": p_val,
            "cohens_h": h,
            "significant": p_val < 0.05,
        })

    return pd.DataFrame(rows)


def encoding_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Compare binary ZW vs Unicode Tags compliance rates per model."""
    captcha = df[df["scheme"] != "control"]
    rows = []

    for model, group in captcha.groupby("model"):
        zw = group[group["encoding"] == "zero_width"]
        tag = group[group["encoding"] == "unicode_tags"]

        if len(zw) == 0 or len(tag) == 0:
            continue

        k_zw = zw["complied"].sum()
        n_zw = len(zw)
        k_tag = tag["complied"].sum()
        n_tag = len(tag)

        p_zw = k_zw / n_zw
        p_tag = k_tag / n_tag

        table = [[k_zw, n_zw - k_zw], [k_tag, n_tag - k_tag]]
        odds_ratio, p_val = stats.fisher_exact(table)
        h = cohens_h(p_zw, p_tag)

        rows.append({
            "model": model,
            "rate_zw": p_zw,
            "rate_tag": p_tag,
            "n_zw": n_zw,
            "n_tag": n_tag,
            "odds_ratio": odds_ratio,
            "p_value": p_val,
            "cohens_h": h,
            "significant": p_val < 0.05,
        })

    return pd.DataFrame(rows)


def payload_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Compare benign vs injection payload compliance rates per model.

    Uses Fisher's exact test to determine whether prompt injection framing
    significantly changes compliance rates.
    """
    captcha = df[df["scheme"] != "control"]
    rows = []

    for model, group in captcha.groupby("model"):
        benign = group[group["payload_type"] == "benign"]
        injection = group[group["payload_type"] == "injection"]

        if len(benign) == 0 or len(injection) == 0:
            continue

        k_benign = benign["complied"].sum()
        n_benign = len(benign)
        k_inject = injection["complied"].sum()
        n_inject = len(injection)

        p_benign = k_benign / n_benign
        p_inject = k_inject / n_inject

        table = [[k_benign, n_benign - k_benign], [k_inject, n_inject - k_inject]]
        odds_ratio, p_val = stats.fisher_exact(table)
        h = cohens_h(p_benign, p_inject)

        rows.append({
            "model": model,
            "rate_benign": p_benign,
            "rate_injection": p_inject,
            "n_benign": n_benign,
            "n_injection": n_inject,
            "odds_ratio": odds_ratio,
            "p_value": p_val,
            "cohens_h": h,
            "significant": p_val < 0.05,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary(cell_stats: pd.DataFrame) -> None:
    print("=" * 90)
    print("COMPLIANCE RATES WITH 95% WILSON CI")
    print("=" * 90)

    for (model, tools), group in cell_stats.groupby(["model", "tools"]):
        tools_str = "tools=ON" if tools else "tools=OFF"
        print(f"\n  {model} ({tools_str})")
        print(f"  {'Scheme':<25s} {'N':>4s} {'Rate':>7s} {'95% CI':>16s}")
        print(f"  {'-'*55}")
        for _, row in group.iterrows():
            ci_str = f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]"
            print(f"  {row['scheme']:<25s} {row['n']:>4d} {row['rate']:>7.3f} {ci_str:>16s}")


def main():
    parser = argparse.ArgumentParser(description="Statistical analysis for reverse CAPTCHA journal eval")
    parser.add_argument("--db", default="results/journal/eval.sqlite", help="SQLite database path")
    parser.add_argument("--out", default="results/journal/", help="Output directory for tables")
    args = parser.parse_args()

    db_path = args.db
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        print("Run the journal evaluation first: ./scripts/run_journal_eval.sh")
        return

    print(f"Loading results from {db_path}...")
    df = load_results(db_path)
    print(f"Loaded {len(df)} rows ({df['model'].nunique()} models, {df['scheme'].nunique()} schemes)")

    # Cell stats
    cell_stats = compute_cell_stats(df)
    cell_stats.to_csv(out_dir / "cell_stats.csv", index=False)
    print_summary(cell_stats)

    # Pairwise Fisher's exact
    print("\n" + "=" * 90)
    print("PAIRWISE MODEL COMPARISONS (Fisher's exact, Bonferroni-corrected)")
    print("=" * 90)

    fisher_df = pairwise_fisher(df)
    fisher_df.to_csv(out_dir / "pairwise_fisher.csv", index=False)

    if len(fisher_df) > 0:
        sig = fisher_df[fisher_df["significant"]]
        print(f"\n  {len(sig)}/{len(fisher_df)} pairs significant (p < 0.05, Bonferroni-corrected)")
        for _, row in fisher_df.iterrows():
            marker = "*" if row["significant"] else " "
            print(
                f"  {marker} {row['model_1']:>35s} vs {row['model_2']:<35s} "
                f"({row['rate_1']:.3f} vs {row['rate_2']:.3f}) "
                f"p={row['p_bonferroni']:.4f} h={row['cohens_h']:.3f}"
            )

    # Chi-squared per model
    print("\n" + "=" * 90)
    print("SCHEME × COMPLIANCE INTERACTION (Chi-squared per model)")
    print("=" * 90)

    chi2_df = scheme_chi_squared(df)
    chi2_df.to_csv(out_dir / "chi_squared.csv", index=False)

    for _, row in chi2_df.iterrows():
        marker = "*" if row["significant"] else " "
        print(f"  {marker} {row['model']:<40s} chi2={row['chi2']:>8.3f} p={row['p_value']:.4f} dof={row['dof']:.0f}")

    # Tools ablation
    print("\n" + "=" * 90)
    print("TOOL USE ABLATION (Fisher's exact)")
    print("=" * 90)

    tools_df = tools_ablation(df)
    tools_df.to_csv(out_dir / "tools_ablation.csv", index=False)

    for _, row in tools_df.iterrows():
        marker = "*" if row["significant"] else " "
        print(
            f"  {marker} {row['model']:<40s} "
            f"ON={row['rate_tools_on']:.3f} OFF={row['rate_tools_off']:.3f} "
            f"p={row['p_value']:.4f} h={row['cohens_h']:.3f}"
        )

    # Encoding comparison
    print("\n" + "=" * 90)
    print("ENCODING COMPARISON: Zero-Width vs Unicode Tags (Fisher's exact)")
    print("=" * 90)

    enc_df = encoding_comparison(df)
    enc_df.to_csv(out_dir / "encoding_comparison.csv", index=False)

    for _, row in enc_df.iterrows():
        marker = "*" if row["significant"] else " "
        print(
            f"  {marker} {row['model']:<40s} "
            f"ZW={row['rate_zw']:.3f} TAG={row['rate_tag']:.3f} "
            f"p={row['p_value']:.4f} h={row['cohens_h']:.3f}"
        )

    # Payload comparison (benign vs injection)
    print("\n" + "=" * 90)
    print("PAYLOAD COMPARISON: Benign vs Injection (Fisher's exact)")
    print("=" * 90)

    payload_df = payload_comparison(df)
    payload_df.to_csv(out_dir / "payload_comparison.csv", index=False)

    for _, row in payload_df.iterrows():
        marker = "*" if row["significant"] else " "
        print(
            f"  {marker} {row['model']:<40s} "
            f"BENIGN={row['rate_benign']:.3f} INJECT={row['rate_injection']:.3f} "
            f"p={row['p_value']:.4f} h={row['cohens_h']:.3f}"
        )

    # Save full dataframe
    df.to_csv(out_dir / "raw_results.csv", index=False)

    print(f"\nAll tables saved to {out_dir}/")
    print("  cell_stats.csv, pairwise_fisher.csv, chi_squared.csv,")
    print("  tools_ablation.csv, encoding_comparison.csv, payload_comparison.csv, raw_results.csv")


if __name__ == "__main__":
    main()
