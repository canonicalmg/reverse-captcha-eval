import csv
import json
from pathlib import Path

import click

from .db import get_run_results, init_db
from .pack_loader import list_packs, load_pack
from .runner import run_eval


def _build_adapter(model_spec: str):
    parts = model_spec.split(":", 1)
    if len(parts) != 2:
        raise click.BadParameter(
            f"Invalid model spec '{model_spec}'. Expected format: provider:model_name"
        )
    provider, model_name = parts

    if provider == "openai":
        from .adapters.openai_adapter import OpenAIAdapter

        return OpenAIAdapter(model=model_name)

    if provider == "anthropic":
        from .adapters.anthropic_adapter import AnthropicAdapter

        return AnthropicAdapter(model=model_name)

    if provider == "ollama":
        from .adapters.openai_adapter import OpenAIAdapter

        return OpenAIAdapter(
            model=model_name,
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )

    raise click.BadParameter(f"Unknown provider '{provider}'")


@click.group()
def cli():
    """evalrun - Objective LLM Evaluation Suite"""


@cli.command("list-packs")
@click.option("--dir", "packs_dir", default="packs", help="Packs directory")
def list_packs_cmd(packs_dir: str):
    """List available evaluation packs."""
    packs = list_packs(packs_dir)
    if not packs:
        click.echo("No packs found.")
        return
    for name in packs:
        pack = load_pack(name, packs_dir)
        click.echo(f"  {pack.id:20s}  {pack.name} ({len(pack.cases)} cases)")


@cli.command("run")
@click.option("--pack", "pack_name", required=True, help="Pack name to run")
@click.option(
    "--model",
    "model_specs",
    multiple=True,
    required=True,
    help="Model spec as provider:model_name (repeatable)",
)
@click.option("--n", "n_reps", default=1, type=int, help="Repetitions per case")
@click.option("--out", "db_path", default="results.sqlite", help="Output database path")
@click.option("--dir", "packs_dir", default="packs", help="Packs directory")
@click.option("--temperature", type=float, default=None, help="Sampling temperature")
@click.option("--max-tokens", type=int, default=None, help="Max output tokens")
@click.option("--tools", "tools_enabled", is_flag=True, default=False, help="Enable agentic tool use (run_python)")
@click.option("--max-tool-turns", type=int, default=10, help="Max tool-use turns per generation (default: 10)")
@click.option("--case-timeout", type=int, default=120, help="Max seconds per case when using tools (default: 120)")
def run_cmd(
    pack_name: str,
    model_specs: tuple[str, ...],
    n_reps: int,
    db_path: str,
    packs_dir: str,
    temperature: float | None,
    max_tokens: int | None,
    tools_enabled: bool,
    max_tool_turns: int,
    case_timeout: int,
):
    """Run an evaluation pack against one or more models."""
    pack = load_pack(pack_name, packs_dir)
    adapters = [_build_adapter(spec) for spec in model_specs]

    params: dict = {}
    if temperature is not None:
        params["temperature"] = temperature
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if tools_enabled:
        params["tools_enabled"] = True
        params["max_tool_turns"] = max_tool_turns
        params["case_timeout"] = case_timeout

    click.echo(f"Running pack '{pack.name}' with {len(pack.cases)} cases, n={n_reps}")
    run_ids = run_eval(pack, adapters, n=n_reps, db_path=db_path, params=params or None)

    click.echo(f"\nCompleted. Run IDs:")
    for rid in run_ids:
        click.echo(f"  {rid}")
    click.echo(f"Results saved to {db_path}")


@cli.command("report")
@click.option("--db", "db_path", default="results.sqlite", help="Database path")
@click.option("--out", "out_dir", default="report", help="Output directory for report")
def report_cmd(db_path: str, out_dir: str):
    """Generate a summary report from evaluation results."""
    from .reporting import generate_report

    if not Path(db_path).exists():
        click.echo(f"Database not found: {db_path}")
        return

    md_path = generate_report(db_path, out_dir)
    click.echo(f"Report generated: {md_path}")


@cli.command("export")
@click.option("--db", "db_path", default="results.sqlite", help="Database path")
@click.option("--run", "run_id", default=None, help="Specific run ID (default: latest)")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["csv", "json"]),
    default="csv",
    help="Export format",
)
@click.option("--out", "out_path", default=None, help="Output file path")
def export_cmd(db_path: str, run_id: str | None, fmt: str, out_path: str | None):
    """Export evaluation results to CSV or JSON."""
    conn = init_db(db_path)

    if run_id is None:
        row = conn.execute(
            "SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            click.echo("No runs found in database.")
            return
        run_id = row["run_id"]

    results = get_run_results(conn, run_id)
    if not results:
        click.echo(f"No results found for run {run_id}.")
        return

    if out_path is None:
        out_path = f"results.{fmt}"

    if fmt == "csv":
        fieldnames = list(results[0].keys())
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    elif fmt == "json":
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    click.echo(f"Exported {len(results)} rows to {out_path}")
    conn.close()
