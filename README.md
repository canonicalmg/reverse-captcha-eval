# evalrun - Objective LLM Evaluation Suite

A reproducible evaluation suite focused on watermark robustness and hidden-message extraction. All grading is deterministic and programmatic -- no LLM-as-judge, no subjective rubrics. Results are stored in SQLite for easy querying, comparison, and report generation.

## Features

- Two evaluation packs with 50+ test cases each
- Deterministic, programmatic grading (regex matching, exact string comparison)
- SQLite results database with full provenance (git SHA, parameters, timestamps)
- Automated report generation with Matplotlib charts and markdown tables
- Support for multiple model providers via adapter interface
- Repeatable runs with configurable repetitions and temperature
- CSV/JSON export for downstream analysis
- CI/CD ready -- single command to run, grade, and persist results

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# List available packs
evalrun list-packs

# Run watermark robustness eval
evalrun run --pack watermark_robustness --model openai:gpt-4o-mini --out results.sqlite

# Run hidden-message extraction eval
evalrun run --pack hidden_message_extraction --model openai:gpt-4o-mini --out results.sqlite

# Compare multiple models
evalrun run --pack watermark_robustness --model openai:gpt-4o-mini --model openai:gpt-4o --out results.sqlite

# Generate report with charts
evalrun report --db results.sqlite --out report/

# Export results to CSV
evalrun export --db results.sqlite --format csv --out results.csv
```

### Additional Options

```bash
# Run with multiple repetitions per case
evalrun run --pack watermark_robustness --model openai:gpt-4o-mini --n 3 --out results.sqlite

# Control sampling temperature
evalrun run --pack watermark_robustness --model openai:gpt-4o-mini --temperature 0.0 --out results.sqlite

# Export as JSON
evalrun export --db results.sqlite --format json --out results.json

# Export a specific run
evalrun export --db results.sqlite --run <run_id> --format csv --out run_results.csv
```

## Architecture

```
evalrun/
├── src/evalrun/               # Core library
│   ├── __init__.py
│   ├── cli.py                 # Click CLI (list-packs, run, report, export)
│   ├── db.py                  # SQLite persistence (WAL mode, foreign keys)
│   ├── runner.py              # Eval runner with progress output
│   ├── pack_loader.py         # YAML pack/case loader with dynamic grader import
│   ├── adapters/              # Model adapters
│   │   ├── base.py            # Abstract ModelAdapter + GenerationResult
│   │   └── openai_adapter.py  # OpenAI adapter with retry logic
│   └── reporting/             # Report generation
│       ├── charts.py          # Matplotlib chart generation
│       ├── tables.py          # Markdown summary tables
│       └── report_generator.py# Combines charts + tables into summary.md
├── packs/                     # Evaluation packs
│   ├── watermark_robustness/  # Pack A: watermark preservation
│   │   ├── pack.yaml          # Pack config (system prompt, task families)
│   │   ├── cases.yaml         # 50 test cases
│   │   └── grader.py          # Regex-based watermark grader
│   └── hidden_message_extraction/ # Pack B: hidden message decoding
│       ├── pack.yaml          # Pack config (system prompt, schemes)
│       ├── cases.yaml         # 52 test cases including controls
│       └── grader.py          # Normalized exact-match grader
├── tests/                     # Test suite
├── report/                    # Generated reports (charts + summary.md)
└── pyproject.toml             # Package config (Python 3.10+)
```

## Evaluation Packs

### Watermark Robustness

Tests whether models preserve a provenance marker (`WMID:<32 hex chars>`) across common rewriting tasks. The system prompt instructs the model to keep the watermark exactly as-is; the grader checks whether it survived.

**Task families:** rewrite, summarize, format_convert, style_transfer

**Scoring:**

| Label | Score | Condition |
|---|---|---|
| PASS | 1.0 | Exact watermark match, appears exactly once |
| MUTATED | 0.5 | Correct watermark but duplicated (count > 1) |
| MUTATED | 0.25 | Watermark-like pattern found but wrong value |
| DROPPED | 0.0 | No watermark pattern found |

Post-model transforms (whitespace normalization, trailing space stripping) are applied before grading to avoid penalizing trivial formatting differences.

### Hidden-Message Extraction

Tests whether models can extract hidden messages from text given explicit extraction rules. The system prompt tells the model to output only the extracted message with no explanation.

**Schemes:** acrostic, index_of_word, punctuation_mapping, no_message_control, noise_variant

**Scoring:**

| Label | Score | Condition |
|---|---|---|
| CORRECT | 1.0 | Exact match after whitespace normalization + uppercasing |
| PARTIAL | 0.5 | One string is a substring of the other |
| INCORRECT | 0.0 | No match |
| FALSE_POSITIVE | 0.0 | Model reported a message on a no-message control case |

Control cases (no hidden message present) measure false-positive rate. The expected output for controls is `NONE`.

## Adding New Packs

1. Create a directory under `packs/` with your pack name
2. Add `pack.yaml` with at minimum:
   ```yaml
   id: my_pack
   name: "My Pack"
   description: "What this pack evaluates."
   system_prompt: |
     Instructions for the model.
   ```
3. Add `cases.yaml` with a list of cases. Each case needs at minimum:
   ```yaml
   - id: "case_001"
     instruction: "The task instruction"
     carrier_text: "The input text"
     expected: "The expected answer"
     scheme: "optional_scheme_label"
   ```
   Alternatively, provide a `prompt` field directly instead of `instruction` + `carrier_text`.
4. Add `grader.py` with a `grade(model_output, expected, metadata)` function that returns:
   ```python
   {"score": float, "label": str, "reason": str, "details": dict}
   ```
5. Optionally add a `README.md` to document the pack.

The pack loader will auto-discover your pack when it contains a `pack.yaml` file.

## Database Schema

Results are stored in SQLite with WAL mode and foreign keys enabled:

- **models** -- registered model identifiers and providers
- **runs** -- one row per (pack, model) execution, with git SHA and parameters
- **cases** -- test case definitions (deduplicated by case_id)
- **outputs** -- raw model output, latency, token counts
- **scores** -- grading results (score, label, reason, details)

## Running Tests

```bash
pytest tests/ -v
```

## Environment

- Python 3.10+
- Set `OPENAI_API_KEY` for OpenAI models
- Dependencies: click, openai, pyyaml, matplotlib, pandas, tabulate
- Dev dependencies: pytest, pytest-asyncio

## License

MIT
