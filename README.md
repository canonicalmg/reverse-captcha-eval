# Reverse CAPTCHA: Evaluating LLM Susceptibility to Invisible Unicode Instruction Injection

An evaluation framework that tests whether large language models follow invisible Unicode-encoded instructions embedded in otherwise normal-looking text. Where traditional CAPTCHAs exploit tasks humans can solve but machines cannot, Reverse CAPTCHA exploits a perception channel that machines can access but humans cannot.

**Paper:** [`paper/reverse_captcha.pdf`](paper/reverse_captcha.pdf)

## Key Findings

Across **8,308 graded outputs** from 5 models (GPT-5.2, GPT-4o-mini, Claude Opus 4, Claude Sonnet 4, Claude Haiku 4.5):

### Tool use is the primary amplifier

Code execution access transforms invisible Unicode from an ignorable artifact to a decodable instruction channel. Without tools, compliance stays below 17%. With tools and hints, it reaches 98-100%.

![Tool Use Ablation](results/journal/figures/tools_ablation.png)

### Encoding vulnerability is provider-specific

OpenAI models decode zero-width binary but not Unicode Tags. Anthropic models prefer Unicode Tags. An attacker must tailor encoding to the target provider.

![Encoding Comparison](results/journal/figures/encoding_comparison.png)

### Full compliance heatmap

The hint gradient is consistent across models: unhinted << codepoint hints < full hints. The combination of tool access and decoding instructions is the critical enabler.

![Compliance Heatmap](results/journal/figures/heatmap_tools_on.png)

## How It Works

Two invisible encoding schemes are embedded in trivia questions:

- **Zero-Width Binary (ZW):** Each ASCII character encoded as 8 binary digits using U+200B (0) and U+200C (1)
- **Unicode Tags:** Each ASCII character mapped to U+E0000 + codepoint (one invisible char per ASCII char)

The model receives a trivia question with a known visible answer. Hidden in the invisible characters is a different answer. If the model outputs the hidden answer, it followed the invisible instruction.

### Test Matrix

- 30 trivia Q/A pairs
- 2 encoding schemes x 4 hint levels + 1 control = 9 conditions
- 2 tool conditions (ON/OFF)
- 5 models
- 270 cases per eval run

### Statistical Analysis

- Wilson score 95% confidence intervals
- Fisher's exact test with Bonferroni correction
- Chi-squared tests for per-model scheme variation
- Cohen's h effect sizes

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run reverse CAPTCHA eval
evalrun run --pack reverse_captcha --model openai:gpt-4o-mini --tools --n 3 --out results.sqlite

# Run without tools
evalrun run --pack reverse_captcha --model openai:gpt-4o-mini --n 3 --out results.sqlite

# Run with Anthropic models
evalrun run --pack reverse_captcha --model anthropic:claude-sonnet-4-20250514 --tools --n 3 --out results.sqlite

# Generate analysis
python3 scripts/analyze_journal.py

# Generate figures
python3 scripts/generate_figures.py
```

## Project Structure

```
├── packs/reverse_captcha/       # Eval pack (cases, grader, generator)
│   ├── cases.yaml               # 270 test cases
│   ├── qa_pairs.yaml            # 30 trivia Q/A pairs
│   ├── grader.py                # Deterministic grading (5 categories)
│   └── generate_cases.py        # Case generator for both encodings
├── src/evalrun/                 # Core framework
│   ├── adapters/                # OpenAI + Anthropic API adapters
│   ├── cli.py                   # CLI (run, report, export)
│   ├── runner.py                # Eval runner with tool-use support
│   └── db.py                    # SQLite persistence
├── paper/                       # Workshop paper
│   ├── reverse_captcha.tex      # LaTeX source
│   └── reverse_captcha.pdf      # Compiled PDF
├── results/journal/             # Raw data
│   ├── analysis/                # Statistical analysis CSVs
│   └── figures/                 # Publication figures (PDF + PNG)
├── scripts/                     # Analysis and figure generation
└── tests/                       # Test suite
```

## Other Eval Packs

The framework also includes two additional packs:

- **Watermark Robustness** — Tests whether models preserve a provenance marker across rewriting tasks
- **Hidden Message Extraction** — Tests whether models can extract hidden messages given explicit rules

## Environment

- Python 3.10+
- `OPENAI_API_KEY` for OpenAI models
- `ANTHROPIC_API_KEY` for Anthropic models
- Dependencies: `click`, `openai`, `anthropic`, `pyyaml`, `matplotlib`, `pandas`, `scipy`, `numpy`

## License

MIT
