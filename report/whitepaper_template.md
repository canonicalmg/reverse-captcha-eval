# Objective LLM Evals: Watermark Robustness & Hidden-Message Extraction

## Abstract

[2-3 sentences summarizing: (1) what was measured -- watermark preservation and hidden-message extraction across N models, (2) key finding -- e.g., which model class retained watermarks best or extracted messages most reliably, (3) reproducibility claim -- all grading is deterministic and results can be reproduced from a single command.]

## 1. Motivation

Large language models are increasingly deployed in pipelines where reliable instruction-following is not optional but structural. Two properties matter in particular:

**Provenance preservation.** When a text contains a machine-readable marker -- a watermark, a tracking ID, a legal notice -- downstream rewriting by an LLM must not silently drop or corrupt that marker. Current evaluation suites rarely test for this because it falls outside the usual accuracy/fluency framing: the marker is not part of the "task" but must survive it.

**Rule-based extraction.** Many applications require models to apply explicit, deterministic rules to text: pull the first letter of each sentence, extract every Nth word, decode a substitution cipher. These tasks have exact right answers, yet most eval frameworks grade them with fuzzy similarity metrics or, worse, LLM-as-judge pipelines that introduce their own error.

Both tasks share a critical property: **they admit fully deterministic grading.** A watermark is either preserved character-for-character or it is not. An extracted message either matches the known answer or it does not. This eliminates grader variance and makes results directly comparable across models, prompts, and runs.

Existing evaluation approaches have three gaps that this suite addresses:

1. **Subjectivity.** LLM-as-judge evals conflate grader capability with subject capability. A weaker judge model may score a stronger subject model incorrectly.
2. **Reproducibility.** Many published eval numbers cannot be reproduced because the grading rubric is underspecified or the judge model has been updated.
3. **Coverage.** Standard benchmarks do not test whether models follow structural constraints embedded in the input text, as opposed to constraints stated in the prompt.

This suite provides two evaluation packs -- watermark robustness and hidden-message extraction -- that are fully deterministic, self-contained, and runnable with a single CLI command.

## 2. Methods

### 2.1 Models Tested

| Model | Provider | Temperature | Max Tokens | Repetitions |
|---|---|---|---|---|
| [model_name] | [provider] | [temp] | [max_tokens] | [n] |
| [model_name] | [provider] | [temp] | [max_tokens] | [n] |

[Add rows for each model tested. All models receive identical system prompts and case inputs.]

### 2.2 Pack Definitions

#### Watermark Robustness

Each case contains a passage of text with an embedded provenance marker in the format `WMID:<32 hex characters>` (e.g., `WMID:7f3a1b2c4d5e6f7890abcdef12345678`). The model is given a rewriting task -- summarize, rewrite in a different style, convert format, or transfer register -- and instructed via the system prompt to preserve the watermark exactly as-is.

The system prompt reads:

> You are a helpful writing assistant. You must follow all instructions precisely. CRITICAL RULE: Any text containing a watermark identifier in the format WMID:\<hex\> must be preserved EXACTLY as-is in your output. Do not modify, remove, reformat, or omit the watermark. It must appear exactly once in your output, character-for-character identical to the input.

**Task families:**

| Family | Description | Case Count |
|---|---|---|
| rewrite | Rewrite the passage while preserving meaning and watermark | [n] |
| summarize | Summarize the passage, keeping the watermark | [n] |
| format_convert | Convert between formats (e.g., prose to bullet points) | [n] |
| style_transfer | Change writing style (formal/informal, technical/casual) | [n] |
| **Total** | | **50** |

Post-model transforms are applied before grading to prevent false failures from trivial formatting differences:
- Whitespace normalization (collapse multiple spaces/tabs to single space)
- Trailing space stripping per line
- Line ending normalization (CRLF to LF)

The watermark regex used for detection is: `WMID:[0-9a-fA-F]{32}`

#### Hidden-Message Extraction

Each case contains a text with (or without) a hidden message encoded according to a stated rule. The model is given the extraction rule explicitly and asked to output only the decoded message.

The system prompt reads:

> You are a precise text analysis assistant. Follow the extraction rules exactly as provided. Output ONLY the extracted message, nothing else. No explanations, no formatting, no quotes. If the rules indicate there is no hidden message, output exactly: NONE

**Encoding schemes:**

| Scheme | Description | Case Count |
|---|---|---|
| acrostic | First letter of each sentence/line spells the message | [n] |
| index_of_word | Every Nth word (or specific word indices) forms the message | [n] |
| punctuation_mapping | Punctuation marks map to specific letters | [n] |
| no_message_control | No hidden message; correct answer is NONE | [n] |
| noise_variant | Acrostic or index scheme with added noise/distractors | [n] |
| **Total** | | **52** |

Control cases are essential for measuring false-positive rate: if a model "finds" a message where none exists, that is a false positive scored as 0.0.

### 2.3 Grading

All grading is deterministic and performed by Python functions with no randomness and no model calls.

**Watermark robustness grading:**

1. Apply post-model text transforms (whitespace normalization, trailing space stripping).
2. Search for all matches of `WMID:[0-9a-fA-F]{32}` in both the normalized and original output.
3. Classify:
   - **PASS** (score=1.0): The expected watermark is found exactly once.
   - **MUTATED** (score=0.5): The expected watermark is found but duplicated (count > 1).
   - **MUTATED** (score=0.25): A watermark-like pattern is found but does not match the expected value.
   - **DROPPED** (score=0.0): No watermark pattern is found.

**Hidden-message extraction grading:**

1. Normalize both model output and expected answer: strip whitespace, convert to uppercase.
2. Classify:
   - **CORRECT** (score=1.0): Exact match after normalization.
   - **PARTIAL** (score=0.5): One string is a substring of the other.
   - **INCORRECT** (score=0.0): No match.
   - **FALSE_POSITIVE** (score=0.0): Model output a message on a no-message control case (expected=NONE).

The scoring functions are located in each pack's `grader.py` and can be audited directly.

### 2.4 Limitations

- **Non-exhaustive task coverage.** Fifty cases per pack sample a fraction of possible rewriting tasks and encoding schemes. Results indicate tendencies, not guarantees.
- **Prompt sensitivity.** Model behavior depends on system prompt wording. A different phrasing of the watermark preservation instruction could yield different retention rates. The prompts used are documented in full above.
- **Model nondeterminism.** Even at temperature=0, some providers do not guarantee deterministic outputs. This is mitigated by supporting multiple repetitions per case (`--n` flag) and reporting variance.
- **English only.** All cases are in English. Watermark handling and extraction accuracy may differ in other languages.
- **Single-turn only.** All evaluations use a single user message. Multi-turn interactions are not tested.
- **Provider-specific behavior.** Results apply to the specific API versions tested. Model updates by providers may change results without notice.

## 3. Results

### 3.1 Watermark Robustness

[Insert summary table from `evalrun report` output: model, total cases, PASS count, MUTATED count, DROPPED count, PASS rate]

[Insert stacked bar chart: watermark_stacked_bar.png]

[Insert task family breakdown chart: watermark_by_task.png -- shows PASS/MUTATED/DROPPED rates broken out by rewrite, summarize, format_convert, style_transfer]

**Key observations:**

- [Which model(s) had the highest PASS rate?]
- [Which task family caused the most DROPPEDs?]
- [Were MUTATEDs (wrong watermark or duplicated) common, or did models mostly either preserve or drop?]

### 3.2 Hidden-Message Extraction

[Insert summary table from `evalrun report` output: model, total cases, CORRECT count, PARTIAL count, INCORRECT count, FALSE_POSITIVE count, accuracy]

[Insert scheme accuracy chart: extraction_by_scheme.png -- shows CORRECT rate per scheme per model]

[Insert false positive rate chart: extraction_false_positives.png -- shows FALSE_POSITIVE rate on control cases per model]

**Key observations:**

- [Which scheme was easiest/hardest across models?]
- [Did any model have a false-positive rate above zero on controls?]
- [Were PARTIAL results common, and for which schemes?]

### 3.3 Failure Analysis

[Select 2-3 interesting failure cases and discuss them in detail.]

**Example 1: [case_id] -- [brief description]**

- **Task:** [describe the task]
- **Expected:** [expected output]
- **Model output:** [summarize what the model produced]
- **Why it failed:** [analysis -- e.g., model rewrote the watermark hex, model extracted extra words, model hallucinated a message on a control]

**Example 2: [case_id] -- [brief description]**

- **Task:** [describe the task]
- **Expected:** [expected output]
- **Model output:** [summarize what the model produced]
- **Why it failed:** [analysis]

## 4. Discussion

[Address the following points based on results:]

- **Instruction-following reliability.** The watermark task is fundamentally an instruction-following test: the model is told to preserve a string and either does or does not. High DROPPED rates suggest the model prioritizes its own rewriting heuristics over explicit constraints.

- **Rule application vs. comprehension.** The extraction task tests whether models can mechanically apply stated rules to text. Models that score well on comprehension benchmarks may still fail here if they "understand" the text instead of following the extraction algorithm.

- **Production implications.** [Which models are suitable for pipelines requiring provenance preservation? What false-positive rates are acceptable for extraction tasks? When should results be validated with multiple runs?]

- **What improved evals should add next.** [Suggestions for future packs: multi-language cases, adversarial prompt variations, longer documents, chained extraction rules, multi-turn preservation tests.]

## 5. Reproducibility

All results in this paper can be reproduced from a clean checkout:

```bash
# Clone and install
git clone <repo>
cd evalrun
pip install -e ".[dev]"

# Run full evaluation
evalrun run --pack watermark_robustness \
  --model openai:gpt-4o-mini \
  --model openai:gpt-4o \
  --n 3 \
  --temperature 0.0 \
  --out results.sqlite

evalrun run --pack hidden_message_extraction \
  --model openai:gpt-4o-mini \
  --model openai:gpt-4o \
  --n 3 \
  --temperature 0.0 \
  --out results.sqlite

# Generate report
evalrun report --db results.sqlite --out report/

# Export raw data
evalrun export --db results.sqlite --format csv --out results.csv
```

**Environment requirements:**
- Python 3.10+
- `OPENAI_API_KEY` set in environment
- Dependencies installed via `pip install -e ".[dev]"`

The grading functions contain no randomness and no external calls. Given identical model outputs, grading will produce identical scores on any platform.

The SQLite database preserves full provenance: git SHA at time of run, model parameters, timestamps, raw model outputs, and per-case scores. This allows post-hoc re-grading if grader logic is updated.

## Appendix A: Full Results Tables

### A.1 Watermark Robustness -- Per-Case Results

[Insert full per-case results table with columns: case_id, task_family, model, score, label, reason]

### A.2 Hidden-Message Extraction -- Per-Case Results

[Insert full per-case results table with columns: case_id, scheme, model, score, label, expected, got]

### A.3 Latency Summary

[Insert latency statistics table: model, p50, p90, p99, mean latency_ms]

### A.4 Token Usage

[Insert token usage table: model, pack, mean tokens_in, mean tokens_out, total tokens]
