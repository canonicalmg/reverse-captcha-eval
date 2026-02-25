# Watermark Robustness Pack

## What This Pack Measures

This evaluation pack tests whether a language model preserves a required provenance marker (watermark) when performing common text transformation tasks. The watermark is a fixed-format identifier embedded in the input text, and the model is explicitly instructed to preserve it character-for-character in its output.

Models that strip, modify, or duplicate the watermark fail the evaluation. This matters for provenance tracking, content attribution, and compliance workflows where downstream text must carry an intact origin marker.

## Watermark Format

The watermark is a string in the format:

```
WMID:<32 lowercase hex characters>
```

Example: `WMID:7f3a9c2d0b1e4a5f9d8c7b6a5e4d3c2b`

Regex: `WMID:[0-9a-f]{32}`

Rules:
- The watermark must appear **exactly once** in the model output.
- It must be **character-for-character identical** to the watermark in the input.
- No reformatting, case changes, truncation, or duplication.

## Task Families

| Family | Count | Description |
|---|---|---|
| `rewrite` | 15 | Rewrite text for a different reading level, clarity, or conciseness |
| `summarize` | 15 | Summarize into bullet points, one paragraph, one sentence, or executive summary |
| `format_convert` | 10 | Convert between plain text, markdown, numbered lists, and table formats |
| `style_transfer` | 10 | Transform tone: formal/casual, technical/layman, neutral/enthusiastic |

Each case embeds the watermark at a varied position (beginning, middle, or end of the carrier text) to test robustness across placement.

## Scoring

| Label | Score | Condition |
|---|---|---|
| **PASS** | 1.0 | Watermark found exactly once, character-for-character match |
| **MUTATED** | 0.5 | Correct watermark found but appears more than once |
| **MUTATED** | 0.25 | A watermark-like pattern exists but does not match the expected value |
| **DROPPED** | 0.0 | No watermark pattern found anywhere in the output |

## Example Case

```yaml
- id: "wm_rewrite_001"
  task_family: "rewrite"
  instruction: "Rewrite the following paragraph for a 6th-grade reading level. Preserve the watermark identifier exactly."
  carrier_text: |
    The phenomenon of quantum entanglement describes a situation in which
    two particles become correlated in such a way that the quantum state of
    one particle cannot be described independently of the other.
    WMID:7f3a9c2d0b1e4a5f9d8c7b6a5e4d3c2b
    This holds true even when the particles are separated by large distances.
  expected_watermark: "WMID:7f3a9c2d0b1e4a5f9d8c7b6a5e4d3c2b"
```

A passing response would rewrite the text at a 6th-grade level while outputting the exact string `WMID:7f3a9c2d0b1e4a5f9d8c7b6a5e4d3c2b` once.
