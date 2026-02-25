#!/usr/bin/env bash
# Run the full reverse_captcha journal evaluation matrix.
#
# Usage:
#   ./scripts/run_journal_eval.sh          # full run (n=2, merge pilot → n=3)
#   ./scripts/run_journal_eval.sh --pilot   # pilot run (n=1)
#   ./scripts/run_journal_eval.sh --full3   # fresh n=3, no pilot merge

set -euo pipefail

# Load API keys from .env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    source "$REPO_ROOT/.env"
    set +a
    echo "Loaded API keys from .env"
fi

DB="results/journal/eval.sqlite"
PACK="reverse_captcha"
N=2
MERGE_PILOT=true

if [[ "${1:-}" == "--pilot" ]]; then
    echo "=== PILOT MODE: n=1 ==="
    N=1
    DB="results/journal/pilot.sqlite"
    MERGE_PILOT=false
elif [[ "${1:-}" == "--full3" ]]; then
    echo "=== FULL n=3 MODE (no pilot merge) ==="
    N=3
    MERGE_PILOT=false
fi

mkdir -p results/journal

echo "============================================"
echo "  Reverse CAPTCHA Journal Evaluation"
echo "  Pack: $PACK | Reps: $N | DB: $DB"
if [[ "$MERGE_PILOT" == "true" ]]; then
    echo "  Will merge pilot data after → n=3 total"
fi
echo "============================================"
echo ""

run_model() {
    local model="$1"
    local tools_flag="${2:-}"
    local label="$model"
    if [[ -n "$tools_flag" ]]; then
        label="$model (tools)"
    fi
    echo ""
    echo "--- Running: $label ---"
    if ! evalrun run --pack "$PACK" --model "$model" --n "$N" --out "$DB" $tools_flag; then
        echo "*** FAILED: $label — continuing with next model ***"
    fi
    echo "--- Done: $label ---"
    echo ""
}

# ============================================
# API models (sequential to manage rate limits)
# ============================================

echo "=== OpenAI Models ==="

run_model "openai:gpt-5.2"
run_model "openai:gpt-5.2" "--tools"

run_model "openai:gpt-4o-mini"
run_model "openai:gpt-4o-mini" "--tools --case-timeout 180"

echo "=== Anthropic Models ==="

run_model "anthropic:claude-opus-4-20250514"
run_model "anthropic:claude-opus-4-20250514" "--tools"

run_model "anthropic:claude-sonnet-4-20250514"
run_model "anthropic:claude-sonnet-4-20250514" "--tools"

run_model "anthropic:claude-haiku-4-5-20251001"
run_model "anthropic:claude-haiku-4-5-20251001" "--tools"

# ============================================
# Merge pilot data if applicable
# ============================================

if [[ "$MERGE_PILOT" == "true" ]]; then
    echo ""
    echo "=== Merging pilot data (n=1) into eval DB ==="
    python3 "$SCRIPT_DIR/merge_pilot_into_eval.py" --eval "$DB"
fi

echo ""
echo "============================================"
echo "  All evaluations complete!"
echo "  Results: $DB"
echo "============================================"
