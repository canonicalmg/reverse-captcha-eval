#!/usr/bin/env bash
# Resume just the failed Anthropic n=2 runs
set -euo pipefail

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
        echo "*** FAILED: $label ***"
    fi
    echo "--- Done: $label ---"
}

echo "=== Resuming Anthropic runs (n=2) ==="

# Opus tools had 83/540 — run fresh n=2 (will add alongside partial)
run_model "anthropic:claude-opus-4-20250514" "--tools"

run_model "anthropic:claude-sonnet-4-20250514"
run_model "anthropic:claude-sonnet-4-20250514" "--tools"

run_model "anthropic:claude-haiku-4-5-20251001"
run_model "anthropic:claude-haiku-4-5-20251001" "--tools"

echo ""
echo "=== Anthropic runs complete! ==="
