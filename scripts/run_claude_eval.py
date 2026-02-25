#!/usr/bin/env python3
"""Run reverse_captcha eval cases through Claude Code CLI."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

CLAUDE_BIN = os.path.expanduser("~/.claude/local/claude")
PROMPTS_FILE = Path(__file__).parent / "prompts.json"


def run_case(system_prompt: str, prompt: str, model: str, timeout: int = 120) -> str:
    """Run a single case through claude -p."""
    combined = f"{system_prompt}\n\n{prompt}"
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    try:
        result = subprocess.run(
            [CLAUDE_BIN, "-p", combined, "--model", model,
             "--max-turns", "1", "--no-session-persistence"],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"
    except Exception as e:
        return f"[ERROR: {e}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="opus", help="Model alias or full name (e.g. opus, sonnet, haiku)")
    args = parser.parse_args()

    model = args.model
    safe_name = model.replace("/", "_").replace(":", "_")
    responses_file = Path(__file__).parent / f"responses_{safe_name}.json"

    with open(PROMPTS_FILE) as f:
        prompts = json.load(f)

    # Resume from existing responses if any
    if responses_file.exists():
        with open(responses_file) as f:
            responses = json.load(f)
        print(f"Resuming {model}: {len(responses)} responses already collected", flush=True)
    else:
        responses = {}

    total = len(prompts)
    for i, case in enumerate(prompts):
        case_id = case["case_id"]
        if case_id in responses:
            print(f"  [{i+1}/{total}] {case_id} (cached): {responses[case_id][:60]}", flush=True)
            continue

        start = time.time()
        answer = run_case(case["system_prompt"], case["prompt"], model)
        elapsed = time.time() - start

        responses[case_id] = answer
        print(f"  [{i+1}/{total}] {case_id} ({elapsed:.1f}s): {answer[:60]}", flush=True)

        # Save after each case (crash-safe)
        with open(responses_file, "w") as f:
            json.dump(responses, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(responses)} responses saved to {responses_file}", flush=True)


if __name__ == "__main__":
    main()
