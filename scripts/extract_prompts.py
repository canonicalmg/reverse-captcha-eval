#!/usr/bin/env python3
"""Extract prompts from the reverse_captcha eval pack into a JSON file.

Loads cases.yaml and pack.yaml, constructs each full prompt the same way
the eval runner does (instruction + carrier_text), and writes prompts.json.

The carrier_text contains invisible zero-width Unicode characters (U+200B,
U+200C) that encode hidden instructions. These are preserved byte-for-byte
by reading the YAML with UTF-8 encoding and serializing JSON with
ensure_ascii=False.
"""

import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PACK_DIR = REPO_ROOT / "packs" / "reverse_captcha"
OUTPUT_PATH = Path(__file__).resolve().parent / "prompts.json"


def load_yaml(path: Path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    pack_yaml_path = PACK_DIR / "pack.yaml"
    cases_yaml_path = PACK_DIR / "cases.yaml"

    if not pack_yaml_path.exists():
        print(f"ERROR: pack.yaml not found at {pack_yaml_path}", file=sys.stderr)
        sys.exit(1)
    if not cases_yaml_path.exists():
        print(f"ERROR: cases.yaml not found at {cases_yaml_path}", file=sys.stderr)
        sys.exit(1)

    pack_data = load_yaml(pack_yaml_path)
    cases_data = load_yaml(cases_yaml_path)

    system_prompt = pack_data.get("system_prompt", "").strip()

    prompts = []
    for entry in cases_data:
        # Build prompt exactly as pack_loader.py does:
        #   prompt = f"{instruction}\n\n{carrier_text}"
        instruction = entry.get("instruction", "")
        carrier_text = entry.get("carrier_text", "")

        if instruction and carrier_text:
            prompt = f"{instruction}\n\n{carrier_text}"
        else:
            prompt = entry.get("prompt", "")

        expected = (
            entry.get("expected")
            or entry.get("expected_watermark")
            or entry.get("expected_message")
        )

        metadata = entry.get("metadata", {}) or {}
        if entry.get("scheme"):
            metadata["scheme"] = entry["scheme"]

        prompts.append(
            {
                "case_id": entry.get("id", ""),
                "scheme": entry.get("scheme", ""),
                "system_prompt": system_prompt,
                "prompt": prompt,
                "expected": expected,
                "metadata": metadata,
            }
        )

    # Write JSON with ensure_ascii=False to preserve zero-width Unicode chars
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(prompts)} prompts to {OUTPUT_PATH}")

    # Summary by scheme
    scheme_counts: dict[str, int] = {}
    for p in prompts:
        scheme = p["scheme"]
        scheme_counts[scheme] = scheme_counts.get(scheme, 0) + 1
    print("\nCases per scheme:")
    for scheme, count in sorted(scheme_counts.items()):
        print(f"  {scheme}: {count}")


if __name__ == "__main__":
    main()
