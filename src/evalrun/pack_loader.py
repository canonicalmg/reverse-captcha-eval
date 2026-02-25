import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import yaml


@dataclass
class CaseConfig:
    id: str
    prompt: str
    expected: str | None = None
    metadata: dict | None = None
    scheme: str | None = None


@dataclass
class PackConfig:
    id: str
    name: str
    description: str
    system_prompt: str
    cases: list[CaseConfig] = field(default_factory=list)
    grader: ModuleType | None = None


def _load_yaml(path: Path) -> dict | list:
    with open(path) as f:
        return yaml.safe_load(f)


def _import_module_from_path(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_pack(pack_name: str, packs_dir: str = "packs") -> PackConfig:
    pack_path = Path(packs_dir) / pack_name

    pack_yaml_path = pack_path / "pack.yaml"
    if not pack_yaml_path.exists():
        raise FileNotFoundError(f"Pack config not found: {pack_yaml_path}")
    pack_data = _load_yaml(pack_yaml_path)

    cases_yaml_path = pack_path / "cases.yaml"
    if not cases_yaml_path.exists():
        raise FileNotFoundError(f"Cases file not found: {cases_yaml_path}")
    cases_data = _load_yaml(cases_yaml_path)

    cases: list[CaseConfig] = []
    if isinstance(cases_data, list):
        for entry in cases_data:
            # Build prompt from instruction + carrier_text, or use raw prompt
            instruction = entry.get("instruction", "")
            carrier_text = entry.get("carrier_text", "")
            if instruction and carrier_text:
                prompt = f"{instruction}\n\n{carrier_text}"
            else:
                prompt = entry.get("prompt", "")

            # Get expected value from any of the standard fields
            expected = (
                entry.get("expected")
                or entry.get("expected_watermark")
                or entry.get("expected_message")
            )

            # Build metadata from remaining fields
            metadata = entry.get("metadata", {}) or {}
            if entry.get("task_family"):
                metadata["task_family"] = entry["task_family"]
            if entry.get("scheme"):
                metadata["scheme"] = entry["scheme"]

            cases.append(
                CaseConfig(
                    id=str(entry.get("id", "")),
                    prompt=prompt,
                    expected=expected,
                    metadata=metadata,
                    scheme=entry.get("scheme"),
                )
            )

    grader_path = pack_path / "grader.py"
    grader_mod = None
    if grader_path.exists():
        grader_mod = _import_module_from_path(
            f"evalrun.graders.{pack_name}", grader_path
        )

    return PackConfig(
        id=pack_data.get("id", pack_name),
        name=pack_data.get("name", pack_name),
        description=pack_data.get("description", ""),
        system_prompt=pack_data.get("system_prompt", ""),
        cases=cases,
        grader=grader_mod,
    )


def list_packs(packs_dir: str = "packs") -> list[str]:
    packs_path = Path(packs_dir)
    if not packs_path.is_dir():
        return []
    return sorted(
        d.name
        for d in packs_path.iterdir()
        if d.is_dir() and (d / "pack.yaml").exists()
    )
