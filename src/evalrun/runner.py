import subprocess

from .adapters.base import ModelAdapter
from .db import (
    init_db,
    insert_case,
    insert_model,
    insert_output,
    insert_run,
    insert_score,
)
from .pack_loader import PackConfig


def _get_git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _format_prompt(template: str, case_vars: dict | None) -> str:
    if not case_vars:
        return template
    try:
        return template.format(**case_vars)
    except KeyError:
        return template


def run_eval(
    pack: PackConfig,
    adapters: list[ModelAdapter],
    n: int = 1,
    db_path: str = "results.sqlite",
    params: dict | None = None,
) -> list[str]:
    conn = init_db(db_path)
    git_sha = _get_git_sha()
    run_ids: list[str] = []

    for adapter in adapters:
        insert_model(
            conn,
            model_id=adapter.model_id,
            name=adapter.model_name,
            provider=adapter.provider,
        )

        run_id = insert_run(
            conn,
            pack_id=pack.id,
            model_id=adapter.model_id,
            git_sha=git_sha,
            params=params,
        )
        run_ids.append(run_id)

        total = len(pack.cases) * n
        completed = 0

        print(f"--- Model: {adapter.model_id} | Run: {run_id[:8]} ---")

        for case in pack.cases:
            case_id = insert_case(
                conn,
                pack_id=pack.id,
                case_id=case.id,
                scheme=case.scheme,
                metadata=case.metadata,
                expected=case.expected,
            )

            for rep in range(n):
                prompt = _format_prompt(case.prompt, case.metadata)

                gen_params = dict(params or {})
                result = adapter.generate(
                    prompt=prompt,
                    system=pack.system_prompt,
                    **gen_params,
                )

                output_id = insert_output(
                    conn,
                    run_id=run_id,
                    case_id=case_id,
                    raw_text=result.text,
                    latency_ms=result.latency_ms,
                    tokens_in=result.tokens_in,
                    tokens_out=result.tokens_out,
                    tool_meta=result.tool_meta,
                )

                score = 0.0
                label = None
                reason = None
                details = None

                if pack.grader and hasattr(pack.grader, "grade"):
                    grade_result = pack.grader.grade(
                        model_output=result.text,
                        expected=case.expected,
                        metadata=case.metadata,
                    )
                    if isinstance(grade_result, dict):
                        score = grade_result.get("score", 0.0)
                        label = grade_result.get("label")
                        reason = grade_result.get("reason")
                        details = grade_result.get("details")
                    else:
                        score = float(grade_result)

                insert_score(
                    conn,
                    output_id=output_id,
                    score=score,
                    label=label,
                    reason=reason,
                    details=details,
                )

                completed += 1
                tool_info = ""
                if result.tool_meta and result.tool_meta.get("tool_calls"):
                    tool_info = f" tools={result.tool_meta['tool_calls']}"
                print(
                    f"  [{completed}/{total}] case={case.id} rep={rep + 1}/{n} "
                    f"score={score:.2f} latency={result.latency_ms:.0f}ms{tool_info}"
                )

    conn.close()
    return run_ids
