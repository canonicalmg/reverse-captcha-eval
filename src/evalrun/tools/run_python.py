"""Sandboxed Python code executor for agentic tool use."""

import os
import subprocess
import time

_MAX_OUTPUT = 100 * 1024  # 100 KB


def run_python(code: str, timeout: int = 30) -> dict:
    """Execute Python code in a sandboxed subprocess.

    Returns dict with stdout, stderr, exit_code, duration_ms, truncated.
    """
    # Sanitize environment: remove API keys and sensitive vars
    env = {
        k: v
        for k, v in os.environ.items()
        if not any(
            secret in k.upper()
            for secret in ("API_KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL")
        )
    }
    env["PATH"] = os.environ.get("PATH", "/usr/bin:/bin")

    start = time.perf_counter()
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/tmp",
            env=env,
        )
        duration_ms = (time.perf_counter() - start) * 1000.0

        stdout = result.stdout
        stderr = result.stderr
        truncated = False

        if len(stdout) > _MAX_OUTPUT:
            stdout = stdout[:_MAX_OUTPUT]
            truncated = True
        if len(stderr) > _MAX_OUTPUT:
            stderr = stderr[:_MAX_OUTPUT]
            truncated = True

        return {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": result.returncode,
            "duration_ms": round(duration_ms, 1),
            "truncated": truncated,
        }
    except subprocess.TimeoutExpired:
        duration_ms = (time.perf_counter() - start) * 1000.0
        return {
            "stdout": "",
            "stderr": f"Execution timed out after {timeout}s",
            "exit_code": -1,
            "duration_ms": round(duration_ms, 1),
            "truncated": False,
        }
