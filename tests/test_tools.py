"""Tests for the sandboxed Python executor."""

import os

import pytest

from evalrun.tools.run_python import run_python


class TestRunPython:
    def test_basic_exec(self):
        result = run_python("print('hello world')")
        assert result["exit_code"] == 0
        assert result["stdout"].strip() == "hello world"
        assert result["stderr"] == ""
        assert result["truncated"] is False
        assert result["duration_ms"] > 0

    def test_stderr(self):
        result = run_python("import sys; sys.stderr.write('oops\\n')")
        assert result["exit_code"] == 0
        assert "oops" in result["stderr"]

    def test_syntax_error(self):
        result = run_python("def f(")
        assert result["exit_code"] != 0
        assert "SyntaxError" in result["stderr"]

    def test_timeout(self):
        result = run_python("import time; time.sleep(10)", timeout=1)
        assert result["exit_code"] == -1
        assert "timed out" in result["stderr"]

    def test_truncation(self):
        # Generate output larger than 100KB
        code = "print('x' * 200_000)"
        result = run_python(code)
        assert result["truncated"] is True
        assert len(result["stdout"]) <= 100 * 1024

    def test_env_isolation(self):
        """API keys should not be visible in the subprocess."""
        os.environ["TEST_API_KEY"] = "secret123"
        try:
            result = run_python(
                "import os; print(os.environ.get('TEST_API_KEY', 'NOT_FOUND'))"
            )
            assert result["exit_code"] == 0
            assert "secret123" not in result["stdout"]
            assert "NOT_FOUND" in result["stdout"]
        finally:
            del os.environ["TEST_API_KEY"]

    def test_multiline_code(self):
        code = """
def add(a, b):
    return a + b

print(add(2, 3))
"""
        result = run_python(code)
        assert result["exit_code"] == 0
        assert result["stdout"].strip() == "5"

    def test_runtime_error(self):
        result = run_python("1/0")
        assert result["exit_code"] != 0
        assert "ZeroDivisionError" in result["stderr"]
