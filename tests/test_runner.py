"""Tests for the eval runner using a mock adapter."""

import os
import types

import pytest

from evalrun.adapters.base import GenerationResult, ModelAdapter
from evalrun.db import get_run_results, get_scores_by_run, init_db
from evalrun.pack_loader import CaseConfig, PackConfig
from evalrun.runner import run_eval


# -------------------------------------------------------------------
# Mock adapter
# -------------------------------------------------------------------


class MockAdapter(ModelAdapter):
    """Adapter that returns fixed text for every call."""

    def __init__(self, fixed_text: str = "mock output"):
        self._fixed_text = fixed_text
        self._call_count = 0

    def generate(self, prompt: str, system: str = "", **params) -> GenerationResult:
        self._call_count += 1
        return GenerationResult(
            text=self._fixed_text,
            latency_ms=42.0,
            tokens_in=len(prompt.split()),
            tokens_out=len(self._fixed_text.split()),
        )

    @property
    def model_id(self) -> str:
        return "mock:test-model"

    @property
    def model_name(self) -> str:
        return "Test Model"

    @property
    def provider(self) -> str:
        return "mock"


# -------------------------------------------------------------------
# Simple keyword grader module
# -------------------------------------------------------------------


def _make_keyword_grader(keyword: str) -> types.ModuleType:
    """Create a grader module that checks for a keyword in the output."""
    mod = types.ModuleType("keyword_grader")

    def grade(model_output: str, expected: str, metadata: dict | None = None) -> dict:
        kw = keyword.upper()
        out = model_output.upper()
        if kw in out:
            return {
                "score": 1.0,
                "label": "PASS",
                "reason": f"Keyword '{keyword}' found",
                "details": {"keyword": keyword},
            }
        return {
            "score": 0.0,
            "label": "FAIL",
            "reason": f"Keyword '{keyword}' not found",
            "details": {"keyword": keyword},
        }

    mod.grade = grade
    return mod


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "runner_test.sqlite")


@pytest.fixture
def simple_pack():
    """A minimal pack with 2 cases and a keyword grader."""
    return PackConfig(
        id="test_pack",
        name="Test Pack",
        description="A test pack",
        system_prompt="You are a helpful assistant.",
        cases=[
            CaseConfig(
                id="case_1",
                prompt="Say the word 'mock'.",
                expected="mock",
            ),
            CaseConfig(
                id="case_2",
                prompt="Say something else.",
                expected="other",
            ),
        ],
        grader=_make_keyword_grader("mock"),
    )


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------


class TestRunEval:
    def test_populates_database(self, db_path, simple_pack):
        adapter = MockAdapter(fixed_text="Here is mock output")
        run_ids = run_eval(
            pack=simple_pack,
            adapters=[adapter],
            n=1,
            db_path=db_path,
        )
        assert len(run_ids) == 1

        conn = init_db(db_path)
        results = get_run_results(conn, run_ids[0])
        assert len(results) == 2  # 2 cases x 1 repetition

        # Verify model was inserted
        model_row = conn.execute(
            "SELECT * FROM models WHERE model_id = ?", ("mock:test-model",)
        ).fetchone()
        assert model_row is not None
        assert model_row["provider"] == "mock"
        conn.close()

    def test_scores_are_recorded(self, db_path, simple_pack):
        adapter = MockAdapter(fixed_text="Here is mock output")
        run_ids = run_eval(
            pack=simple_pack,
            adapters=[adapter],
            n=1,
            db_path=db_path,
        )

        conn = init_db(db_path)
        scores = get_scores_by_run(conn, run_ids[0])
        assert len(scores) == 2

        # Both outputs contain "mock", so both should pass
        for s in scores:
            assert s["score"] == 1.0
            assert s["label"] == "PASS"
        conn.close()

    def test_n_repetitions(self, db_path, simple_pack):
        adapter = MockAdapter(fixed_text="Here is mock output")
        run_ids = run_eval(
            pack=simple_pack,
            adapters=[adapter],
            n=3,
            db_path=db_path,
        )

        conn = init_db(db_path)
        results = get_run_results(conn, run_ids[0])
        assert len(results) == 6  # 2 cases x 3 reps
        conn.close()

    def test_adapter_called_correct_number_of_times(self, db_path, simple_pack):
        adapter = MockAdapter(fixed_text="mock output")
        run_eval(
            pack=simple_pack,
            adapters=[adapter],
            n=2,
            db_path=db_path,
        )
        assert adapter._call_count == 4  # 2 cases x 2 reps

    def test_grader_scoring_logic(self, db_path):
        """Test that a grader can produce different scores for different outputs."""
        pack = PackConfig(
            id="scoring_test",
            name="Scoring Test",
            description="Tests grader differentiation",
            system_prompt="",
            cases=[
                CaseConfig(id="c1", prompt="prompt", expected="yes"),
            ],
            grader=_make_keyword_grader("banana"),
        )

        # "mock output" does not contain "banana"
        adapter = MockAdapter(fixed_text="mock output")
        run_ids = run_eval(
            pack=pack,
            adapters=[adapter],
            n=1,
            db_path=db_path,
        )

        conn = init_db(db_path)
        scores = get_scores_by_run(conn, run_ids[0])
        assert len(scores) == 1
        assert scores[0]["score"] == 0.0
        assert scores[0]["label"] == "FAIL"
        conn.close()

    def test_no_grader(self, db_path):
        """When pack has no grader, scores default to 0."""
        pack = PackConfig(
            id="no_grader",
            name="No Grader",
            description="Pack without grader",
            system_prompt="",
            cases=[
                CaseConfig(id="c1", prompt="hello", expected="world"),
            ],
            grader=None,
        )
        adapter = MockAdapter(fixed_text="output")
        run_ids = run_eval(
            pack=pack,
            adapters=[adapter],
            n=1,
            db_path=db_path,
        )

        conn = init_db(db_path)
        scores = get_scores_by_run(conn, run_ids[0])
        assert len(scores) == 1
        assert scores[0]["score"] == 0.0
        conn.close()

    def test_multiple_adapters(self, db_path, simple_pack):
        adapter1 = MockAdapter(fixed_text="mock yes")
        adapter2 = MockAdapter(fixed_text="nope nothing")

        # Give them different IDs
        adapter2._model_id = "mock:model-2"
        original_prop = type(adapter2).model_id
        type(adapter2).model_id = property(lambda self: getattr(self, "_model_id", "mock:test-model"))

        try:
            run_ids = run_eval(
                pack=simple_pack,
                adapters=[adapter1, adapter2],
                n=1,
                db_path=db_path,
            )
            assert len(run_ids) == 2

            conn = init_db(db_path)
            # First adapter finds "mock" -> PASS
            scores1 = get_scores_by_run(conn, run_ids[0])
            assert all(s["score"] == 1.0 for s in scores1)

            # Second adapter doesn't say "mock" -> FAIL
            scores2 = get_scores_by_run(conn, run_ids[1])
            assert all(s["score"] == 0.0 for s in scores2)
            conn.close()
        finally:
            type(adapter2).model_id = original_prop

    def test_params_passed_through(self, db_path, simple_pack):
        """Verify that params dict is stored in the run record."""
        adapter = MockAdapter(fixed_text="mock")
        run_ids = run_eval(
            pack=simple_pack,
            adapters=[adapter],
            n=1,
            db_path=db_path,
            params={"temperature": 0.7, "max_tokens": 100},
        )

        conn = init_db(db_path)
        import json
        row = conn.execute(
            "SELECT params_json FROM runs WHERE run_id = ?", (run_ids[0],)
        ).fetchone()
        params = json.loads(row["params_json"])
        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 100
        conn.close()


# -------------------------------------------------------------------
# Mock adapter with tool_meta
# -------------------------------------------------------------------


class MockToolAdapter(ModelAdapter):
    """Adapter that simulates tool use and returns tool_meta."""

    def __init__(self, fixed_text: str = "decoded: HELLO", tool_calls: int = 2):
        self._fixed_text = fixed_text
        self._tool_calls = tool_calls

    def generate(self, prompt: str, system: str = "", **params) -> GenerationResult:
        return GenerationResult(
            text=self._fixed_text,
            latency_ms=100.0,
            tokens_in=10,
            tokens_out=5,
            tool_meta={"tool_calls": self._tool_calls},
        )

    @property
    def model_id(self) -> str:
        return "mock:tool-model"

    @property
    def model_name(self) -> str:
        return "Tool Model"

    @property
    def provider(self) -> str:
        return "mock"


class TestToolMeta:
    def test_tool_meta_stored(self, db_path):
        """tool_meta should be stored in tool_meta_json column."""
        pack = PackConfig(
            id="tool_test",
            name="Tool Test",
            description="Tests tool meta storage",
            system_prompt="",
            cases=[
                CaseConfig(id="c1", prompt="decode this", expected="HELLO"),
            ],
            grader=None,
        )
        adapter = MockToolAdapter(fixed_text="HELLO", tool_calls=3)
        run_ids = run_eval(pack=pack, adapters=[adapter], n=1, db_path=db_path)

        conn = init_db(db_path)
        row = conn.execute(
            "SELECT tool_meta_json FROM outputs WHERE run_id = ?", (run_ids[0],)
        ).fetchone()
        assert row is not None
        import json
        meta = json.loads(row["tool_meta_json"])
        assert meta["tool_calls"] == 3
        conn.close()

    def test_tool_meta_absent_for_normal_adapter(self, db_path):
        """Normal adapter should have NULL tool_meta_json."""
        pack = PackConfig(
            id="no_tool_test",
            name="No Tool Test",
            description="Tests absence of tool meta",
            system_prompt="",
            cases=[
                CaseConfig(id="c1", prompt="hello", expected="world"),
            ],
            grader=None,
        )
        adapter = MockAdapter(fixed_text="output")
        run_ids = run_eval(pack=pack, adapters=[adapter], n=1, db_path=db_path)

        conn = init_db(db_path)
        row = conn.execute(
            "SELECT tool_meta_json FROM outputs WHERE run_id = ?", (run_ids[0],)
        ).fetchone()
        assert row is not None
        assert row["tool_meta_json"] is None
        conn.close()
