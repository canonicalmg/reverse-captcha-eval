"""Tests for the evalrun database layer."""

import json
import sqlite3

import pytest

from evalrun.db import (
    get_run_results,
    get_scores_by_run,
    init_db,
    insert_case,
    insert_model,
    insert_output,
    insert_run,
    insert_score,
)


@pytest.fixture
def db(tmp_path):
    """Provide a fresh database connection for each test."""
    db_path = str(tmp_path / "test.sqlite")
    conn = init_db(db_path)
    yield conn
    conn.close()


# -------------------------------------------------------------------
# init_db
# -------------------------------------------------------------------


class TestInitDb:
    def test_creates_tables(self, db):
        tables = {
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "models" in tables
        assert "runs" in tables
        assert "cases" in tables
        assert "outputs" in tables
        assert "scores" in tables

    def test_creates_indexes(self, db):
        indexes = {
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert "idx_outputs_run_case" in indexes
        assert "idx_runs_model" in indexes

    def test_row_factory_is_set(self, db):
        assert db.row_factory == sqlite3.Row

    def test_idempotent(self, tmp_path):
        """Calling init_db twice on same path should not fail."""
        db_path = str(tmp_path / "test.sqlite")
        conn1 = init_db(db_path)
        conn1.close()
        conn2 = init_db(db_path)
        tables = {
            row[0]
            for row in conn2.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "models" in tables
        conn2.close()


# -------------------------------------------------------------------
# insert_model
# -------------------------------------------------------------------


class TestInsertModel:
    def test_insert_and_query(self, db):
        insert_model(
            db,
            model_id="openai:gpt-4o",
            name="GPT-4o",
            provider="openai",
            version="2024-01",
        )
        row = db.execute(
            "SELECT * FROM models WHERE model_id = ?", ("openai:gpt-4o",)
        ).fetchone()
        assert row is not None
        assert row["name"] == "GPT-4o"
        assert row["provider"] == "openai"
        assert row["version"] == "2024-01"

    def test_insert_without_version(self, db):
        insert_model(
            db, model_id="test:m1", name="Model1", provider="test"
        )
        row = db.execute(
            "SELECT * FROM models WHERE model_id = ?", ("test:m1",)
        ).fetchone()
        assert row["version"] is None

    def test_insert_or_ignore_duplicate(self, db):
        insert_model(db, model_id="m1", name="First", provider="p1")
        insert_model(db, model_id="m1", name="Second", provider="p2")
        row = db.execute(
            "SELECT * FROM models WHERE model_id = ?", ("m1",)
        ).fetchone()
        # INSERT OR IGNORE keeps the first insertion
        assert row["name"] == "First"


# -------------------------------------------------------------------
# insert_run
# -------------------------------------------------------------------


class TestInsertRun:
    def test_insert_and_query(self, db):
        insert_model(db, model_id="m1", name="M1", provider="p1")
        run_id = insert_run(
            db, pack_id="pack_a", model_id="m1", git_sha="abc123", params={"n": 3}
        )
        assert isinstance(run_id, str)
        assert len(run_id) == 32  # uuid4().hex

        row = db.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        assert row["pack_id"] == "pack_a"
        assert row["model_id"] == "m1"
        assert row["git_sha"] == "abc123"
        assert json.loads(row["params_json"]) == {"n": 3}

    def test_insert_without_optional_fields(self, db):
        insert_model(db, model_id="m1", name="M1", provider="p1")
        run_id = insert_run(db, pack_id="pack_a", model_id="m1")
        row = db.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        assert row["git_sha"] is None
        assert row["params_json"] is None


# -------------------------------------------------------------------
# insert_case
# -------------------------------------------------------------------


class TestInsertCase:
    def test_insert_and_query(self, db):
        case_id = insert_case(
            db,
            pack_id="pack_a",
            scheme="acrostic",
            metadata={"key": "val"},
            expected="SECRET",
            case_id="case_001",
        )
        assert case_id == "case_001"

        row = db.execute(
            "SELECT * FROM cases WHERE case_id = ?", (case_id,)
        ).fetchone()
        assert row["pack_id"] == "pack_a"
        assert row["scheme"] == "acrostic"
        assert json.loads(row["metadata_json"]) == {"key": "val"}
        assert row["expected"] == "SECRET"

    def test_auto_generated_id(self, db):
        case_id = insert_case(db, pack_id="pack_a")
        assert isinstance(case_id, str)
        assert len(case_id) == 32  # uuid4().hex

    def test_insert_or_ignore_duplicate(self, db):
        insert_case(db, pack_id="pack_a", expected="first", case_id="c1")
        insert_case(db, pack_id="pack_a", expected="second", case_id="c1")
        row = db.execute(
            "SELECT * FROM cases WHERE case_id = ?", ("c1",)
        ).fetchone()
        assert row["expected"] == "first"


# -------------------------------------------------------------------
# insert_output
# -------------------------------------------------------------------


class TestInsertOutput:
    def test_insert_and_query(self, db):
        insert_model(db, model_id="m1", name="M1", provider="p1")
        run_id = insert_run(db, pack_id="pack_a", model_id="m1")
        case_id = insert_case(db, pack_id="pack_a", case_id="c1")

        output_id = insert_output(
            db,
            run_id=run_id,
            case_id=case_id,
            raw_text="Model said this.",
            latency_ms=123.4,
            tokens_in=10,
            tokens_out=20,
        )
        assert isinstance(output_id, str)
        assert len(output_id) == 32

        row = db.execute(
            "SELECT * FROM outputs WHERE output_id = ?", (output_id,)
        ).fetchone()
        assert row["run_id"] == run_id
        assert row["case_id"] == case_id
        assert row["raw_text"] == "Model said this."
        assert row["latency_ms"] == pytest.approx(123.4)
        assert row["tokens_in"] == 10
        assert row["tokens_out"] == 20


# -------------------------------------------------------------------
# insert_score
# -------------------------------------------------------------------


class TestInsertScore:
    def test_insert_and_query(self, db):
        insert_model(db, model_id="m1", name="M1", provider="p1")
        run_id = insert_run(db, pack_id="pack_a", model_id="m1")
        case_id = insert_case(db, pack_id="pack_a", case_id="c1")
        output_id = insert_output(
            db, run_id=run_id, case_id=case_id, raw_text="x", latency_ms=50.0
        )

        insert_score(
            db,
            output_id=output_id,
            score=0.75,
            label="PARTIAL",
            reason="Partial match",
            details={"found": "x"},
        )
        row = db.execute(
            "SELECT * FROM scores WHERE output_id = ?", (output_id,)
        ).fetchone()
        assert row["score"] == pytest.approx(0.75)
        assert row["label"] == "PARTIAL"
        assert row["reason"] == "Partial match"
        assert json.loads(row["details_json"]) == {"found": "x"}

    def test_insert_score_without_optional_fields(self, db):
        insert_model(db, model_id="m1", name="M1", provider="p1")
        run_id = insert_run(db, pack_id="pack_a", model_id="m1")
        case_id = insert_case(db, pack_id="pack_a", case_id="c1")
        output_id = insert_output(
            db, run_id=run_id, case_id=case_id, raw_text="x", latency_ms=10.0
        )

        insert_score(db, output_id=output_id, score=1.0)
        row = db.execute(
            "SELECT * FROM scores WHERE output_id = ?", (output_id,)
        ).fetchone()
        assert row["score"] == 1.0
        assert row["label"] is None
        assert row["reason"] is None
        assert row["details_json"] is None


# -------------------------------------------------------------------
# get_run_results
# -------------------------------------------------------------------


class TestGetRunResults:
    def test_returns_joined_data(self, db):
        insert_model(db, model_id="m1", name="M1", provider="p1")
        run_id = insert_run(db, pack_id="pack_a", model_id="m1")
        case_id = insert_case(
            db, pack_id="pack_a", expected="WMID:abc", case_id="c1"
        )
        output_id = insert_output(
            db,
            run_id=run_id,
            case_id=case_id,
            raw_text="output text",
            latency_ms=100.0,
            tokens_in=5,
            tokens_out=10,
        )
        insert_score(
            db,
            output_id=output_id,
            score=1.0,
            label="PASS",
            reason="Exact match",
            details={"found": "WMID:abc"},
        )

        results = get_run_results(db, run_id)
        assert len(results) == 1
        r = results[0]
        assert r["output_id"] == output_id
        assert r["case_id"] == case_id
        assert r["raw_text"] == "output text"
        assert r["score"] == 1.0
        assert r["label"] == "PASS"
        assert r["expected"] == "WMID:abc"

    def test_returns_empty_for_unknown_run(self, db):
        results = get_run_results(db, "nonexistent_run_id")
        assert results == []

    def test_multiple_outputs(self, db):
        insert_model(db, model_id="m1", name="M1", provider="p1")
        run_id = insert_run(db, pack_id="pack_a", model_id="m1")
        case_id = insert_case(db, pack_id="pack_a", case_id="c1")

        for i in range(3):
            oid = insert_output(
                db,
                run_id=run_id,
                case_id=case_id,
                raw_text=f"output_{i}",
                latency_ms=float(i * 10),
            )
            insert_score(db, output_id=oid, score=float(i) / 2.0)

        results = get_run_results(db, run_id)
        assert len(results) == 3


# -------------------------------------------------------------------
# get_scores_by_run
# -------------------------------------------------------------------


class TestGetScoresByRun:
    def test_returns_score_data(self, db):
        insert_model(db, model_id="m1", name="M1", provider="p1")
        run_id = insert_run(db, pack_id="pack_a", model_id="m1")
        case_id = insert_case(db, pack_id="pack_a", case_id="c1")
        output_id = insert_output(
            db,
            run_id=run_id,
            case_id=case_id,
            raw_text="text",
            latency_ms=50.0,
            tokens_in=5,
            tokens_out=10,
        )
        insert_score(
            db,
            output_id=output_id,
            score=0.5,
            label="PARTIAL",
            reason="Partial match",
        )

        scores = get_scores_by_run(db, run_id)
        assert len(scores) == 1
        s = scores[0]
        assert s["output_id"] == output_id
        assert s["score"] == 0.5
        assert s["label"] == "PARTIAL"
        assert s["case_id"] == case_id
        assert s["latency_ms"] == pytest.approx(50.0)
        assert s["tokens_in"] == 5
        assert s["tokens_out"] == 10

    def test_returns_empty_for_unknown_run(self, db):
        scores = get_scores_by_run(db, "nonexistent_run_id")
        assert scores == []
