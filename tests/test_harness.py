import json
import pytest
from pathlib import Path
from quantum_eval.harness import compute_suite_hash, load_suite, write_header, append_result, SuiteExample


def test_compute_suite_hash_is_deterministic(tmp_path):
    f = tmp_path / "suite.jsonl"
    f.write_text('{"id": "qiskitHumanEval_0", "instruction": "test"}\n')
    h1 = compute_suite_hash(f)
    h2 = compute_suite_hash(f)
    assert h1 == h2
    assert h1.startswith("sha256:")


def test_compute_suite_hash_changes_with_content(tmp_path):
    f = tmp_path / "suite.jsonl"
    f.write_text('{"id": "qiskitHumanEval_0"}\n')
    h1 = compute_suite_hash(f)
    f.write_text('{"id": "qiskitHumanEval_1"}\n')
    h2 = compute_suite_hash(f)
    assert h1 != h2


def test_load_suite_returns_examples(tmp_path):
    f = tmp_path / "suite.jsonl"
    f.write_text(
        '{"id": "qiskitHumanEval_0", "instruction": "create a Bell state"}\n'
        '{"id": "qiskitHumanEval_1", "instruction": "apply Hadamard gate"}\n'
    )
    examples = load_suite(f)
    assert len(examples) == 2
    assert examples[0].id == "qiskitHumanEval_0"
    assert examples[0].prompt == "create a Bell state"
    assert examples[1].id == "qiskitHumanEval_1"


def test_load_suite_accepts_prompt_field(tmp_path):
    """Suite files may use 'prompt' instead of 'instruction'."""
    f = tmp_path / "suite.jsonl"
    f.write_text('{"id": "qiskitHumanEval_0", "prompt": "create a GHZ state"}\n')
    examples = load_suite(f)
    assert examples[0].prompt == "create a GHZ state"


def test_write_header_creates_file_with_correct_fields(tmp_path):
    out = tmp_path / "results.jsonl"
    write_header(out, suite="humaneval", suite_hash="sha256:abc123", model="claude-sonnet-4-6")
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 1
    header = json.loads(lines[0])
    assert header["_header"] is True
    assert header["suite"] == "humaneval"
    assert header["suite_hash"] == "sha256:abc123"
    assert header["model"] == "claude-sonnet-4-6"
    assert "started" in header


def test_append_result_writes_json_line(tmp_path):
    out = tmp_path / "results.jsonl"
    out.write_text("")
    append_result(
        out,
        example_id="qiskitHumanEval_0",
        generated_code="qc = QuantumCircuit(2)",
        syntax_pass=True,
        execution_pass=True,
        semantic_pass=False,
        error="semantic mismatch",
    )
    record = json.loads(out.read_text().strip())
    assert record["id"] == "qiskitHumanEval_0"
    assert record["syntax_pass"] is True
    assert record["semantic_pass"] is False
    assert record["error"] == "semantic mismatch"


from quantum_eval.harness import get_completed_ids, get_header_suite_hash


def test_get_completed_ids_skips_header(tmp_path):
    out = tmp_path / "results.jsonl"
    out.write_text(
        '{"_header": true, "suite_hash": "sha256:abc"}\n'
        '{"id": "qiskitHumanEval_0", "semantic_pass": true}\n'
        '{"id": "qiskitHumanEval_1", "semantic_pass": false}\n'
    )
    ids = get_completed_ids(out)
    assert ids == {"qiskitHumanEval_0", "qiskitHumanEval_1"}


def test_get_completed_ids_empty_file(tmp_path):
    out = tmp_path / "results.jsonl"
    out.write_text("")
    assert get_completed_ids(out) == set()


def test_get_completed_ids_missing_file(tmp_path):
    out = tmp_path / "nonexistent.jsonl"
    assert get_completed_ids(out) == set()


def test_get_header_suite_hash_returns_hash(tmp_path):
    out = tmp_path / "results.jsonl"
    out.write_text(
        '{"_header": true, "suite_hash": "sha256:xyz789"}\n'
        '{"id": "qiskitHumanEval_0"}\n'
    )
    assert get_header_suite_hash(out) == "sha256:xyz789"


def test_get_header_suite_hash_missing_file(tmp_path):
    out = tmp_path / "nonexistent.jsonl"
    assert get_header_suite_hash(out) is None
