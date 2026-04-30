import json
import pytest
from pathlib import Path
from quantum_eval.results import aggregate_jsonl, wilson_ci, build_release_json


def test_wilson_ci_50_percent():
    # At 50/100, Wilson 95% CI is approximately 40–60%.
    low, high = wilson_ci(50, 100)
    assert 38 <= low <= 42
    assert 58 <= high <= 62


def test_wilson_ci_perfect():
    # k=n=151: Wilson lower bound ≈ 97.6%, rounds to 98.
    low, high = wilson_ci(151, 151)
    assert low >= 97
    assert high == 100


def test_wilson_ci_zero():
    # k=0, n=151: Wilson upper bound ≈ 2.4%, rounds to 2.
    low, high = wilson_ci(0, 151)
    assert low == 0
    assert high <= 3


def test_aggregate_jsonl_basic(tmp_path):
    out = tmp_path / "results.jsonl"
    out.write_text(
        '{"_header": true, "suite_hash": "sha256:abc", "model": "test-model"}\n'
        '{"id": "qiskitHumanEval_0", "syntax_pass": true, "execution_pass": true, "semantic_pass": true, "error": null}\n'
        '{"id": "qiskitHumanEval_1", "syntax_pass": true, "execution_pass": false, "semantic_pass": false, "error": "exec error"}\n'
        '{"id": "qiskitHumanEval_2", "syntax_pass": false, "execution_pass": false, "semantic_pass": false, "error": "syntax error"}\n'
    )
    stats = aggregate_jsonl(out)
    assert stats["n_examples"] == 3
    assert stats["syntax_pass"] == 2
    assert stats["execution_pass"] == 1
    assert stats["semantic_pass"] == 1
    assert stats["syntax_pct"] == 67
    assert "syntax_ci_low" in stats
    assert "syntax_ci_high" in stats


def test_aggregate_jsonl_skips_header(tmp_path):
    out = tmp_path / "results.jsonl"
    out.write_text(
        '{"_header": true, "suite_hash": "sha256:abc"}\n'
        '{"id": "qiskitHumanEval_0", "syntax_pass": true, "execution_pass": true, "semantic_pass": true, "error": null}\n'
    )
    stats = aggregate_jsonl(out)
    assert stats["n_examples"] == 1


def test_aggregate_jsonl_rejects_duplicate_ids(tmp_path):
    """Duplicate IDs indicate a botched resume. aggregate_jsonl must refuse, not silently double-count."""
    out = tmp_path / "results.jsonl"
    out.write_text(
        '{"_header": true}\n'
        '{"id": "qiskitHumanEval_0", "syntax_pass": true, "execution_pass": true, "semantic_pass": true, "error": null}\n'
        '{"id": "qiskitHumanEval_0", "syntax_pass": true, "execution_pass": true, "semantic_pass": true, "error": null}\n'
    )
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_jsonl(out)


def test_aggregate_jsonl_raises_on_empty(tmp_path):
    out = tmp_path / "results.jsonl"
    out.write_text('{"_header": true}\n')
    with pytest.raises(ValueError, match="No result records"):
        aggregate_jsonl(out)


def test_build_release_json_shape(tmp_path):
    out = tmp_path / "results.jsonl"
    out.write_text(
        '{"_header": true, "suite_hash": "sha256:abc", "model": "claude-sonnet-4-6", "suite": "humaneval"}\n'
        '{"id": "qiskitHumanEval_0", "syntax_pass": true, "execution_pass": true, "semantic_pass": true, "error": null}\n'
    )
    release = build_release_json(
        jsonl_path=out,
        version="2026-04",
        suite="humaneval",
        suite_hash="sha256:abc",
        model_id="claude-sonnet-4-6",
        model_label="Claude Sonnet 4.6",
        provider="anthropic",
        expected_n=1,
    )
    assert release["version"] == "2026-04"
    assert release["suite_hash"] == "sha256:abc"
    assert release["models"][0]["semantic_pass"] == 1
    assert release["models"][0]["semantic_pct"] == 100
    assert "semantic_ci_low" in release["models"][0]
    assert release["complete"] is True


def test_build_release_json_incomplete_run(tmp_path):
    """Publish of a partial run sets complete=False."""
    out = tmp_path / "results.jsonl"
    out.write_text(
        '{"_header": true, "suite_hash": "sha256:abc", "model": "claude-sonnet-4-6", "suite": "humaneval"}\n'
        '{"id": "qiskitHumanEval_0", "syntax_pass": true, "execution_pass": true, "semantic_pass": true, "error": null}\n'
    )
    release = build_release_json(
        jsonl_path=out,
        version="2026-04",
        suite="humaneval",
        suite_hash="sha256:abc",
        model_id="claude-sonnet-4-6",
        model_label="Claude Sonnet 4.6",
        provider="anthropic",
        expected_n=151,
    )
    assert release["complete"] is False
