import subprocess
import pytest


def run_cli(*args):
    return subprocess.run(
        ["uv", "run", "quantum-llm-benchmarks", *args],
        capture_output=True, text=True,
        cwd="/Users/david/Github/quantum-llm-benchmarks",
    )


def test_cli_help():
    result = run_cli("--help")
    assert result.returncode == 0
    assert "run" in result.stdout
    assert "publish" in result.stdout
    assert "list" in result.stdout
    assert "merge" in result.stdout


def test_cli_list_shows_registered_models():
    result = run_cli("list")
    assert result.returncode == 0
    assert "claude-sonnet-4-6" in result.stdout
    assert "Mistral Large" in result.stdout


def test_cli_run_unknown_model_exits_nonzero():
    result = run_cli("run", "--model", "nonexistent-model-xyz-99", "--suite", "humaneval")
    assert result.returncode != 0
    assert "not found" in result.stderr or "not found" in result.stdout


def test_cli_run_unknown_suite_exits_nonzero():
    result = run_cli("run", "--model", "claude-sonnet-4-6", "--suite", "nonexistent-suite")
    assert result.returncode != 0


def test_cli_merge_exits_with_not_implemented():
    """merge is not implemented in v1 — must exit non-zero with a helpful message."""
    result = run_cli("merge")
    assert result.returncode != 0
    assert "not implemented" in (result.stdout + result.stderr).lower()
