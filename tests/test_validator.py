import pytest
from quantum_eval.validator import validate_example, validate_test_code, ValidationResult, ValidationLevel


def test_syntax_failure_on_bad_code():
    result, _ = validate_example({"response": "```python\ndef foo(: bad syntax !!!!\n```", "category": "humaneval"})
    assert isinstance(result, ValidationResult)
    assert result.level_passed == ValidationLevel.NONE


def test_execution_failure_on_runtime_error():
    result, _ = validate_example({
        "response": "```python\nfrom qiskit import QuantumCircuit\nraise RuntimeError('boom')\n```",
        "category": "humaneval",
    })
    # Should pass syntax but fail execution
    assert result.level_passed == ValidationLevel.SYNTAX


def test_syntax_pass_with_valid_qiskit():
    result, _ = validate_example({
        "response": "from qiskit import QuantumCircuit\nqc = QuantumCircuit(2)",
        "category": "humaneval",
    })
    assert result.level_passed.value >= ValidationLevel.SYNTAX.value


def test_result_has_required_fields():
    result, _ = validate_example({"response": "x = 1", "category": "humaneval"})
    assert hasattr(result, "level_passed")
    assert hasattr(result, "error")
    assert isinstance(result.level_passed, ValidationLevel)


# =============================================================================
# validate_test_code raise/return convention
# =============================================================================

def test_validate_test_code_passing_test_returns_semantic():
    """Valid code + passing assertion → returns ValidationResult at SEMANTIC level."""
    pytest.importorskip("qiskit", reason="Qiskit not installed; subprocess will crash")
    code = "from qiskit import QuantumCircuit\nqc = QuantumCircuit(2)\nqc.h(0)"
    test_code = "assert qc.num_qubits == 2"
    result = validate_test_code(code, test_code)
    assert isinstance(result, ValidationResult)
    assert result.valid is True
    assert result.level_passed == ValidationLevel.SEMANTIC


def test_validate_test_code_assertion_failure_returns():
    """Valid code + failing assertion → returns ValidationResult (does not raise)."""
    code = "x = 1"
    test_code = "assert x == 99, 'wrong value'"
    result = validate_test_code(code, test_code)
    assert isinstance(result, ValidationResult)
    assert result.valid is False
    assert result.level_passed == ValidationLevel.EXECUTION


def test_validate_test_code_crash_raises_runtime_error():
    """Broken code that crashes at import/runtime → raises RuntimeError."""
    code = "raise RuntimeError('crash in model code')"
    test_code = "assert True"
    with pytest.raises(RuntimeError):
        validate_test_code(code, test_code)


def test_validate_test_code_missing_marker_raises():
    """Exit code 0 but missing __TEST_PASSED__ marker → raises RuntimeError."""
    from unittest.mock import patch, MagicMock

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""
    with patch("quantum_eval.validator.subprocess.run", return_value=mock_result):
        with pytest.raises(RuntimeError, match="did not complete"):
            validate_test_code("x = 1", "assert x == 1")


def test_validate_example_wraps_test_code_runtime_error():
    """validate_example catches RuntimeError from validate_test_code and returns a result."""
    from unittest.mock import patch, MagicMock

    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "ImportError: No module named 'qiskit'"
    with patch("quantum_eval.validator.subprocess.run", return_value=mock_result):
        result, _ = validate_example({
            "response": "from qiskit import QuantumCircuit\nqc = QuantumCircuit(1)",
            "category": "humaneval",
            "test_code": "assert qc.num_qubits == 1",
        })
    assert isinstance(result, ValidationResult)
    # Should not raise — run loop must survive the crash
