from quantum_eval.validator import validate_example, ValidationResult, ValidationLevel


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
