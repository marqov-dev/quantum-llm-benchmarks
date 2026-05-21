# tests/test_kl_validator.py
import pytest
from quantum_eval.kl_validator import (
    run_circuit_and_get_counts,
    run_canonical_circuit,
    kl_divergence,
    validate_kl_divergence,
    KLResult,
)

BELL_STATE = """
from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
"""

ZERO_STATE = """
from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
"""

# canonical_solution is a function body; prompt defines the signature.
BELL_PROMPT = "def create_bell_state():\n"
BELL_CANONICAL = "    qc = QuantumCircuit(2)\n    qc.h(0)\n    qc.cx(0, 1)\n    return qc\n"


def test_kl_divergence_identical_distributions():
    p = {"00": 0.5, "11": 0.5}
    q = {"00": 0.5, "11": 0.5}
    assert kl_divergence(p, q) < 1e-9


def test_kl_divergence_very_different_distributions():
    p = {"00": 1.0}
    q = {"11": 1.0}
    assert kl_divergence(p, q) > 1.0


def test_run_circuit_bell_state_standalone():
    """Standalone code: scanner finds QuantumCircuit in locals."""
    counts = run_circuit_and_get_counts(BELL_STATE, shots=1024, seed=42)
    total = sum(counts.values())
    assert total == pytest.approx(1.0, abs=0.01)
    assert set(counts.keys()).issubset({"00", "11"})


def test_run_circuit_reference_with_entry_point():
    """Reference runner: calls entry_point() to get circuit."""
    full_code = BELL_PROMPT + BELL_CANONICAL
    counts = run_circuit_and_get_counts(
        full_code, shots=1024, seed=42, entry_point="create_bell_state"
    )
    total = sum(counts.values())
    assert total == pytest.approx(1.0, abs=0.01)
    assert set(counts.keys()).issubset({"00", "11"})


def test_validate_kl_divergence_matching_circuits():
    """Same circuit (standalone vs function) should KL-pass."""
    result = validate_kl_divergence(
        generated_code=BELL_STATE,
        canonical_code=BELL_PROMPT + BELL_CANONICAL,
        entry_point="create_bell_state",
        seed=42,
    )
    assert result.passed is True
    assert result.kl_divergence is not None
    assert result.kl_divergence < 0.05


def test_validate_kl_divergence_wrong_circuit():
    """Wrong circuit: evaluated successfully but distributions diverge — passed=False (not None)."""
    result = validate_kl_divergence(
        generated_code=ZERO_STATE,
        canonical_code=BELL_PROMPT + BELL_CANONICAL,
        entry_point="create_bell_state",
        seed=42,
    )
    assert result.passed is False   # False = divergent (could evaluate, wrong answer)
    assert result.kl_divergence is not None
    assert result.kl_divergence > 0.05


def test_validate_kl_divergence_bad_generated_code():
    """Non-executable code: couldn't evaluate — passed=None, not False."""
    result = validate_kl_divergence(
        generated_code="this is not python!!!!",
        canonical_code=BELL_PROMPT + BELL_CANONICAL,
        entry_point="create_bell_state",
        seed=42,
    )
    assert result.passed is None    # None = couldn't evaluate (distinct from divergent)
    assert result.error is not None


def test_run_canonical_circuit_calls_entry_point():
    """run_canonical_circuit must call entry_point() — not just define it."""
    counts = run_canonical_circuit(
        BELL_PROMPT + BELL_CANONICAL,
        entry_point="create_bell_state",
        shots=1024,
        seed=42,
    )
    total = sum(counts.values())
    assert total == pytest.approx(1.0, abs=0.01)
    assert set(counts.keys()).issubset({"00", "11"})


def test_run_canonical_circuit_different_seeds_differ():
    """Two seeds should produce different (but valid) distributions."""
    d1 = run_canonical_circuit(BELL_PROMPT + BELL_CANONICAL, entry_point="create_bell_state", shots=1024, seed=1)
    d2 = run_canonical_circuit(BELL_PROMPT + BELL_CANONICAL, entry_point="create_bell_state", shots=1024, seed=2)
    assert sum(d1.values()) == pytest.approx(1.0, abs=0.01)
    assert sum(d2.values()) == pytest.approx(1.0, abs=0.01)
    # KL between two independent runs of the same circuit should be small but non-zero
    kl = kl_divergence(d1, d2)
    assert kl < 0.05  # sampling noise only


def test_run_circuit_handles_print_in_generated_code():
    """Generated code with print() statements must not break JSON parsing."""
    code_with_print = BELL_STATE + '\nprint("debug: circuit ready")\nprint(42)\n'
    counts = run_circuit_and_get_counts(code_with_print, shots=1024, seed=42)
    total = sum(counts.values())
    assert total == pytest.approx(1.0, abs=0.01)
    assert set(counts.keys()).issubset({"00", "11"})


def test_run_circuit_normalises_multi_register_keys():
    """Keys from multi-register circuits ('0 1' style) are normalised to '01'."""
    code = """
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
qr = QuantumRegister(2)
cr1 = ClassicalRegister(1)
cr2 = ClassicalRegister(1)
qc = QuantumCircuit(qr, cr1, cr2)
qc.measure(0, cr1[0])
qc.measure(1, cr2[0])
"""
    counts = run_circuit_and_get_counts(code, shots=128, seed=42)
    for key in counts:
        assert " " not in key, f"Key contains space: {key!r}"
