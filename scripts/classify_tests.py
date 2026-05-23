"""
Static AST classification of IBM HumanEval test code and canonical solution output types.

Test classification tiers (max-tier wins for hybrid tests):
  behavioral — verifies quantum correctness via state/operator/measurement checks
  structural — checks circuit shape (qubits, depth, gate names) without simulation
  interface  — only checks return type (isinstance)

Canonical output types (from test_code and extracted_code analysis):
  QuantumCircuit, Statevector, Operator, SparsePauliOp, float, dict, list, tuple, other

Classification rules (written before coding, for reproducibility):

BEHAVIORAL signals in test_code:
  - `AerSimulator` anywhere in test_code
  - `Statevector.from_instruction(` — computes state vector from circuit
  - `state_fidelity(` — compares quantum states
  - `.equiv(` — Statevector.equiv or Operator.equiv (quantum equivalence check)
  - Sampler import with actual result access
  - Measurement distribution dict with bit-string keys (checking dict keys
    like "00", "11" etc. with fraction or set assertions)

STRUCTURAL signals (only when no behavioral signals):
  - `.num_qubits`, `.depth()`, `.count_ops()`, `.num_clbits`
  - `.data[` followed by `.operation` access
  - `operation.name` access on circuit instructions
  - `CircuitInstruction(` in assertions

INTERFACE signals (only when no behavioral AND no structural):
  - Only isinstance checks with no shape assertions

For hybrid tests (both structural and behavioral signals): BEHAVIORAL wins.

CANONICAL OUTPUT TYPE rules (test_code is authoritative; extracted_code is fallback):
  1. isinstance(result, X) in test_code → X is the output type
  2. result is wrapped in Operator(result) or Statevector(result) → QuantumCircuit
  3. result.equiv(Statevector(…)) → Statevector
  4. Operator(solution).equiv(result) → Operator
  5. result.num_qubits, result.data, result.layout → QuantumCircuit
  6. Return type annotation in extracted_code
  7. Last return statement keywords in extracted_code
"""

import ast
import json
import re
from collections import Counter
from pathlib import Path


# ── Behavioral detection ─────────────────────────────────────────────────────

_BEHAVIORAL_PATTERNS = [
    re.compile(r"\.equiv\s*\("),               # .equiv( with optional whitespace
    re.compile(r"from_instruction\s*\("),       # Statevector.from_instruction(
    re.compile(r"state_fidelity\s*\("),         # state_fidelity(
    re.compile(r"AerSimulator\s*[(\[]"),        # AerSimulator() or AerSimulator[
    re.compile(r"AerSimulator\b"),              # plain AerSimulator import/use
]

# Measurement distribution: bit-string keys in a dict from the run result
_MEAS_BIT_PATTERN = re.compile(r'["\']([01]+)["\']')


def _is_behavioral(test_code: str) -> bool:
    for pat in _BEHAVIORAL_PATTERNS:
        if pat.search(test_code):
            return True
    # Measurement dict checks: bit-string key access + value comparison
    if _MEAS_BIT_PATTERN.search(test_code) and (
        "sum(result.values())" in test_code
        or "result.keys()" in test_code
        or "result.get(" in test_code
        or 'result["' in test_code
        or "result['" in test_code
    ):
        return True
    return False


# ── Structural detection ─────────────────────────────────────────────────────

_STRUCTURAL_PATTERNS = [
    re.compile(r"\.num_qubits\b"),
    re.compile(r"\.depth\s*\("),
    re.compile(r"\.count_ops\s*\("),
    re.compile(r"\.num_nonlocal_gates\s*\("),
    re.compile(r"\.num_clbits\b"),
    re.compile(r"\.data\["),
    re.compile(r"\.data\b"),                   # circuit.data iteration
    re.compile(r"operation\.name\b"),
    re.compile(r"CircuitInstruction\s*\("),
    re.compile(r"\.layout\b"),
]


def _is_structural(test_code: str) -> bool:
    for pat in _STRUCTURAL_PATTERNS:
        if pat.search(test_code):
            return True
    return False


# ── Interface detection ──────────────────────────────────────────────────────

_ISINSTANCE_PATTERN = re.compile(r"isinstance\s*\(")


def _is_interface_only(test_code: str) -> bool:
    """True only if every assert in the test is an isinstance check."""
    lines = test_code.splitlines()
    assert_lines = [l.strip() for l in lines if l.strip().startswith("assert")]
    if not assert_lines:
        return False
    return all(_ISINSTANCE_PATTERN.search(a) for a in assert_lines)


# ── Test classifier ──────────────────────────────────────────────────────────

def classify_test(test_code: str) -> str:
    """Classify a test as 'behavioral', 'structural', or 'interface'."""
    if not test_code:
        return "unknown"
    # Max-tier rule: behavioral > structural > interface
    if _is_behavioral(test_code):
        return "behavioral"
    if _is_structural(test_code):
        return "structural"
    if _is_interface_only(test_code):
        return "interface"
    return "structural"  # conservative fallback


# ── Canonical output type classifier ─────────────────────────────────────────

_ISINSTANCE_TYPE_PATTERN = re.compile(
    r"isinstance\s*\(\s*\w+\s*,\s*(\w[\w\.]*?)\)"
)
_RETURN_ANNOTATION = re.compile(r"->\s*(\w[\w\[\], .]*?):")

_TYPE_MAP = {
    "QuantumCircuit": "QuantumCircuit",
    "Statevector": "Statevector",
    "SparsePauliOp": "SparsePauliOp",
    "Operator": "Operator",
    "DAGCircuit": "DAGCircuit",
    "NoiseModel": "NoiseModel",
    "DensityMatrix": "DensityMatrix",
    "Choi": "Choi",
    "CNOTDihedral": "CNOTDihedral",
    "Clifford": "Clifford",
    "StabilizerState": "StabilizerState",
    "LinearFunction": "LinearFunction",
    "ScalarOp": "ScalarOp",
    "TransformationPass": "TransformationPass",
    "Target": "Target",
    "PrimitiveJob": "PrimitiveJob",
    "Figure": "Figure",
    "RXGate": "Gate",
    "RZGate": "Gate",
}

_BASIC_TYPE_MAP = {
    "dict": "dict",
    "list": "list",
    "tuple": "tuple",
    "float": "float",
    "int": "float",
    "str": "str",
    "bool": "bool",
}


def _type_from_string(s: str) -> str | None:
    for kw, label in _TYPE_MAP.items():
        if kw in s:
            return label
    for kw, label in _BASIC_TYPE_MAP.items():
        if kw == s.strip():
            return label
    return None


def classify_output_type(test_code: str, extracted_code: str) -> str:
    """
    Infer what type the canonical solution returns.
    test_code is authoritative; extracted_code is the fallback.
    """
    # 1. isinstance check in test_code — most reliable
    for m in _ISINSTANCE_TYPE_PATTERN.finditer(test_code):
        typ = _type_from_string(m.group(1))
        if typ:
            return typ

    # 2. result is wrapped in Operator(result) / Statevector(result):
    #    this means the function returned a QuantumCircuit (or compatible)
    if re.search(r"Operator\s*\(\s*\w+\(\)\s*\)", test_code):
        return "QuantumCircuit"  # function returns QC, test wraps it in Operator
    if re.search(r"Operator\s*\(\s*candidate\s*\(\)", test_code):
        return "QuantumCircuit"

    # 3. result.equiv(Statevector(...)) → result IS a Statevector
    if ".equiv(" in test_code and "Statevector" in test_code:
        # Check if the result variable itself has .equiv called on it
        if re.search(r"\bresult\.equiv\s*\(", test_code):
            return "Statevector"

    # 4. Operator(solution).equiv(result) → result IS an Operator
    if re.search(r"Operator\s*\(\s*\w+\s*\)\s*\.equiv\s*\(\s*\w+\s*\)", test_code):
        if "Operator" in test_code and "equiv" in test_code:
            # Check if equiv argument is the result variable
            if re.search(r"\.equiv\s*\(\s*(?:result|output|candidate_result)\s*\)", test_code):
                return "Operator"

    # 5. result.num_qubits / result.data / result.layout accessed directly → QuantumCircuit
    if re.search(
        r"\b(?:result|circuit|qc|t_circuit|candidate_circuit|synthesized_qc)\s*\.\s*"
        r"(?:num_qubits|depth|data|layout|num_clbits|num_parameters)\b",
        test_code,
    ):
        return "QuantumCircuit"

    # 5b. candidate().num_parameters → QuantumCircuit (no assignment)
    if re.search(r"candidate\s*\(.*\)\s*\.num_parameters", test_code):
        return "QuantumCircuit"

    # 5c. type(result) == X pattern (using type() instead of isinstance)
    type_check = re.search(r"type\s*\(\s*\w+\s*\)\s*==\s*(\w+)", test_code)
    if type_check:
        typ = _type_from_string(type_check.group(1))
        if typ:
            return typ

    # 5d. Direct dict literal comparison: result == {"00": N, "11": M}
    if re.search(r'assert\s+\w+\s*==\s*\{["\'][01]+["\']', test_code):
        return "dict"

    # 5e. Direct numeric equality: assert result == 4.0 / assert complexity_can == N
    if re.search(r"assert\s+\w+\s*==\s*\d+[\d.]*\b", test_code) and not re.search(
        r"\.\s*(?:num_qubits|depth|data|layout)", test_code
    ):
        return "float"

    # 5f. Tuple unpacking from candidate: x, y = candidate()
    if re.search(r"\w+\s*,\s*\w+\s*=\s*candidate\s*\(", test_code):
        return "tuple"

    # 5g. candidate() called with result accessed as circuit .data
    if re.search(r"candidate\s*\(.*\)\s*\.data\b", test_code):
        return "QuantumCircuit"

    # 6. Return type annotation in extracted_code
    if extracted_code:
        ann_m = _RETURN_ANNOTATION.search(extracted_code)
        if ann_m:
            typ = _type_from_string(ann_m.group(1))
            if typ:
                return typ

        # 7. Return statement keywords in extracted_code
        try:
            tree = ast.parse(extracted_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Return) and node.value is not None:
                    ret_str = ast.unparse(node.value)
                    typ = _type_from_string(ret_str)
                    if typ:
                        return typ
        except SyntaxError:
            pass

    # 8. AerSimulator + measurement → dict
    if "AerSimulator" in test_code and (
        "get_counts" in test_code or "Sampler" in test_code
    ):
        return "dict"

    if not extracted_code and not test_code:
        return "unknown"

    return "other"


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    suite_path = Path("quantum_eval/_data/benchmarks/humaneval/humaneval.jsonl")
    examples = []
    with suite_path.open() as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    results = []
    for ex in examples:
        test_code = ex.get("test_code") or ""
        extracted_code = ex.get("extracted_code") or ""

        test_class = classify_test(test_code)
        output_type = classify_output_type(test_code, extracted_code)

        results.append({
            "id": ex["id"],
            "test_class": test_class,
            "output_type": output_type,
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    test_counts = Counter(r["test_class"] for r in results)
    output_counts = Counter(r["output_type"] for r in results)

    print("=== Test classification ===")
    for cls, n in sorted(test_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls:25s} {n:3d}")

    print()
    print("=== Canonical output types ===")
    for typ, n in sorted(output_counts.items(), key=lambda x: -x[1]):
        print(f"  {typ:25s} {n:3d}")

    # ── Spot-check behavioral ─────────────────────────────────────────────────
    print()
    print("=== Spot-check: behavioral (first 8) ===")
    beh = [r for r in results if r["test_class"] == "behavioral"]
    for r in beh[:8]:
        ex = next(e for e in examples if e["id"] == r["id"])
        tc = ex.get("test_code", "")
        signal = "(unknown)"
        for line in tc.splitlines():
            if any(
                sig in line
                for sig in [
                    ".equiv(", "from_instruction", "state_fidelity",
                    "AerSimulator", "sum(result",
                ]
            ):
                signal = line.strip()[:80]
                break
        print(f"  {r['id']:30s} output={r['output_type']:15s} ↳ {signal}")

    # ── Spot-check structural ─────────────────────────────────────────────────
    print()
    print("=== Spot-check: structural (first 8) ===")
    struct = [r for r in results if r["test_class"] == "structural"]
    for r in struct[:8]:
        ex = next(e for e in examples if e["id"] == r["id"])
        tc = ex.get("test_code", "")
        signal = "(unknown)"
        for line in tc.splitlines():
            if any(
                sig in line
                for sig in [
                    "num_qubits", "depth(", "operation.name", ".data[",
                    "num_clbits", "CircuitInstruction",
                ]
            ):
                signal = line.strip()[:80]
                break
        print(f"  {r['id']:30s} output={r['output_type']:15s} ↳ {signal}")

    # ── Interface ─────────────────────────────────────────────────────────────
    print()
    print("=== Interface tests (all) ===")
    iface = [r for r in results if r["test_class"] == "interface"]
    for r in iface:
        ex = next(e for e in examples if e["id"] == r["id"])
        tc = (ex.get("test_code") or "").strip()[:120]
        print(f"  {r['id']:30s} output={r['output_type']:15s}")
        print(f"    {tc}")

    # ── Write classification JSON ─────────────────────────────────────────────
    out_path = Path("results/v11_test_classification.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nClassification written to {out_path}")


if __name__ == "__main__":
    main()
