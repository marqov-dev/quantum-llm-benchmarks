# v1.1 Semantic Validator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the benchmark from "does the code run?" to "does the code produce the correct quantum output?" using two parallel approaches — IBM unit test execution (Approach A) and KL divergence on measurement distributions (Approach B) — then compare them retroactively across all 16 existing model result sets.

**Architecture:** Enrich `humaneval.jsonl` with `test_code`, `canonical_solution`, and `entry_point` from IBM's public dataset. Add a KL divergence validator. Add a `rescore` script with resume support. Run a calibration pass (canonical vs canonical noise floor) before the full rescore. Produce a comparison analysis with Cohen's κ and McNemar's test.

**Tech Stack:** Python 3.12, Qiskit 2.x, qiskit-aer 0.15+, scipy 1.13+ (already in pyproject.toml), urllib (stdlib).

---

## Background and design decisions

IBM's `qiskit-community/qiskit-human-eval` dataset fields per problem:
- `prompt`: function signature and docstring (not a standalone program)
- `canonical_solution`: function *body* (continuation of prompt, not standalone)
- `test`: unit test code that calls the function by `entry_point` name
- `entry_point`: the function name the test calls

**Critical format mismatch:** `prompt + canonical_solution` defines a named function. Our leaderboard prompts elicit standalone programs. This affects both approaches:

- **Approach A (unit tests):** IBM's tests call `entry_point()`. Generated code that doesn't define that function fails regardless of correctness. Mitigation: detect whether `entry_point` is defined; if not, synthesise a wrapper that runs the generated code and returns the last QuantumCircuit. Add `defines_entry_point` flag to all rescore records to isolate format-mismatch effects in analysis.

- **Approach B (KL divergence):** The canonical runner must call `entry_point()` to get a circuit — executing `prompt + canonical_solution` at module scope only *defines* the function, never calls it. Without this fix, every reference run returns `[]` and Approach B produces zero passes. Additionally: generated code may not define `entry_point`, so the generated runner must also try calling it if present, falling back to scanning locals for QuantumCircuit objects.

**τ=0.05 and shot count:** QuanBench+ (arXiv:2604.08570, preprint March 2026) uses KL-divergence-based acceptance. The specific values τ=0.05 and 1024 shots must be verified against the paper text before going into METHODOLOGY.md — the abstract doesn't confirm them. Do not claim a venue (e.g. "ICLR 2026 Workshop") without verifying in the paper — cite as "arXiv:2604.08570 (preprint)" unless the paper text says otherwise. Run a calibration task (canonical vs canonical) across the full suite before relying on any fixed τ. For high-qubit circuits with many bins, 1024 shots may produce sampling noise that exceeds 0.05.

**AerSimulator reproducibility:** Set `seed_simulator` in all runner calls so that rescoring the same file twice produces identical KL values.

**Subprocess overhead:** 2,416 examples × 2 subprocesses × ~3s Qiskit/Aer cold-start = ~4 hours of import time alone. The rescore script must support resume-by-task-id (same pattern as the main CLI) so a crash 10 hours in doesn't lose everything.

---

## File map

**New files:**
- `scripts/enrich_suite.py` — fetches IBM dataset, adds fields to humaneval.jsonl
- `quantum_eval/kl_validator.py` — KL divergence validator (Approach B)
- `scripts/calibrate_kl.py` — noise floor calibration: canonical vs canonical
- `scripts/rescore_v11.py` — retroactive rescore with resume support
- `scripts/analyze_v11.py` — comparison analysis (Cohen's κ, McNemar's test)
- `tests/test_kl_validator.py` — tests for KL validator

**Modified files:**
- `quantum_eval/_data/benchmarks/humaneval/humaneval.jsonl` — add `test_code`, `canonical_solution`, `entry_point`
- `quantum_eval/harness.py` — add `test_code`, `canonical_solution`, `entry_point` to `SuiteExample`
- `quantum_eval/cli.py` — pass `test_code` from `SuiteExample` to `validate_example`
- `METHODOLOGY.md` — v1.1 section

---

## Task 1: Pre-flight checks

Two cheap checks before any implementation. Either can block later tasks if they fail.

- [ ] **Step 1: Verify the validator API exists**

```bash
.venv/bin/python -c "from quantum_eval.validator import validate_test_code, ValidationLevel; print('OK', ValidationLevel.SEMANTIC.value)"
```

Expected: `OK 3` (or whatever the integer value is).

**If this errors:** `validate_test_code` is not yet implemented. Insert Task 1.5 before proceeding:

> **Task 1.5 (conditional): Implement `validate_test_code` in `quantum_eval/validator.py`**
>
> Read `validator.py` to understand the existing `validate_example` structure and `ValidationLevel` enum.
> Add a `validate_test_code(code: str, test_code: str, timeout: int = 60) -> ValidationResult` function
> that executes `code + "\n" + test_code` in a subprocess and returns a `ValidationResult` with
> `level_passed=ValidationLevel.SEMANTIC` on assertion success.
>
> **Critical: raise on subprocess errors, return on assertion failures.** The three-state semantics
> in the rescore (`unit_test_pass=True/False/None`) depend on this distinction:
> - If assertions fail (exit code 1, AssertionError in stderr): return `ValidationResult` with a
>   failed level so the rescore sets `unit_test_pass=False` ("ran and failed").
> - If the subprocess crashes before reaching assertions (import error, SyntaxError, timeout):
>   **raise** `RuntimeError` so the rescore catches it and sets `unit_test_pass=None`
>   ("couldn't evaluate"). This aligns with Approach B's None semantics.
>
> Mirror the existing subprocess pattern in `validate_example`. Write a test for it in
> `tests/test_validator.py` that verifies: (1) valid code + passing test → SEMANTIC, (2) valid code
> + failing assertion → returns with error (not raises), (3) broken code → raises RuntimeError.
> Re-run Step 1 above to confirm it passes before proceeding to Task 6.

**If Step 1 succeeds**, verify the existing function's raise/return behaviour before trusting it:

```bash
.venv/bin/python -c "
from quantum_eval.validator import validate_test_code
# Should RAISE (subprocess crash before assertions)
try:
    validate_test_code('raise RuntimeError(\"boom\")', 'assert True')
    print('PROBLEM: did not raise on subprocess crash — sets unit_test_pass=False instead of None')
except Exception as e:
    print('OK: raises on crash:', type(e).__name__)
# Should RETURN (code runs but assertion fails)
try:
    r = validate_test_code('x = 1', 'assert x == 2')
    print('OK: returns on assertion failure:', r)
except Exception as e:
    print('PROBLEM: raises on assertion failure — sets unit_test_pass=None instead of False:', e)
"
```

If the behaviour doesn't match the expected convention, fix `validate_test_code` before proceeding to Task 6. Document the actual behaviour in METHODOLOGY.md.

Do not proceed to Task 6 without Step 1 passing and the raise/return convention confirmed.

- [ ] **Step 2: Inspect a stored generated_code field for markdown fences**

`defines_entry_point_fn` and `synthesise_wrapper` both call `ast.parse` on stored `generated_code`. If the field contains raw model output (with ` ```python ... ``` ` fences or prose), `ast.parse` will fail and every row gets `defines_entry_point=False`.

```bash
.venv/bin/python -c "
import json
from pathlib import Path
path = Path('results/claude-sonnet-4-6_humaneval.jsonl')
rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
results = [r for r in rows if not r.get('_header')]
code = results[0].get('generated_code', '')
print('First 200 chars:', repr(code[:200]))
print('Has fences:', '\`\`\`' in code)
"
```

If `generated_code` contains fences: add a code-extraction step in `rescore_v11.py` before calling `defines_entry_point_fn` or `synthesise_wrapper`. The existing `validate_example` extractor in `validator.py` handles this — check its interface and use it.

- [ ] **Step 3: Verify QuanBench+ citation**

Fetch arXiv:2604.08570 and confirm:
- Is τ=0.05 stated explicitly in the paper?
- Is 1024 shots stated explicitly?
- Is the venue "ICLR 2026 Workshop" confirmed, or is it a standalone arXiv preprint?

Record the exact quote and section number. If the values are not confirmed, use them as reasonable defaults but attribute as "following QuanBench+ methodology" rather than "QuanBench+ calibrated values."

- [ ] **Step 4: Note findings**

Update the comment in `kl_validator.py` (Task 4) with the verified citation before committing.

---

## Task 2: Fetch IBM dataset and enrich humaneval.jsonl

**Files:**
- Create: `scripts/enrich_suite.py`
- Modify: `quantum_eval/_data/benchmarks/humaneval/humaneval.jsonl`

- [ ] **Step 1: Write enrich_suite.py**

```python
#!/usr/bin/env python3
"""Enrich humaneval.jsonl with IBM's test_code, canonical_solution, and entry_point.

Fetches from qiskit-community/qiskit-human-eval GitHub repo (no extra deps).
Matches on task_id. Writes in-place after backing up original.
"""
import json
import shutil
import urllib.request
from pathlib import Path

SUITE_PATH = Path("quantum_eval/_data/benchmarks/humaneval/humaneval.jsonl")
IBM_GITHUB_URL = (
    "https://raw.githubusercontent.com/qiskit-community/qiskit-human-eval"
    "/main/dataset/dataset_qiskit_test_human_eval.json"
)


def fetch_ibm_data() -> dict[str, dict]:
    """Return {task_id: {canonical_solution, test_code, entry_point}}."""
    print(f"Fetching IBM dataset...")
    with urllib.request.urlopen(IBM_GITHUB_URL, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
    data = json.loads(raw)
    problems = data if isinstance(data, list) else data.get("problems", [])
    result = {}
    for p in problems:
        task_id = p.get("task_id", "")
        result[task_id] = {
            "canonical_solution": p.get("canonical_solution", ""),
            "test_code": p.get("test", ""),
            "entry_point": p.get("entry_point", ""),
        }
    print(f"  Loaded {len(result)} problems.")
    return result


def enrich_suite(suite_path: Path, ibm_data: dict) -> None:
    backup = suite_path.with_suffix(".jsonl.bak")
    shutil.copy2(suite_path, backup)
    print(f"  Backed up to {backup}")
    lines = suite_path.read_text(encoding="utf-8").splitlines()
    enriched, matched = [], 0
    for line in lines:
        if not line.strip():
            continue
        ex = json.loads(line)
        if ex["id"] in ibm_data:
            ex.update(ibm_data[ex["id"]])
            matched += 1
        else:
            print(f"  WARNING: {ex['id']} not found in IBM dataset")
        enriched.append(json.dumps(ex))
    suite_path.write_text("\n".join(enriched) + "\n", encoding="utf-8")
    print(f"  Enriched {matched}/{len(lines)} examples.")


if __name__ == "__main__":
    ibm_data = fetch_ibm_data()
    enrich_suite(SUITE_PATH, ibm_data)
    print("Done. Suite hash has changed — this is the v1.1 suite.")
```

- [ ] **Step 2: Run it**

```bash
.venv/bin/python scripts/enrich_suite.py
```

Expected: `Enriched 151/151 examples.`

- [ ] **Step 3: Inspect one example to verify all three fields are populated**

```bash
.venv/bin/python -c "
import json
with open('quantum_eval/_data/benchmarks/humaneval/humaneval.jsonl') as f:
    ex = json.loads(f.readline())
print('Fields:', list(ex.keys()))
print('entry_point:', repr(ex.get('entry_point')))
print('canonical_solution[:80]:', repr(ex.get('canonical_solution', '')[:80]))
print('test_code[:80]:', repr(ex.get('test_code', '')[:80]))
"
```

Expected: all three fields non-empty strings for the first example.

- [ ] **Step 4: Verify 151/151 have all three fields**

```bash
.venv/bin/python -c "
import json
missing = []
with open('quantum_eval/_data/benchmarks/humaneval/humaneval.jsonl') as f:
    for line in f:
        ex = json.loads(line)
        if not ex.get('test_code') or not ex.get('canonical_solution') or not ex.get('entry_point'):
            missing.append(ex['id'])
print('Missing:', len(missing))
if missing: print(missing[:5])
"
```

Expected: `Missing: 0`

- [ ] **Step 5: Commit**

```bash
git add quantum_eval/_data/benchmarks/humaneval/humaneval.jsonl scripts/enrich_suite.py
git commit -m "feat: enrich humaneval suite with IBM test_code, canonical_solution, entry_point"
```

---

## Task 3: Update SuiteExample to carry new fields; wire into CLI

**Files:**
- Modify: `quantum_eval/harness.py`
- Modify: `quantum_eval/cli.py`
- Modify: `tests/test_harness.py`

- [ ] **Step 1: Write failing tests in test_harness.py**

Add to `tests/test_harness.py`:

```python
def test_load_suite_carries_ibm_fields(tmp_path):
    f = tmp_path / "suite.jsonl"
    f.write_text(
        '{"id": "qiskitHumanEval_0", "instruction": "create a Bell state", '
        '"test_code": "assert qc is not None", '
        '"canonical_solution": "    qc = QuantumCircuit(2)\\n    return qc", '
        '"entry_point": "create_bell"}\n'
    )
    examples = load_suite(f)
    assert examples[0].test_code == "assert qc is not None"
    assert examples[0].canonical_solution == "    qc = QuantumCircuit(2)\n    return qc"
    assert examples[0].entry_point == "create_bell"


def test_load_suite_missing_ibm_fields_are_none(tmp_path):
    f = tmp_path / "suite.jsonl"
    f.write_text('{"id": "qiskitHumanEval_0", "instruction": "test"}\n')
    examples = load_suite(f)
    assert examples[0].test_code is None
    assert examples[0].canonical_solution is None
    assert examples[0].entry_point is None
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/test_harness.py::test_load_suite_carries_ibm_fields -v
```

Expected: `FAILED` — `AttributeError: 'SuiteExample' object has no attribute 'test_code'`

- [ ] **Step 3: Update SuiteExample and load_suite in harness.py**

Replace the existing `SuiteExample` dataclass and `load_suite` function:

```python
@dataclass
class SuiteExample:
    id: str
    prompt: str
    category: str
    test_code: str | None = None
    canonical_solution: str | None = None
    entry_point: str | None = None


def load_suite(suite_path: Path) -> list[SuiteExample]:
    """Load suite from JSONL. Each line must have 'id' and 'instruction' or 'prompt'."""
    examples = []
    with suite_path.open() as f:
        for line in f:
            r = json.loads(line)
            if "instruction" in r:
                prompt = r["instruction"]
            elif "prompt" in r:
                prompt = r["prompt"]
            else:
                raise ValueError(f"Example {r.get('id')} has neither 'instruction' nor 'prompt' field")
            examples.append(SuiteExample(
                id=r["id"],
                prompt=prompt,
                category=r.get("category", "humaneval"),
                test_code=r.get("test_code"),
                canonical_solution=r.get("canonical_solution"),
                entry_point=r.get("entry_point"),
            ))
    return examples
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/test_harness.py -v
```

Expected: all pass.

- [ ] **Step 5: Wire test_code and entry_point into validate_example call in cli.py**

In `cmd_run`, find:
```python
        result, extracted_code = validate_example({"response": generated, "category": ex.category})
```

Replace with:
```python
        result, extracted_code = validate_example({
            "response": generated,
            "category": ex.category,
            "test_code": ex.test_code,
            "entry_point": ex.entry_point,
        })
```

- [ ] **Step 6: Run full test suite**

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add quantum_eval/harness.py quantum_eval/cli.py tests/test_harness.py
git commit -m "feat: wire entry_point and test_code through SuiteExample to validate_example"
```

---

## Task 4: KL divergence validator

**Files:**
- Create: `quantum_eval/kl_validator.py`
- Create: `tests/test_kl_validator.py`

**Critical design note on the runners:** IBM's `canonical_solution` is a function body, not a standalone program. `prompt + canonical_solution` defines a function named `entry_point` but never calls it. The reference runner MUST call `entry_point()` to obtain a circuit. The generated-code runner must try calling `entry_point()` if defined, falling back to scanning locals for QuantumCircuit objects.

Additional fixes applied here:
- AerSimulator called with `seed_simulator` for reproducibility
- Measurement detection uses instruction name check, not `num_clbits == 0`
- Multi-register bitstring keys normalised to remove spaces
- `warnings.filterwarnings` narrowed to `DeprecationWarning` from non-Qiskit modules only

- [ ] **Step 1: Write failing tests**

```python
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
# We test the reference runner by providing the full function + entry_point.
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
    assert total == 1024
    assert set(counts.keys()).issubset({"00", "11"})


def test_run_circuit_reference_with_entry_point():
    """Reference runner: calls entry_point() to get circuit."""
    full_code = BELL_PROMPT + BELL_CANONICAL
    counts = run_circuit_and_get_counts(
        full_code, shots=1024, seed=42, entry_point="create_bell_state"
    )
    total = sum(counts.values())
    assert total == 1024
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
    assert total == 1024
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
    assert total == 1024
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/test_kl_validator.py -v
```

Expected: `FAILED` — `ModuleNotFoundError: No module named 'quantum_eval.kl_validator'`

- [ ] **Step 3: Implement kl_validator.py**

```python
# quantum_eval/kl_validator.py
"""
Approach B: KL divergence on measurement distributions.

Runs reference circuit (via entry_point function call) and generated circuit
on AerSimulator with a fixed seed. Computes KL(P_gen || P_ref).
Passes if divergence < tau (default 0.05, following QuanBench+ arXiv:2604.08570).

Design decisions:
- Reference runner calls entry_point() — canonical_solution is a function body, not standalone.
- Generated runner tries entry_point() first, falls back to scanning locals.
- AerSimulator seed is fixed for reproducibility.
- Measurement detection uses instruction name, not num_clbits.
- Multi-register keys normalised to remove spaces.
- DeprecationWarning suppressed only for non-Qiskit-core modules.
"""
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

TAU = 0.05
DEFAULT_SHOTS = 1024
DEFAULT_SEED = 42
EPSILON = 1e-10

# Runner for the REFERENCE circuit.
# canonical_solution is a function body: prompt + canonical_solution = function def.
# We call entry_point() to obtain the QuantumCircuit.
_REF_RUNNER = '''
import json, sys, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit")
import matplotlib
matplotlib.use("Agg")

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

{code}

_entry = "{entry_point}"
_ns = dict(locals())
if _entry and _entry in _ns and callable(_ns[_entry]):
    _result = _ns[_entry]()
    if isinstance(_result, QuantumCircuit):
        _qc = _result
    else:
        print(json.dumps({{"error": f"entry_point returned {{type(_result).__name__}}, expected QuantumCircuit"}}))
        sys.exit(0)
else:
    print(json.dumps({{"error": f"entry_point function {{_entry!r}} not found in namespace"}}))
    sys.exit(0)

_has_meas = any(inst.operation.name == "measure" for inst in _qc.data)
if not _has_meas:
    _qc = _qc.copy()
    _qc.measure_all()

_sim = AerSimulator()
_raw = _sim.run(_qc, shots={shots}, seed_simulator={seed}).result().get_counts()
_counts = {{"".join(k.split()): v for k, v in _raw.items()}}
_total = sum(_counts.values())
print(json.dumps({{"counts": _counts, "total": _total}}))
'''

# Runner for the GENERATED circuit.
# Model output is standalone code. Try calling entry_point() if defined;
# otherwise scan locals for QuantumCircuit objects.
_GEN_RUNNER = '''
import json, sys, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit")
import matplotlib
matplotlib.use("Agg")

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

{code}

_entry = "{entry_point}"
_ns = dict(locals())
_qc = None

if _entry and _entry in _ns and callable(_ns[_entry]):
    _result = _ns[_entry]()
    if isinstance(_result, QuantumCircuit):
        _qc = _result

if _qc is None:
    _circuits = [v for v in _ns.values() if isinstance(v, QuantumCircuit)]
    if _circuits:
        _qc = _circuits[-1]

if _qc is None:
    print(json.dumps({{"error": "no QuantumCircuit found in generated code"}}))
    sys.exit(0)

_has_meas = any(inst.operation.name == "measure" for inst in _qc.data)
if not _has_meas:
    _qc = _qc.copy()
    _qc.measure_all()

_sim = AerSimulator()
_raw = _sim.run(_qc, shots={shots}, seed_simulator={seed}).result().get_counts()
_counts = {{"".join(k.split()): v for k, v in _raw.items()}}
_total = sum(_counts.values())
print(json.dumps({{"counts": _counts, "total": _total}}))
'''


@dataclass
class KLResult:
    passed: Optional[bool]   # True=pass, False=divergent, None=couldn't evaluate (error)
    kl_divergence: Optional[float]
    error: Optional[str] = None


def _run_template(template: str, code: str, entry_point: str, shots: int, seed: int) -> dict[str, float]:
    """Execute a runner template in a subprocess. Returns normalised probability dict.

    JSON parsing: scans stdout lines in reverse to find the runner's JSON output.
    Generated code may contain print() calls that land on stdout before the JSON line.
    json.loads(full_stdout) would raise JSONDecodeError on those rows — silently
    degrading Approach B on any model that added debug prints, circuit diagrams, etc.
    Reverse iteration is O(1) for the common case (runner JSON is always the last line).
    """
    runner = template.format(code=code, entry_point=entry_point, shots=shots, seed=seed)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(runner)
        tmp = Path(f.name)
    try:
        result = subprocess.run(
            [sys.executable, str(tmp)],
            capture_output=True, text=True, timeout=60,
            errors="replace",  # avoid UnicodeDecodeError if model emits non-UTF-8
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip()[-500:] or "non-zero exit")
        import json as _json
        data = None
        for line in reversed(result.stdout.strip().splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                data = _json.loads(line)
                break
            except _json.JSONDecodeError:
                continue
        if data is None:
            raise RuntimeError("no parseable JSON in subprocess output")
        if "error" in data:
            raise RuntimeError(data["error"])
        counts = data["counts"]
        total = data["total"]
        return {k: v / total for k, v in counts.items()}
    except subprocess.TimeoutExpired:
        raise RuntimeError("timeout")  # caller sees "timeout" in kl_error vs other errors
    finally:
        tmp.unlink(missing_ok=True)


def run_circuit_and_get_counts(
    code: str,
    shots: int = DEFAULT_SHOTS,
    seed: int = DEFAULT_SEED,
    entry_point: str = "",
) -> dict[str, float]:
    """Run code in subprocess using the generated-code runner; return normalised probability dict.

    If entry_point is given and defined in the code, calls it to get the circuit.
    Otherwise scans locals for QuantumCircuit objects.
    Raises RuntimeError on failure.
    """
    return _run_template(_GEN_RUNNER, code, entry_point, shots, seed)


def run_canonical_circuit(
    canonical_code: str,
    entry_point: str,
    shots: int = DEFAULT_SHOTS,
    seed: int = DEFAULT_SEED,
) -> dict[str, float]:
    """Run canonical code using the reference runner; return normalised probability dict.

    canonical_code should be prompt + canonical_solution (a complete function definition).
    Calls entry_point() to obtain the circuit — does NOT scan locals.
    This is what the calibration script uses: run twice with different seeds to
    measure sampling noise, not model error.
    Raises RuntimeError on failure.
    """
    return _run_template(_REF_RUNNER, canonical_code, entry_point, shots, seed)


def kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """KL(P || Q) with additive smoothing. p=generated, q=reference."""
    import math
    all_keys = set(p) | set(q)
    total = 0.0
    for k in all_keys:
        pk = p.get(k, 0.0) + EPSILON
        qk = q.get(k, 0.0) + EPSILON
        total += pk * math.log(pk / qk)
    return total


def validate_kl_divergence(
    generated_code: str,
    canonical_code: str,
    entry_point: str = "",
    shots: int = DEFAULT_SHOTS,
    tau: float = TAU,
    seed: int = DEFAULT_SEED,
) -> KLResult:
    """Compare measurement distributions of generated vs reference circuit.

    canonical_code should be prompt + canonical_solution (complete function def).
    entry_point is the function name to call on the reference side.
    Never raises — errors captured in KLResult.error.
    """
    try:
        ref_dist = _run_template(_REF_RUNNER, canonical_code, entry_point, shots, seed)
    except Exception as e:
        # passed=None: couldn't evaluate (not the same as "evaluated and divergent")
        return KLResult(passed=None, kl_divergence=None, error=f"Reference circuit error: {e}")

    try:
        gen_dist = _run_template(_GEN_RUNNER, generated_code, entry_point, shots, seed)
    except Exception as e:
        return KLResult(passed=None, kl_divergence=None, error=f"Generated circuit error: {e}")

    kl = kl_divergence(gen_dist, ref_dist)
    # passed=True: distributions agree within tau; passed=False: evaluated but divergent
    return KLResult(passed=kl < tau, kl_divergence=round(kl, 6))
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/test_kl_validator.py -v
```

Expected: all 8 tests pass. Bell state tests may take 10–20s due to subprocess overhead.

- [ ] **Step 5: Run full test suite**

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add quantum_eval/kl_validator.py tests/test_kl_validator.py
git commit -m "feat: KL divergence validator (Approach B) with entry_point runner fix"
```

---

## Task 5: Noise floor calibration

**Before the full rescore**, characterise KL divergence between two independent runs of the same canonical circuit. This establishes whether τ=0.05 is appropriate or needs adjustment.

**Files:**
- Create: `scripts/calibrate_kl.py`

- [ ] **Step 1: Implement calibrate_kl.py**

```python
#!/usr/bin/env python3
"""
Calibration: measure KL divergence between two independent runs of the same
canonical circuit (different seeds, same code). This is the sampling noise floor —
the baseline divergence you'd get even with a perfectly correct model.

If the 95th percentile KL exceeds tau=0.05, either increase shots or raise tau.

IMPORTANT: Do NOT use validate_kl_divergence(code, code, seed=i) for this.
That function passes the same seed to both runners, giving identical distributions
and KL≈0. We need KL(P_seed_i ‖ P_seed_j) — two independent samples of the same ideal
distribution. Use run_canonical_circuit() directly.
"""
import itertools
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_eval.kl_validator import run_canonical_circuit, kl_divergence, TAU

SUITE_PATH = Path("quantum_eval/_data/benchmarks/humaneval/humaneval.jsonl")
N_RUNS = 5        # runs per example; pairs = N*(N-1)/2 = 10
SHOTS = 1024
MAX_EXAMPLES = 30  # calibrate on a random sample (see stratification note below)
SAMPLE_SEED = 42   # for reproducible random.sample


def load_examples(suite_path: Path):
    examples = []
    with suite_path.open() as f:
        for line in f:
            ex = json.loads(line)
            if ex.get("canonical_solution") and ex.get("entry_point"):
                examples.append(ex)
    return examples


def build_canonical_code(ex: dict) -> str:
    """Reconstruct full function definition from prompt + canonical_solution."""
    prompt = ex.get("instruction") or ex.get("prompt", "")
    return prompt + ex["canonical_solution"]


def main():
    examples = load_examples(SUITE_PATH)
    # Use random.sample rather than first-N: qubit count and circuit complexity
    # vary across the suite, and 2-qubit circuits tell you nothing about 8-qubit noise.
    rng = random.Random(SAMPLE_SEED)
    sample = rng.sample(examples, min(MAX_EXAMPLES, len(examples)))
    print(f"Calibrating on {len(sample)} examples (random sample, seed={SAMPLE_SEED}), {N_RUNS} runs each...")

    all_kl = []

    for ex in sample:
        canonical_code = build_canonical_code(ex)
        entry_point = ex["entry_point"]
        kl_values = []

        # Run the same canonical circuit N times with different seeds.
        # Compute KL between every pair of independent runs.
        # This measures sampling noise, not model error.
        seeds = list(range(N_RUNS))
        distributions = {}
        for seed in seeds:
            try:
                distributions[seed] = run_canonical_circuit(
                    canonical_code, entry_point=entry_point, shots=SHOTS, seed=seed
                )
            except Exception as e:
                print(f"  {ex['id']} seed={seed}: FAILED ({e})")

        # Note: these N*(N-1)/2 pairs are not iid — pairs sharing a seed share an empirical
        # distribution, so a tail sample at seed=0 inflates all four pairs touching it.
        # Fine for the purpose of confirming tau is in the right ballpark; do not treat
        # the 300 aggregate pairs as 300 independent observations.
        for i, j in itertools.combinations(seeds, 2):
            if i in distributions and j in distributions:
                kl_values.append(kl_divergence(distributions[i], distributions[j]))

        if kl_values:
            median = sorted(kl_values)[len(kl_values) // 2]
            print(f"  {ex['id']}: median_kl={median:.4f}  n_pairs={len(kl_values)}")
            all_kl.extend(kl_values)

    if not all_kl:
        print("No KL values collected — check that run_canonical_circuit returns distributions.")
        return

    all_kl.sort()
    n = len(all_kl)
    p50 = all_kl[n // 2]
    p95 = all_kl[int(n * 0.95)]
    p99 = all_kl[int(n * 0.99)]

    print(f"\nNoise floor KL distribution (n={n} pairs across {len(sample)} examples):")
    print(f"  p50: {p50:.4f}")
    print(f"  p95: {p95:.4f}")
    print(f"  p99: {p99:.4f}")
    print(f"  max: {max(all_kl):.4f}")
    print(f"\nConfigured tau: {TAU}")
    if p95 > TAU:
        print(f"  WARNING: 95th percentile ({p95:.4f}) exceeds tau ({TAU}). "
              "Consider raising tau or increasing shots before the full rescore.")
    else:
        print(f"  OK: tau={TAU} is above the 95th percentile noise floor.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run calibration**

```bash
.venv/bin/python scripts/calibrate_kl.py
```

This runs ~30 examples × 10 pairs = 300 KL computations. Takes ~10 min.

Expected output example:
```
Calibrating on 30 examples (random sample, seed=42), 5 runs each...
  qiskitHumanEval_0: median_kl=0.0012
  ...
Noise floor KL distribution (n=300 pairs):
  p50: 0.0015
  p95: 0.0089
  p99: 0.0134
  max: 0.0203
Configured tau: 0.05
  OK: tau=0.05 is above 95th percentile noise floor.
```

- [ ] **Step 3: If p95 > 0.05, raise tau or shots**

If calibration shows p95 > 0.05: either double shots to 2048 and re-run, or set `TAU = 0.10` in `kl_validator.py`. Document the decision and the calibration numbers in METHODOLOGY.md (Task 8). Do NOT proceed to the full rescore with a tau that the noise floor exceeds.

- [ ] **Step 4: Canonical sanity pass — run every canonical solution once**

Run all 151 canonical solutions through `run_canonical_circuit` and report failures. If a canonical fails (Qiskit API drift, missing import, etc.), every model gets `kl_pass=None` on that row uniformly — a silent 8–16 hour run would only reveal this in elevated `n_errored`. A 10-minute check now is cheap insurance.

```python
# scripts/check_canonicals.py (write inline, don't commit)
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from quantum_eval.kl_validator import run_canonical_circuit

suite = [json.loads(l) for l in Path("quantum_eval/_data/benchmarks/humaneval/humaneval.jsonl").read_text().splitlines()]
failures = []
for ex in suite:
    ep = ex.get("entry_point", "")
    cs = ex.get("canonical_solution", "")
    prompt = ex.get("instruction") or ex.get("prompt", "")
    if not ep or not cs:
        continue
    try:
        counts = run_canonical_circuit(prompt + cs, entry_point=ep, shots=256, seed=42)
        assert sum(counts.values()) > 0
    except Exception as e:
        failures.append((ex["id"], str(e)[:120]))

print(f"Canonical sanity: {len(suite) - len(failures)}/151 OK, {len(failures)} failed")
for task_id, err in failures:
    print(f"  FAIL {task_id}: {err}")
```

```bash
.venv/bin/python scripts/check_canonicals.py
```

Expected: `151/151 OK`. If failures appear, inspect whether they're Qiskit API issues or IBM dataset problems. Do not proceed to the full rescore if more than a handful fail — the analysis will be unreliable.

- [ ] **Step 6: Commit calibration script and any tau adjustment**

```bash
git add scripts/calibrate_kl.py
# If tau was adjusted:
# git add quantum_eval/kl_validator.py
git commit -m "feat: KL noise floor calibration script; tau verified against canonical-vs-canonical runs"
```

---

## Task 6: Retroactive rescore script

**Files:**
- Create: `scripts/rescore_v11.py`

Key features:
- Resume-by-task-id (skip already-completed rows)
- Records both v1.0 suite hash (from original header) and v1.1 suite hash (enriched suite)
- Adds `defines_entry_point` flag per row (needed to isolate format-mismatch effects in analysis)
- Unit test (Approach A) attempts function wrapper synthesis when `entry_point` not defined in generated code

- [ ] **Step 1: Implement rescore_v11.py**

```python
#!/usr/bin/env python3
"""
Retroactive v1.1 rescoring: run Approach A (unit tests) and Approach B (KL divergence)
on stored generated_code from existing result JSONLs.

Usage:
    .venv/bin/python scripts/rescore_v11.py results/claude-sonnet-4-6_humaneval.jsonl
    .venv/bin/python scripts/rescore_v11.py results/*_humaneval.jsonl

Output: results/v11/<original_name> — new JSONL, extra fields per row:
    unit_test_pass:      bool | null
    kl_pass:             bool | null
    kl_divergence:       float | null
    kl_error:            str | null
    defines_entry_point: bool | null

Resumable: already-scored rows in the output file are skipped.
"""
import ast
import json
import sys
from pathlib import Path

SUITE_PATH = Path("quantum_eval/_data/benchmarks/humaneval/humaneval.jsonl")
OUTPUT_DIR = Path("results/v11")


def load_suite_index(suite_path: Path) -> dict[str, dict]:
    index = {}
    with suite_path.open() as f:
        for line in f:
            ex = json.loads(line)
            index[ex["id"]] = {
                "test_code": ex.get("test_code"),
                "canonical_solution": ex.get("canonical_solution"),
                "entry_point": ex.get("entry_point", ""),
                "prompt": ex.get("instruction") or ex.get("prompt", ""),
            }
    return index


def compute_suite_hash(suite_path: Path) -> str:
    import hashlib
    return "sha256:" + hashlib.sha256(suite_path.read_bytes()).hexdigest()


def defines_entry_point_fn(generated_code: str, entry_point: str) -> bool:
    """Return True if generated_code defines entry_point as a TOP-LEVEL function.

    Uses tree.body (not ast.walk) to avoid matching nested definitions like:
        def helper():
            def create_bell_state(): ...  # nested — not callable from module scope
    """
    if not entry_point or not generated_code:
        return False
    try:
        tree = ast.parse(generated_code)
        return any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == entry_point
            for node in tree.body  # top-level only
        )
    except SyntaxError:
        return False


def has_toplevel_return(code: str) -> bool:
    """Return True if code has a top-level return statement (invalid inside a function body)."""
    try:
        tree = ast.parse(code)
        return any(isinstance(node, ast.Return) for node in tree.body)
    except SyntaxError:
        return False  # already broken; let the test runner handle it


def synthesise_wrapper(generated_code: str, entry_point: str) -> str:
    """Wrap standalone generated_code in a function named entry_point.

    Inlines the entire generated code into the function body. The function
    then returns the last QuantumCircuit found in its locals.

    Assumptions:
    - generated_code has no top-level return statements (check with has_toplevel_return).
    - generated_code has no `from __future__` imports (those must be at module top;
      indenting them into a function body raises SyntaxError). Rare in Qiskit code but
      screen for it: check `"from __future__" in generated_code` before calling.
    - Uses expandtabs(4) before indenting to avoid mixing tabs and spaces at adjacent
      indent levels, which raises TabError in Python 3.
    """
    import textwrap
    normalised = generated_code.expandtabs(4)
    indented = textwrap.indent(normalised, "    ")
    return (
        f"from qiskit import QuantumCircuit\n\n"
        f"def {entry_point}():\n"
        f"{indented}\n"
        f"    _circuits = [v for v in dict(locals()).values() if isinstance(v, QuantumCircuit)]\n"
        f"    return _circuits[-1] if _circuits else None\n"
    )


def get_completed_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    completed = set()
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("_header"):
                continue
            if "id" in r:
                completed.add(r["id"])
    return completed


def rescore_file(jsonl_path: Path, suite_index: dict, v11_suite_hash: str) -> Path:
    from quantum_eval.validator import validate_test_code, ValidationLevel
    from quantum_eval.kl_validator import validate_kl_divergence

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / jsonl_path.name

    rows = [l for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    completed = get_completed_ids(out_path)

    result_rows = [r for r in rows if not json.loads(r).get("_header")]
    todo = [r for r in result_rows if json.loads(r).get("id") not in completed]

    print(f"\n{jsonl_path.name}: {len(result_rows)} total, {len(completed)} done, {len(todo)} to score")
    if not todo:
        print("  Already complete.")
        return out_path

    # Write header on first run
    if not out_path.exists() or len(completed) == 0:
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                if json.loads(row).get("_header"):
                    header = json.loads(row)
                    header["v10_suite_hash"] = header.pop("suite_hash", None)
                    header["v11_suite_hash"] = v11_suite_hash
                    f.write(json.dumps(header) + "\n")
                    break

    with out_path.open("a", encoding="utf-8") as out_f:
        for idx, row in enumerate(todo):
            record = json.loads(row)
            task_id = record["id"]
            generated_code = record.get("generated_code", "")
            suite_entry = suite_index.get(task_id, {})
            test_code = suite_entry.get("test_code")
            entry_point = suite_entry.get("entry_point", "")
            prompt = suite_entry.get("prompt", "")
            canonical_solution = suite_entry.get("canonical_solution", "")

            # --- defines_entry_point flag ---
            dep = defines_entry_point_fn(generated_code, entry_point)
            record["defines_entry_point"] = dep

            # --- Approach A: unit test ---
            unit_test_pass = None
            if test_code and generated_code:
                code_for_test = generated_code
                if (not dep and entry_point
                        and not has_toplevel_return(generated_code)
                        and "from __future__" not in generated_code):
                    # Standalone code — synthesise a wrapper so IBM tests can call entry_point()
                    code_for_test = synthesise_wrapper(generated_code, entry_point)
                try:
                    ut_result = validate_test_code(code_for_test, test_code, timeout=60)
                    unit_test_pass = ut_result.level_passed.value >= ValidationLevel.SEMANTIC.value
                except Exception as e:
                    unit_test_pass = None  # None = couldn't evaluate; False = ran and failed
                    record["unit_test_error"] = str(e)
            record["unit_test_pass"] = unit_test_pass

            # --- Approach B: KL divergence ---
            kl_pass = None
            kl_divergence_val = None
            kl_error = None
            if canonical_solution and generated_code:
                full_canonical = prompt + canonical_solution
                kl_result = validate_kl_divergence(
                    generated_code, full_canonical,
                    entry_point=entry_point,
                )
                kl_pass = kl_result.passed
                kl_divergence_val = kl_result.kl_divergence
                kl_error = kl_result.error
            record["kl_pass"] = kl_pass
            record["kl_divergence"] = kl_divergence_val
            record["kl_error"] = kl_error

            out_f.write(json.dumps(record) + "\n")
            print(
                f"  [{len(completed) + idx + 1}/{len(result_rows)}] {task_id}"
                f"  dep={dep}  ut={unit_test_pass}  kl={kl_pass}  kl_div={kl_divergence_val}",
                flush=True,
            )

    print(f"  Written: {out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: rescore_v11.py <result.jsonl> [...]")
        sys.exit(1)

    suite_index = load_suite_index(SUITE_PATH)
    v11_hash = compute_suite_hash(SUITE_PATH)
    print(f"Suite index: {len(suite_index)} examples. v1.1 suite hash: {v11_hash[:24]}...")

    for arg in sys.argv[1:]:
        for path in sorted(Path(".").glob(arg)) if "*" in arg else [Path(arg)]:
            if not path.exists():
                print(f"SKIP: {path} not found")
                continue
            rescore_file(path, suite_index, v11_hash)

    print("\nAll done.")
```

- [ ] **Step 2: Smoke-test on 5 examples from two models**

```bash
# Extract first 5 result rows from two models for smoke test
.venv/bin/python -c "
from pathlib import Path
import json

for model in ['claude-sonnet-4-6', 'llama-4-scout']:
    src = Path(f'results/{model}_humaneval.jsonl') if 'sonnet' in model else Path(f'results/meta-llama_llama-4-scout-17b-16e-instruct_humaneval.jsonl')
    lines = src.read_text().splitlines()
    header = [l for l in lines if json.loads(l).get('_header')]
    results = [l for l in lines if not json.loads(l).get('_header')][:5]
    out = Path(f'/tmp/smoke_{model}.jsonl')
    out.write_text('\n'.join(header + results) + '\n')
    print(f'Created {out} ({len(results)} rows)')
"

.venv/bin/python scripts/rescore_v11.py /tmp/smoke_claude-sonnet-4-6.jsonl /tmp/smoke_llama-4-scout.jsonl
```

- [ ] **Step 3: Inspect smoke test output**

```bash
.venv/bin/python -c "
import json
from pathlib import Path
for f in sorted(Path('results/v11').glob('smoke_*.jsonl')):
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip() and not json.loads(l).get('_header')]
    print(f.name)
    for r in rows:
        print(f'  {r[\"id\"]}  dep={r.get(\"defines_entry_point\")}  ut={r.get(\"unit_test_pass\")}  kl={r.get(\"kl_pass\")}  kl_div={r.get(\"kl_divergence\")}')
"
```

Expected: rows show all four new fields (no KeyError), kl_divergence is a float or null, no uncaught exceptions.

- [ ] **Step 4: Commit**

```bash
git add scripts/rescore_v11.py
git commit -m "feat: v1.1 rescore script with resume, defines_entry_point, suite hash provenance"
```

---

## Task 7: Run the full rescore

- [ ] **Step 1: Start rescore on all 16 models**

```bash
nohup .venv/bin/python scripts/rescore_v11.py \
  results/claude-opus-4-7_humaneval.jsonl \
  results/claude-opus-4-6_humaneval.jsonl \
  results/claude-sonnet-4-6_humaneval.jsonl \
  results/gemini-2.5-flash_humaneval.jsonl \
  results/gemini-2.5-pro_humaneval.jsonl \
  results/deepseek-v4-pro_humaneval.jsonl \
  results/deepseek-v4-flash_humaneval.jsonl \
  results/qwen3-coder-480b-a35b-instruct_humaneval.jsonl \
  results/qwen3-coder-plus_humaneval.jsonl \
  results/codestral-latest_humaneval.jsonl \
  results/mistral-large-latest_humaneval.jsonl \
  results/moonshot-v1-8k_humaneval.jsonl \
  results/openai_gpt-oss-20b_humaneval.jsonl \
  results/openai_gpt-oss-120b_humaneval.jsonl \
  results/meta-llama_llama-4-scout-17b-16e-instruct_humaneval.jsonl \
  results/gemma4_latest_humaneval.jsonl \
  > /tmp/rescore_v11.log 2>&1 &
echo "Rescore PID: $!"
```

Expected wall-clock: 8–16 hours. The script is resumable — if interrupted, re-run the same command.

- [ ] **Step 2: Verify all 16 output files have 151 rows**

```bash
for f in results/v11/*_humaneval.jsonl; do
  count=$(grep -v '"_header"' "$f" | wc -l | tr -d ' ')
  echo "$f: $count/151"
done
```

Expected: 16 files, each showing `151`.

---

## Task 8: Analysis script

**Files:**
- Create: `scripts/analyze_v11.py`

- [ ] **Step 1: Implement analyze_v11.py**

```python
#!/usr/bin/env python3
"""
v1.1 comparison analysis: Approach A (unit tests) vs Approach B (KL divergence).

Produces:
  1. Per-model table: v1.0 semantic%, unit_test%, kl_semantic%, defines_entry_point%
  2. Overall 2x2 agreement matrix with Cohen's kappa
  3. McNemar's test on (A-only vs B-only) asymmetry
  4. Disagreement breakdown split by defines_entry_point flag
  5. Pearson correlations (per-example, n=2416)

Usage: .venv/bin/python scripts/analyze_v11.py
"""
import json
import math
from pathlib import Path

V11_DIR = Path("results/v11")


def load_results(jsonl_path: Path) -> list[dict]:
    rows = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("_header"):
                continue
            rows.append(r)
    return rows


def model_label(path: Path) -> str:
    """Derive display label from filename.

    Handles version dots: `claude-sonnet-4-6` → `Claude Sonnet 4.6`.
    Pattern: digit-digit sequences become digit.digit before title-casing.
    """
    import re
    stem = path.stem.replace("_humaneval", "")
    stem = re.sub(r"(\d)-(\d)", r"\1.\2", stem)  # 4-6 → 4.6
    return stem.replace("_", " ").replace("-", " ").title()


def agreement_matrix(rows: list[dict]) -> dict:
    # Only include rows where BOTH methods produced a definitive result (True or False).
    # None = couldn't evaluate (error/timeout). Excluding these means "both_fail" in the
    # matrix genuinely means "ran and failed" — not "one or both errored out."
    # Track n_errored separately so METHODOLOGY.md can report error rates alongside agreement.
    valid = [r for r in rows
             if r.get("unit_test_pass") is not None and r.get("kl_pass") is not None]
    n_errored = len(rows) - len(valid)
    aa = sum(1 for r in valid if r["unit_test_pass"] and r["kl_pass"])
    ab = sum(1 for r in valid if r["unit_test_pass"] and not r["kl_pass"])
    ba = sum(1 for r in valid if not r["unit_test_pass"] and r["kl_pass"])
    bb = sum(1 for r in valid if not r["unit_test_pass"] and not r["kl_pass"])
    return {"both_pass": aa, "a_only": ab, "b_only": ba, "both_fail": bb,
            "n": len(valid), "n_errored": n_errored}


def cohens_kappa(mat: dict) -> float:
    n = mat["n"]
    if n == 0:
        return 0.0
    p_o = (mat["both_pass"] + mat["both_fail"]) / n
    p_a = (mat["both_pass"] + mat["a_only"]) / n
    p_b = (mat["both_pass"] + mat["b_only"]) / n
    p_e = p_a * p_b + (1 - p_a) * (1 - p_b)
    return (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0


def mcnemar_p(mat: dict) -> float:
    """McNemar's test p-value (continuity-corrected chi-squared).

    Use for the overall row (n=2,416) where chi-squared is appropriate.
    Do NOT use for per-model cells where a_only+b_only may be single digits.
    """
    a_only, b_only = mat["a_only"], mat["b_only"]
    denom = a_only + b_only
    if denom == 0:
        return 1.0
    chi2 = (abs(a_only - b_only) - 1) ** 2 / denom
    from scipy.stats import chi2 as chi2_dist
    return float(chi2_dist.sf(chi2, 1))


def mcnemar_p_exact(mat: dict) -> float:
    """McNemar's exact binomial test.

    Use for per-model cells where a_only+b_only may be in the single digits.
    chi-squared is unreliable with small n.
    """
    from scipy.stats import binomtest
    a, b = mat["a_only"], mat["b_only"]
    if a + b == 0:
        return 1.0
    return float(binomtest(min(a, b), a + b, 0.5).pvalue)


def pct(k: int, n: int) -> str:
    return f"{round(k / n * 100)}%" if n else "N/A"


def pearsonr(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else 0.0


def main():
    files = sorted(V11_DIR.glob("*_humaneval.jsonl"))
    if not files:
        print(f"No v11 files in {V11_DIR}. Run rescore_v11.py first.")
        return

    all_rows = []
    model_stats = []

    print("\n" + "=" * 115)
    print(f"{'Model':<38} {'v1.0':>6} {'unit%':>6} {'kl%':>6} {'dep%':>6} {'err%':>6} {'κ':>6} {'agree':>7} {'p(McN)':>8}")
    print("=" * 115)

    for fpath in files:
        label = model_label(fpath)
        rows = load_results(fpath)
        n = len(rows)

        v10 = sum(1 for r in rows if r.get("semantic_pass"))
        ut = sum(1 for r in rows if r.get("unit_test_pass"))
        kl = sum(1 for r in rows if r.get("kl_pass"))
        dep = sum(1 for r in rows if r.get("defines_entry_point"))

        mat = agreement_matrix(rows)
        kappa = cohens_kappa(mat)
        p_mcn = mcnemar_p_exact(mat)  # exact binomial: per-model a_only+b_only may be tiny
        agree = mat["both_pass"] + mat["both_fail"]
        n_err = mat["n_errored"]  # rows where ≥1 method couldn't evaluate

        print(
            f"{label:<38} {pct(v10, n):>6} {pct(ut, n):>6} {pct(kl, n):>6}"
            f" {pct(dep, n):>6} {pct(n_err, n):>6} {kappa:>6.3f} {pct(agree, mat['n']):>7} {p_mcn:>8.4f}"
        )
        all_rows.extend(rows)
        model_stats.append({
            "label": label, "n": n,
            "v10_pct": round(v10 / n * 100),
            "unit_test_pct": round(ut / n * 100),
            "kl_pct": round(kl / n * 100),
        })

    # --- Overall agreement ---
    overall = agreement_matrix(all_rows)
    kappa_all = cohens_kappa(overall)
    p_all = mcnemar_p(overall)

    print("\n" + "=" * 105)
    print(f"Overall ({len(all_rows)} examples across {len(files)} models):")
    print(f"  Evaluable (both methods ran):  {overall['n']:>5}  {pct(overall['n'], len(all_rows))}")
    print(f"  Errored (>=1 method failed):   {overall['n_errored']:>5}  {pct(overall['n_errored'], len(all_rows))}")
    print(f"  Both pass (A∩B):     {overall['both_pass']:>5}  {pct(overall['both_pass'], overall['n'])}")
    print(f"  A only (unit test):  {overall['a_only']:>5}  {pct(overall['a_only'], overall['n'])}")
    print(f"  B only (KL div):     {overall['b_only']:>5}  {pct(overall['b_only'], overall['n'])}")
    print(f"  Both fail:           {overall['both_fail']:>5}  {pct(overall['both_fail'], overall['n'])}")
    print(f"  Cohen's κ:           {kappa_all:.3f}")
    print(f"  McNemar p-value:     {p_all:.4f}  {'(significant asymmetry)' if p_all < 0.05 else '(no significant asymmetry)'}")

    # --- Disagreement breakdown by defines_entry_point ---
    print("\n  Disagreement breakdown by defines_entry_point:")
    for dep_val in [True, False]:
        subset = [r for r in all_rows
                  if r.get("defines_entry_point") == dep_val
                  and r.get("unit_test_pass") is not None
                  and r.get("kl_pass") is not None]
        if not subset:
            continue
        a_only = sum(1 for r in subset if r["unit_test_pass"] and not r["kl_pass"])
        b_only = sum(1 for r in subset if not r["unit_test_pass"] and r["kl_pass"])
        label = "defines entry_point=True " if dep_val else "defines entry_point=False"
        print(f"    {label}: n={len(subset)}, A-only={a_only}, B-only={b_only}")

    # --- Correlations (per-example) ---
    v10_vals = [1 if r.get("semantic_pass") else 0 for r in all_rows if r.get("unit_test_pass") is not None]
    ut_vals = [1 if r.get("unit_test_pass") else 0 for r in all_rows if r.get("unit_test_pass") is not None]
    kl_vals = [1 if r.get("kl_pass") else 0 for r in all_rows if r.get("unit_test_pass") is not None]

    r_v10_ut = pearsonr(v10_vals, ut_vals)
    r_v10_kl = pearsonr(v10_vals, kl_vals)
    r_ut_kl = pearsonr(ut_vals, kl_vals)
    # For binary outcomes, Pearson r is the phi coefficient (not misleading, but name it correctly)
    print(f"\n  Per-example phi coefficient (binary Pearson, n={len(v10_vals)}):")
    print(f"    v1.0 semantic vs unit_test:  φ = {r_v10_ut:.3f}")
    print(f"    v1.0 semantic vs KL:         φ = {r_v10_kl:.3f}")
    print(f"    unit_test vs KL:             φ = {r_ut_kl:.3f}")

    # --- Save JSON ---
    out = Path("results/v11_analysis.json")
    out.write_text(json.dumps({
        "model_stats": model_stats,
        "overall_matrix": overall,  # n = evaluable rows only; n_errored = rows where ≥1 method failed
        "cohens_kappa": kappa_all,
        "mcnemar_p": p_all,
        "phi_coefficients": {
            "v10_vs_unit_test": r_v10_ut,
            "v10_vs_kl": r_v10_kl,
            "unit_test_vs_kl": r_ut_kl,
        },
    }, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run analysis (after Task 7 completes)**

```bash
.venv/bin/python scripts/analyze_v11.py
```

- [ ] **Step 3: Commit**

```bash
git add scripts/analyze_v11.py
git commit -m "feat: v1.1 analysis with Cohen's kappa, McNemar's test, defines_entry_point breakdown"
```

---

## Task 9: METHODOLOGY.md v1.1

**Files:**
- Modify: `METHODOLOGY.md`

- [ ] **Step 1: Append v1.1 section**

Add after the existing content (fill in `[VERIFIED_SHOTS]` and `[VERIFIED_TAU]` from Task 1 and Task 5):

```markdown
---

## v1.1 Semantic Validation (in progress)

**Status:** Results from v1.1 are not directly comparable to v1.0.

### Approach A: Unit test execution

Each problem includes hand-authored unit tests from IBM's public dataset
(`qiskit-community/qiskit-human-eval`, HuggingFace: `Qiskit/qiskit_humaneval`).
The generated code is executed, followed by the unit test assertions, in an
isolated subprocess with a 60-second timeout.

**Format mismatch:** IBM's tests call a function named `entry_point`. Our prompts
elicit standalone programs. If `entry_point` is not defined in the generated code,
a synthesised wrapper is used that runs the standalone code and captures the last
QuantumCircuit. Results are disaggregated by `defines_entry_point` (whether the
model happened to define the expected function) in the comparison analysis.

### Approach B: KL divergence on measurement distributions

The generated code and IBM's `canonical_solution` (invoked via `entry_point()`)
are each run on `AerSimulator` with `[VERIFIED_SHOTS]` shots and a fixed seed
(`seed_simulator=42`). The KL divergence between output distributions is computed
with additive smoothing (ε=1e-10).

**Pass criterion:** KL(P_gen ∥ P_ref) < τ = `[VERIFIED_TAU]`

Noise-floor calibration (canonical vs canonical, 30 examples, 5 independent runs)
confirmed the 95th percentile sampling KL is well below τ. See `scripts/calibrate_kl.py`.

Citation: arXiv:2604.08570 (QuanBench+, preprint March 2026). τ and shot count verified against paper text. Venue (if any) verified from paper — do not add a venue claim without confirmation.

### Retroactive comparison

Both validators were run retroactively on stored `generated_code` from all 16 model
result sets (2,416 data points). Disagreements are analysed with:
- Cohen's κ (agreement beyond chance)
- McNemar's test (asymmetry between A-only and B-only cases)
- `defines_entry_point` split (isolates format-mismatch effect from genuine disagreement)

**Pass/fail/error semantics:** Each validator records three states: `True` (passed),
`False` (ran successfully but answer is wrong), `None` (could not evaluate — execution
error, timeout, or no circuit found). Agreement matrix denominators include only rows
where both methods produced a definitive result. Error rates are reported separately.

**Partial measurement:** If generated code measures fewer qubits than the canonical
solution, the KL divergence is computed over different key spaces. With additive
smoothing, this typically produces a high divergence and a `False` result. This is
defensible — they are not computing the same observable — but a high failure rate on
such rows reflects measurement convention mismatch rather than algorithmic error.

### v1.1 leaderboard columns

Two new columns alongside the existing `semantic_pct`:
- `unit_test_pct` — Approach A pass rate
- `kl_semantic_pct` — Approach B pass rate

v1.0 `semantic_pct` (execution pass) is retained for continuity.
```

- [ ] **Step 2: Commit**

```bash
git add METHODOLOGY.md
git commit -m "docs: METHODOLOGY.md v1.1 — unit test and KL divergence approaches"
```

---

## Self-review

**Spec coverage:**
- ✅ IBM dataset fetch with `test_code`, `canonical_solution`, `entry_point` (Task 2)
- ✅ SuiteExample carries new fields; wired to CLI (Task 3)
- ✅ KL validator with entry_point runner fix — reference calls `entry_point()`, not just defines it (Task 4)
- ✅ Noise floor calibration before full rescore (Task 5)
- ✅ Rescore with resume, `defines_entry_point` flag, both suite hashes (Task 6)
- ✅ Full 16-model rescore (Task 7)
- ✅ Cohen's κ, McNemar's test, `defines_entry_point` disaggregation, per-example Pearson (Task 8)
- ✅ METHODOLOGY.md v1.1 (Task 9)
- ✅ QuanBench+ citation verified before use (Task 1)

**Reviewer issues addressed (v2 review):**
- ✅ Canonical runner calls `entry_point()` — no longer returns `[]`
- ✅ Generated runner tries `entry_point()` first, falls back to locals scan
- ✅ Synthesised wrapper for Approach A when model used standalone format
- ✅ `defines_entry_point` flag added per row, used in analysis breakdown
- ✅ `seed_simulator` set for reproducibility
- ✅ Measurement check uses `inst.operation.name == "measure"`, not `num_clbits`
- ✅ Multi-register keys normalised (`"".join(k.split())`)
- ✅ `warnings.filterwarnings` narrowed to `module="qiskit"` only
- ✅ Dead `IBM_URL` parquet constant removed
- ✅ Both v1.0 and v1.1 suite hashes recorded in rescore header
- ✅ Resume-by-task-id in rescore script
- ✅ Cohen's κ and McNemar's test in analysis
- ✅ Per-example phi coefficient (n=2,416) not per-model (n=16)
- ✅ Calibration task inserted before full rescore
- ✅ τ verified empirically before use

**Reviewer issues addressed (v6 review):**
- ✅ A/B False asymmetry resolved: Task 1 Step 1 now verifies raise/return convention with a live test; Task 1.5 specifies that `validate_test_code` must raise on subprocess crashes, return on assertion failures — aligning unit_test_pass=None/False semantics with Approach B
- ✅ Canonical sanity pass (Task 5 Step 4): runs all 151 canonical solutions through `run_canonical_circuit` before the full rescore — catches Qiskit API drift cheaply
- ✅ Per-model errored counts added to analysis table (`err%` column)

**Reviewer issues addressed (v5 review):**
- ✅ stdout JSON parsing: reverse-iterate lines to find last valid JSON — survives print() in generated code
- ✅ Test added: `test_run_circuit_handles_print_in_generated_code` confirms print() doesn't break parsing
- ✅ `errors="replace"` on subprocess.run — UnicodeDecodeError from non-UTF-8 output fails gracefully
- ✅ `kl_pass` and `unit_test_pass` use three-state semantics: `True`=pass, `False`=divergent/failed, `None`=couldn't evaluate
- ✅ `unit_test_pass=None` (not False) on exception, with `unit_test_error` recorded
- ✅ Agreement matrix denominator: only rows where both methods ran; `n_errored` tracked separately
- ✅ Analysis output reports evaluable vs errored row counts
- ✅ METHODOLOGY.md template documents None semantics and partial-measurement behavior

**Reviewer issues addressed (v4 review):**
- ✅ `validate_test_code` failure branch: explicit Task 1.5 block with implementation instructions if API check fails
- ✅ `synthesise_wrapper` tab/space hazard: `expandtabs(4)` + `textwrap.indent` instead of string concatenation
- ✅ `from __future__` guard: call site skips wrapper synthesis when `"from __future__"` present in code
- ✅ Calibration pairwise correlation: comment added noting 300 pairs are not iid
- ✅ QuanBench+ venue claim dropped: cite as "preprint" unless paper text confirms venue
- ✅ Task 5 expected output fixed to match actual print string `"(random sample, seed=42)"`
- ✅ Timeout differentiation: `except subprocess.TimeoutExpired: raise RuntimeError("timeout")` — produces `kl_error="timeout"` vs other errors in rescore output

**Reviewer issues addressed (v3 review):**
- ✅ Calibration bug fixed: now calls `run_canonical_circuit()` twice with different seeds and computes `kl_divergence(d1, d2)` directly — no longer `abs(0 - 0) = 0`
- ✅ `run_canonical_circuit()` exposed as a public function in `kl_validator.py`
- ✅ Calibration uses `random.sample` (fixed seed) instead of first-N — avoids biasing toward 2-qubit circuits
- ✅ `synthesise_wrapper` inlines code into function body — no longer uses `sys._getframe(1)` which found test locals, not module globals
- ✅ `has_toplevel_return()` guard added — wrapper only applied when safe to indent
- ✅ `defines_entry_point_fn` uses `tree.body` not `ast.walk` — no longer matches nested definitions
- ✅ Pre-flight task (Task 1) includes API check for `validate_test_code` and `generated_code` markdown fence inspection
- ✅ Per-model McNemar uses exact binomial (`binomtest`) — chi-squared unreliable when `a+b` is single digits
- ✅ Overall McNemar uses continuity-corrected chi-squared (appropriate for n=2,416)
- ✅ Model label regex handles version dots: `4-6` → `4.6` via `re.sub(r"(\d)-(\d)", r"\1.\2", ...)`
- ✅ Binary Pearson correctly named "phi coefficient" in output and saved JSON

**Placeholder scan:** None found.

**Type consistency:** `KLResult.kl_divergence: float | None`, `unit_test_pass: bool | None`, `kl_pass: bool | None`, `defines_entry_point: bool` — consistent throughout Tasks 4–8.
