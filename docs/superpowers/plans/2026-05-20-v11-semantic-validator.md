# v1.1 Semantic Validator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the benchmark from "does the code run?" to "does the code produce the correct quantum output?" using two parallel approaches — IBM unit test execution (Approach A) and KL divergence on measurement distributions (Approach B) — then compare them retroactively across all 16 existing model result sets.

**Architecture:** Enrich `humaneval.jsonl` with `test_code` and `canonical_solution` from IBM's public dataset (`Qiskit/qiskit_humaneval`). Add a KL divergence validator module. Add a `rescore` CLI command that runs both validators retroactively on stored `generated_code` from existing result JSONLs. Produce a comparison analysis revealing agreement/disagreement between the two approaches across 2,416 data points (16 models × 151 examples).

**Tech Stack:** Python 3.12, Qiskit 2.x, qiskit-aer 0.15+, scipy 1.13+ (already in pyproject.toml), urllib (stdlib), existing validator.py subprocess pattern.

---

## Background and design decisions

IBM's `Qiskit/qiskit_humaneval` dataset has three relevant fields per problem:
- `canonical_solution`: the reference function body (continuation of the prompt)
- `test`: unit test code that calls the function by `entry_point` name
- `entry_point`: the function name the test calls

**Format mismatch (important):** IBM's unit tests assume the model output *defines a named function* (e.g., `def create_bell_state()`). Our leaderboard prompts ask for standalone programs. This mismatch means IBM tests will pass for models that happen to define the right function, and fail for models that write working standalone code. Both outcomes are informative and worth measuring.

**KL divergence approach:** Run both the canonical solution and the generated code on AerSimulator (1024 shots, τ=0.05). This works with standalone code since we only care about the measurement distribution, not function structure.

**Retroactive rescore:** We already have `generated_code` stored in 16 result JSONLs. The `rescore` command reads these and runs both new validators without re-querying any model.

---

## File map

**New files:**
- `scripts/enrich_suite.py` — fetches IBM dataset, adds `test_code`/`canonical_solution` to humaneval.jsonl
- `quantum_eval/kl_validator.py` — KL divergence validator (Approach B)
- `scripts/rescore_v11.py` — retroactive rescoring script (reads existing JSONLs, runs both validators)
- `scripts/analyze_v11.py` — comparison analysis: agreement matrix, correlation, disagreement breakdown
- `tests/test_kl_validator.py` — tests for KL validator

**Modified files:**
- `quantum_eval/_data/benchmarks/humaneval/humaneval.jsonl` — add `test_code`, `canonical_solution`, `entry_point` fields
- `quantum_eval/harness.py` — add `test_code` and `canonical_solution` to `SuiteExample`, update `load_suite`
- `quantum_eval/cli.py` — pass `test_code` from `SuiteExample` to `validate_example`
- `METHODOLOGY.md` — v1.1 section

---

## Task 1: Fetch IBM dataset and enrich humaneval.jsonl

**Files:**
- Create: `scripts/enrich_suite.py`
- Modify: `quantum_eval/_data/benchmarks/humaneval/humaneval.jsonl`

- [ ] **Step 1: Write enrich_suite.py**

```python
#!/usr/bin/env python3
"""Enrich humaneval.jsonl with IBM's test_code and canonical_solution.

Fetches from HuggingFace Hub API (no datasets library required).
Matches on task_id (e.g. "qiskitHumanEval_0").
Adds fields: test_code, canonical_solution, entry_point.
Writes in-place (backs up original first).
"""
import json
import shutil
import urllib.request
from pathlib import Path

SUITE_PATH = Path("quantum_eval/_data/benchmarks/humaneval/humaneval.jsonl")
# HuggingFace Hub raw file URL for the IBM dataset JSON
IBM_URL = (
    "https://huggingface.co/datasets/Qiskit/qiskit_humaneval"
    "/resolve/main/data/test-00000-of-00001.parquet"
)
# Fallback: use the GitHub repo JSON directly
IBM_GITHUB_URL = (
    "https://raw.githubusercontent.com/qiskit-community/qiskit-human-eval"
    "/main/dataset/dataset_qiskit_test_human_eval.json"
)


def fetch_ibm_data() -> dict[str, dict]:
    """Fetch IBM dataset from GitHub. Returns {task_id: {canonical_solution, test, entry_point}}."""
    print(f"Fetching IBM dataset from GitHub...")
    with urllib.request.urlopen(IBM_GITHUB_URL, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
    data = json.loads(raw)
    # The JSON is a list of problem dicts
    if isinstance(data, list):
        problems = data
    elif isinstance(data, dict) and "problems" in data:
        problems = data["problems"]
    else:
        raise ValueError(f"Unexpected IBM dataset format: {list(data.keys()) if isinstance(data, dict) else type(data)}")
    
    result = {}
    for p in problems:
        task_id = p.get("task_id", "")
        result[task_id] = {
            "canonical_solution": p.get("canonical_solution", ""),
            "test_code": p.get("test", ""),
            "entry_point": p.get("entry_point", ""),
        }
    print(f"  Loaded {len(result)} problems from IBM dataset.")
    return result


def enrich_suite(suite_path: Path, ibm_data: dict[str, dict]) -> None:
    """Add IBM fields to each example in the JSONL. Backs up original."""
    backup = suite_path.with_suffix(".jsonl.bak")
    shutil.copy2(suite_path, backup)
    print(f"  Backed up original to {backup}")

    lines = suite_path.read_text(encoding="utf-8").splitlines()
    enriched = []
    matched = 0
    for line in lines:
        if not line.strip():
            continue
        ex = json.loads(line)
        task_id = ex["id"]  # e.g. "qiskitHumanEval_0"
        if task_id in ibm_data:
            ex.update(ibm_data[task_id])
            matched += 1
        else:
            print(f"  WARNING: {task_id} not found in IBM dataset")
        enriched.append(json.dumps(ex))

    suite_path.write_text("\n".join(enriched) + "\n", encoding="utf-8")
    print(f"  Enriched {matched}/{len(lines)} examples. Written to {suite_path}")


if __name__ == "__main__":
    ibm_data = fetch_ibm_data()
    enrich_suite(SUITE_PATH, ibm_data)
    print("\nDone. Run the benchmark to verify the new suite hash.")
```

- [ ] **Step 2: Run the enrichment script**

```bash
cd /Users/david/Github/quantum-llm-benchmarks
.venv/bin/python scripts/enrich_suite.py
```

Expected output:
```
Fetching IBM dataset from GitHub...
  Loaded 151 problems from IBM dataset.
  Backed up original to quantum_eval/_data/benchmarks/humaneval/humaneval.jsonl.bak
  Enriched 151/151 examples. Written to quantum_eval/_data/benchmarks/humaneval/humaneval.jsonl
```

- [ ] **Step 3: Inspect a sample to verify the new fields**

```bash
.venv/bin/python -c "
import json
with open('quantum_eval/_data/benchmarks/humaneval/humaneval.jsonl') as f:
    ex = json.loads(f.readline())
print('Fields:', list(ex.keys()))
print('entry_point:', ex.get('entry_point'))
print('canonical_solution[:100]:', ex.get('canonical_solution', '')[:100])
print('test_code[:100]:', ex.get('test_code', '')[:100])
"
```

Expected: `Fields:` includes `canonical_solution`, `test_code`, `entry_point`. Both fields are non-empty strings.

- [ ] **Step 4: Verify all 151 examples have the new fields**

```bash
.venv/bin/python -c "
import json
missing = []
with open('quantum_eval/_data/benchmarks/humaneval/humaneval.jsonl') as f:
    for line in f:
        ex = json.loads(line)
        if not ex.get('test_code') or not ex.get('canonical_solution'):
            missing.append(ex['id'])
print(f'Missing: {len(missing)} examples')
if missing:
    print(missing[:5])
"
```

Expected: `Missing: 0 examples`

- [ ] **Step 5: Commit**

```bash
git add quantum_eval/_data/benchmarks/humaneval/humaneval.jsonl scripts/enrich_suite.py
git commit -m "feat: enrich humaneval suite with IBM test_code and canonical_solution"
```

---

## Task 2: Update SuiteExample to carry test_code and canonical_solution

**Files:**
- Modify: `quantum_eval/harness.py` (lines 8-37)
- Modify: `quantum_eval/cli.py` (around line 125)
- Modify: `tests/test_harness.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_harness.py`:

```python
def test_load_suite_carries_test_code(tmp_path):
    f = tmp_path / "suite.jsonl"
    f.write_text(
        '{"id": "qiskitHumanEval_0", "instruction": "create a Bell state", '
        '"test_code": "assert True", "canonical_solution": "qc = QuantumCircuit(2)"}\n'
    )
    examples = load_suite(f)
    assert examples[0].test_code == "assert True"
    assert examples[0].canonical_solution == "qc = QuantumCircuit(2)"


def test_load_suite_missing_test_code_is_none(tmp_path):
    f = tmp_path / "suite.jsonl"
    f.write_text('{"id": "qiskitHumanEval_0", "instruction": "create a Bell state"}\n')
    examples = load_suite(f)
    assert examples[0].test_code is None
    assert examples[0].canonical_solution is None
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/test_harness.py::test_load_suite_carries_test_code -v
```

Expected: `FAILED` with `AttributeError: 'SuiteExample' object has no attribute 'test_code'`

- [ ] **Step 3: Update SuiteExample and load_suite in harness.py**

Replace the `SuiteExample` dataclass and `load_suite` function:

```python
@dataclass
class SuiteExample:
    id: str
    prompt: str
    category: str
    test_code: str | None = None
    canonical_solution: str | None = None


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
            ))
    return examples
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/test_harness.py -v
```

Expected: all pass.

- [ ] **Step 5: Wire test_code into validate_example call in cli.py**

In `cli.py`, in the `cmd_run` function, find the `validate_example` call (around line 125):

```python
        result, extracted_code = validate_example({"response": generated, "category": ex.category})
```

Replace with:

```python
        result, extracted_code = validate_example({
            "response": generated,
            "category": ex.category,
            "test_code": ex.test_code,
        })
```

- [ ] **Step 6: Run full test suite to confirm nothing broke**

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add quantum_eval/harness.py quantum_eval/cli.py tests/test_harness.py
git commit -m "feat: wire test_code through SuiteExample to validate_example"
```

---

## Task 3: KL divergence validator

**Files:**
- Create: `quantum_eval/kl_validator.py`
- Create: `tests/test_kl_validator.py`

The KL validator runs both the canonical solution and the generated code on AerSimulator (1024 shots), extracts measurement distributions, and computes KL divergence with additive smoothing. It handles circuits with or without measurements by auto-inserting `measure_all()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_kl_validator.py
import pytest
from quantum_eval.kl_validator import (
    run_circuit_and_get_counts,
    kl_divergence,
    validate_kl_divergence,
    KLResult,
)


def test_kl_divergence_identical_distributions():
    """Identical distributions should have KL divergence of 0."""
    p = {"00": 0.5, "11": 0.5}
    q = {"00": 0.5, "11": 0.5}
    assert kl_divergence(p, q) < 1e-9


def test_kl_divergence_very_different_distributions():
    """Very different distributions should have KL divergence >> 0.05."""
    p = {"00": 1.0}
    q = {"11": 1.0}
    assert kl_divergence(p, q) > 1.0


def test_run_circuit_bell_state():
    """Bell state should produce ~50/50 '00'/'11' distribution."""
    code = """
from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
"""
    counts = run_circuit_and_get_counts(code, shots=2048)
    total = sum(counts.values())
    assert total == 2048
    assert "00" in counts or "11" in counts


def test_validate_kl_divergence_matching_bell_states():
    """Two Bell state implementations should KL-pass."""
    canonical = """
from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
"""
    generated = """
from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
"""
    result = validate_kl_divergence(generated, canonical)
    assert result.passed is True
    assert result.kl_divergence < 0.05


def test_validate_kl_divergence_wrong_circuit():
    """Completely wrong circuit should KL-fail."""
    canonical = """
from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
"""
    generated = """
from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
# No gates — |00> state
"""
    result = validate_kl_divergence(generated, canonical)
    assert result.passed is False
    assert result.kl_divergence > 0.05


def test_validate_kl_divergence_error_on_bad_code():
    """Non-executable code should return an error result, not raise."""
    result = validate_kl_divergence("this is not python!!!!", "qc = QuantumCircuit(2)")
    assert result.passed is False
    assert result.error is not None
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/test_kl_validator.py -v
```

Expected: `FAILED` — `ModuleNotFoundError: No module named 'quantum_eval.kl_validator'`

- [ ] **Step 3: Implement kl_validator.py**

```python
# quantum_eval/kl_validator.py
"""
Approach B: KL divergence on measurement distributions.

Runs both the canonical reference circuit and model-generated circuit on
AerSimulator (1024 shots). Computes KL divergence between output distributions.
Passes if KL(P_gen || P_ref) < tau=0.05.

Threshold and shot count per QuanBench+ (arXiv:2604.08570, ICLR 2026).
"""
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


TAU = 0.05          # KL divergence threshold (QuanBench+ calibrated)
DEFAULT_SHOTS = 1024
EPSILON = 1e-10     # Additive smoothing to avoid log(0)

# Template: runs arbitrary Qiskit code, finds the last QuantumCircuit in locals,
# measures all qubits, runs on AerSimulator, prints counts as JSON.
_RUNNER_TEMPLATE = '''
import json, sys, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

{code}

# Find QuantumCircuit objects defined in the code
_circuits = [v for v in list(locals().values()) if isinstance(v, QuantumCircuit)]
if not _circuits:
    print(json.dumps({{"error": "no QuantumCircuit found in generated code"}}))
    sys.exit(0)

_qc = _circuits[-1]  # use the last one defined

# Auto-insert measure_all if no measurements present
if _qc.num_clbits == 0:
    _qc = _qc.copy()
    _qc.measure_all()

_sim = AerSimulator()
_job = _sim.run(_qc, shots={shots})
_counts = _job.result().get_counts()

print(json.dumps({{"counts": _counts, "total": sum(_counts.values())}}))
'''


@dataclass
class KLResult:
    passed: bool
    kl_divergence: Optional[float]
    error: Optional[str] = None


def run_circuit_and_get_counts(code: str, shots: int = DEFAULT_SHOTS) -> dict[str, float]:
    """Execute Qiskit code in a subprocess and return normalised measurement counts.

    Returns a dict of {bitstring: probability}. Raises RuntimeError on failure.
    """
    runner = _RUNNER_TEMPLATE.format(code=code, shots=shots)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(runner)
        tmp = Path(f.name)
    try:
        result = subprocess.run(
            [sys.executable, str(tmp)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip()[-500:] or "non-zero exit")
        import json as _json
        data = _json.loads(result.stdout.strip())
        if "error" in data:
            raise RuntimeError(data["error"])
        counts = data["counts"]
        total = data["total"]
        return {k: v / total for k, v in counts.items()}
    finally:
        tmp.unlink(missing_ok=True)


def kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """KL divergence KL(P || Q) with additive smoothing.

    p: generated distribution, q: reference distribution.
    All keys from both dicts are included (smoothed if missing).
    """
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
    shots: int = DEFAULT_SHOTS,
    tau: float = TAU,
) -> KLResult:
    """Run both circuits and compare measurement distributions via KL divergence.

    Returns KLResult(passed, kl_divergence, error).
    Never raises — errors are captured in KLResult.error.
    """
    try:
        ref_dist = run_circuit_and_get_counts(canonical_code, shots=shots)
    except Exception as e:
        return KLResult(passed=False, kl_divergence=None, error=f"Reference circuit error: {e}")

    try:
        gen_dist = run_circuit_and_get_counts(generated_code, shots=shots)
    except Exception as e:
        return KLResult(passed=False, kl_divergence=None, error=f"Generated circuit error: {e}")

    kl = kl_divergence(gen_dist, ref_dist)
    return KLResult(passed=kl < tau, kl_divergence=round(kl, 6))
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/test_kl_validator.py -v
```

Expected: all 6 tests pass. The Bell state tests may take 5-10s due to AerSimulator subprocess overhead.

- [ ] **Step 5: Run full test suite**

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add quantum_eval/kl_validator.py tests/test_kl_validator.py
git commit -m "feat: KL divergence validator (Approach B, QuanBench+ standard)"
```

---

## Task 4: Retroactive rescore script

**Files:**
- Create: `scripts/rescore_v11.py`

This script reads existing result JSONLs (which contain `generated_code`), loads the enriched `humaneval.jsonl` for `test_code` and `canonical_solution`, runs both new validators on each example, and writes enriched output JSONLs with additional fields: `unit_test_pass`, `kl_pass`, `kl_divergence`, `kl_error`.

- [ ] **Step 1: Implement rescore_v11.py**

```python
#!/usr/bin/env python3
"""
Retroactive v1.1 rescoring: run unit test (Approach A) and KL divergence (Approach B)
validators on stored generated_code from existing result JSONLs.

Usage:
    .venv/bin/python scripts/rescore_v11.py results/claude-sonnet-4-6_humaneval.jsonl
    .venv/bin/python scripts/rescore_v11.py results/*.jsonl        # all at once

Output: results/v11/<original_name> — new JSONL with extra fields per row:
    unit_test_pass: bool | null
    kl_pass: bool | null
    kl_divergence: float | null
    kl_error: str | null
"""
import json
import sys
from pathlib import Path

SUITE_PATH = Path("quantum_eval/_data/benchmarks/humaneval/humaneval.jsonl")
OUTPUT_DIR = Path("results/v11")


def load_suite_index(suite_path: Path) -> dict[str, dict]:
    """Return {task_id: {test_code, canonical_solution}} for all examples."""
    index = {}
    with suite_path.open() as f:
        for line in f:
            ex = json.loads(line)
            index[ex["id"]] = {
                "test_code": ex.get("test_code"),
                "canonical_solution": ex.get("canonical_solution"),
            }
    return index


def rescore_file(jsonl_path: Path, suite_index: dict[str, dict]) -> Path:
    """Rescore one result JSONL. Returns path of output file."""
    from quantum_eval.validator import validate_test_code, ValidationLevel
    from quantum_eval.kl_validator import validate_kl_divergence

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / jsonl_path.name

    rows = jsonl_path.read_text(encoding="utf-8").splitlines()
    n_total = len([r for r in rows if r.strip() and not json.loads(r).get("_header")])
    print(f"\n{jsonl_path.name}: {n_total} examples to rescore")

    with out_path.open("w", encoding="utf-8") as out_f:
        for i, row in enumerate(rows):
            if not row.strip():
                continue
            record = json.loads(row)

            # Pass header through unchanged
            if record.get("_header"):
                out_f.write(row + "\n")
                continue

            task_id = record["id"]
            generated_code = record.get("generated_code", "")
            suite_entry = suite_index.get(task_id, {})
            test_code = suite_entry.get("test_code")
            canonical_solution = suite_entry.get("canonical_solution")

            # Approach A: unit test execution
            unit_test_pass = None
            if test_code and generated_code:
                try:
                    ut_result = validate_test_code(generated_code, test_code, timeout=60)
                    unit_test_pass = ut_result.level_passed.value >= ValidationLevel.SEMANTIC.value
                except Exception as e:
                    unit_test_pass = False

            # Approach B: KL divergence
            kl_pass = None
            kl_divergence_val = None
            kl_error = None
            if canonical_solution and generated_code:
                kl_result = validate_kl_divergence(generated_code, canonical_solution)
                kl_pass = kl_result.passed
                kl_divergence_val = kl_result.kl_divergence
                kl_error = kl_result.error

            record["unit_test_pass"] = unit_test_pass
            record["kl_pass"] = kl_pass
            record["kl_divergence"] = kl_divergence_val
            record["kl_error"] = kl_error
            out_f.write(json.dumps(record) + "\n")

            # Progress
            done = i + 1
            print(f"  [{done}/{n_total}] {task_id}  unit_test={unit_test_pass}  kl={kl_pass}  kl_div={kl_divergence_val}", flush=True)

    print(f"  Written: {out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: rescore_v11.py <result.jsonl> [result2.jsonl ...]")
        sys.exit(1)

    suite_index = load_suite_index(SUITE_PATH)
    print(f"Loaded suite index: {len(suite_index)} examples")

    for arg in sys.argv[1:]:
        for path in sorted(Path(".").glob(arg)) if "*" in arg else [Path(arg)]:
            if not path.exists():
                print(f"SKIP: {path} not found")
                continue
            rescore_file(path, suite_index)

    print("\nAll done.")
```

- [ ] **Step 2: Smoke-test with a single small JSONL**

First create a tiny test JSONL to verify the script works before running on all 16 model files:

```bash
.venv/bin/python -c "
import json
from pathlib import Path

# Take first 3 rows from claude-sonnet-4-6 results
rows = Path('results/claude-sonnet-4-6_humaneval.jsonl').read_text().splitlines()[:4]
Path('/tmp/test_rescore.jsonl').write_text('\n'.join(rows) + '\n')
print('Created /tmp/test_rescore.jsonl')
"

.venv/bin/python scripts/rescore_v11.py /tmp/test_rescore.jsonl
```

Expected: Script runs, prints 3 examples with `unit_test=` and `kl=` values, writes `/tmp/v11/test_rescore.jsonl` (it may write to a different path — adjust as needed). Spot-check that the output has `unit_test_pass`, `kl_pass`, `kl_divergence` fields.

- [ ] **Step 3: Verify output structure**

```bash
.venv/bin/python -c "
import json
from pathlib import Path
rows = [json.loads(l) for l in Path('results/v11/test_rescore.jsonl').read_text().splitlines() if l.strip()]
result_rows = [r for r in rows if not r.get('_header')]
for r in result_rows:
    print(r['id'], r.get('unit_test_pass'), r.get('kl_pass'), r.get('kl_divergence'))
"
```

Expected: 3 rows, each showing `unit_test_pass`, `kl_pass`, and a float or null for `kl_divergence`.

- [ ] **Step 4: Commit**

```bash
git add scripts/rescore_v11.py
git commit -m "feat: retroactive v1.1 rescore script (unit test + KL divergence)"
```

---

## Task 5: Run the rescore on all 16 model files

This will take several hours (each example runs two subprocess validators). Run in background.

- [ ] **Step 1: Start rescore for all 16 model files**

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

- [ ] **Step 2: Monitor progress**

```bash
tail -f /tmp/rescore_v11.log
```

Check periodically. Each model takes ~30-60 min (151 examples × ~15s per example for two validators).

- [ ] **Step 3: Verify all 16 output files exist and have 151 result rows**

```bash
for f in results/v11/*.jsonl; do
  count=$(tail -n +2 "$f" | wc -l | tr -d ' ')
  echo "$f: $count"
done
```

Expected: all 16 files, each showing `151`.

---

## Task 6: Analysis script

**Files:**
- Create: `scripts/analyze_v11.py`

- [ ] **Step 1: Implement analyze_v11.py**

```python
#!/usr/bin/env python3
"""
Compare Approach A (unit tests) vs Approach B (KL divergence) across all rescored model files.

Produces:
  1. Per-model table: v1.0 semantic%, unit_test%, kl_semantic%
  2. Overall 2x2 agreement matrix (A vs B)
  3. Per-model 2x2 agreement matrix
  4. Disagreement case breakdown: where A passes but B fails, or vice versa

Usage:
    .venv/bin/python scripts/analyze_v11.py
"""
import json
from pathlib import Path


V11_DIR = Path("results/v11")
LABEL_MAP = {
    "claude-opus-4-7": "Claude Opus 4.7",
    "claude-opus-4-6": "Claude Opus 4.6",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "deepseek-v4-pro": "DeepSeek V4 Pro",
    "deepseek-v4-flash": "DeepSeek V4 Flash",
    "qwen3-coder-480b-a35b-instruct": "Qwen3 Coder 480B",
    "qwen3-coder-plus": "Qwen3 Coder Plus",
    "codestral-latest": "Codestral",
    "mistral-large-latest": "Mistral Large",
    "moonshot-v1-8k": "Kimi K2.6",
    "openai_gpt-oss-20b": "GPT OSS 20B",
    "openai_gpt-oss-120b": "GPT OSS 120B",
    "meta-llama_llama-4-scout-17b-16e-instruct": "Llama 4 Scout",
    "gemma4_latest": "Gemma 4 12B",
}


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


def model_key(path: Path) -> str:
    name = path.stem.replace("_humaneval", "")
    return name


def agreement_matrix(rows: list[dict]) -> dict:
    """Compute 2x2 agreement matrix for rows with unit_test_pass and kl_pass."""
    both_valid = [r for r in rows if r.get("unit_test_pass") is not None and r.get("kl_pass") is not None]
    aa = sum(1 for r in both_valid if r["unit_test_pass"] and r["kl_pass"])
    ab = sum(1 for r in both_valid if r["unit_test_pass"] and not r["kl_pass"])
    ba = sum(1 for r in both_valid if not r["unit_test_pass"] and r["kl_pass"])
    bb = sum(1 for r in both_valid if not r["unit_test_pass"] and not r["kl_pass"])
    return {"both_pass": aa, "a_only": ab, "b_only": ba, "both_fail": bb, "n": len(both_valid)}


def pct(k: int, n: int) -> str:
    return f"{round(k/n*100)}%" if n else "N/A"


def main():
    files = sorted(V11_DIR.glob("*_humaneval.jsonl"))
    if not files:
        print(f"No files found in {V11_DIR}. Run rescore_v11.py first.")
        return

    all_rows = []
    model_stats = []

    print("\n" + "=" * 90)
    print(f"{'Model':<30} {'v1.0 sem':>9} {'unit_test':>10} {'kl_sem':>8} {'agree':>7} {'A≠B':>6}")
    print("=" * 90)

    for fpath in files:
        key = model_key(fpath)
        label = LABEL_MAP.get(key, key)
        rows = load_results(fpath)
        n = len(rows)

        v10_sem = sum(1 for r in rows if r.get("semantic_pass"))
        unit_test = sum(1 for r in rows if r.get("unit_test_pass"))
        kl_sem = sum(1 for r in rows if r.get("kl_pass"))

        mat = agreement_matrix(rows)
        agree = mat["both_pass"] + mat["both_fail"]
        disagree = mat["a_only"] + mat["b_only"]

        print(
            f"{label:<30} {pct(v10_sem, n):>9} {pct(unit_test, n):>10} "
            f"{pct(kl_sem, n):>8} {pct(agree, mat['n']):>7} {disagree:>6}"
        )

        all_rows.extend(rows)
        model_stats.append({
            "label": label, "n": n,
            "v10_semantic_pct": round(v10_sem / n * 100),
            "unit_test_pct": round(unit_test / n * 100),
            "kl_semantic_pct": round(kl_sem / n * 100),
            "matrix": mat,
        })

    # Overall matrix
    overall = agreement_matrix(all_rows)
    print("\n" + "=" * 90)
    print(f"Overall ({len(all_rows)} examples):")
    print(f"  Both pass (A∩B):      {overall['both_pass']:>5}  {pct(overall['both_pass'], overall['n'])}")
    print(f"  A only (unit test):   {overall['a_only']:>5}  {pct(overall['a_only'], overall['n'])}")
    print(f"  B only (KL div):      {overall['b_only']:>5}  {pct(overall['b_only'], overall['n'])}")
    print(f"  Both fail:            {overall['both_fail']:>5}  {pct(overall['both_fail'], overall['n'])}")
    print(f"  Agreement rate:       {pct(overall['both_pass'] + overall['both_fail'], overall['n'])}")

    # Correlation
    v10_scores = [s["v10_semantic_pct"] for s in model_stats]
    ut_scores = [s["unit_test_pct"] for s in model_stats]
    kl_scores = [s["kl_semantic_pct"] for s in model_stats]

    try:
        from scipy.stats import pearsonr
        r_v10_ut, _ = pearsonr(v10_scores, ut_scores)
        r_v10_kl, _ = pearsonr(v10_scores, kl_scores)
        r_ut_kl, _ = pearsonr(ut_scores, kl_scores)
        print(f"\nPearson correlations (per-model scores):")
        print(f"  v1.0 semantic vs unit_test:  r = {r_v10_ut:.3f}")
        print(f"  v1.0 semantic vs KL:         r = {r_v10_kl:.3f}")
        print(f"  unit_test vs KL:             r = {r_ut_kl:.3f}")
    except Exception as e:
        print(f"\n(Correlation skipped: {e})")

    # Save JSON summary
    out_path = Path("results/v11_analysis.json")
    out_path.write_text(json.dumps({
        "model_stats": model_stats,
        "overall_matrix": overall,
    }, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the analysis (after Task 5 completes)**

```bash
.venv/bin/python scripts/analyze_v11.py
```

Expected: Table printed with all 16 models, overall agreement matrix, Pearson correlations. JSON saved to `results/v11_analysis.json`.

- [ ] **Step 3: Commit**

```bash
git add scripts/analyze_v11.py
git commit -m "feat: v1.1 comparison analysis script (agreement matrix, correlation)"
```

---

## Task 7: METHODOLOGY.md v1.1 and documentation

**Files:**
- Modify: `METHODOLOGY.md`

- [ ] **Step 1: Add v1.1 section to METHODOLOGY.md**

Append the following section after the existing content:

```markdown
---

## v1.1 Semantic Validation (in progress)

**Status:** Implementation in progress. Results from v1.1 are not directly comparable to v1.0.

### Approach A: Unit test execution

Each Qiskit HumanEval problem includes hand-authored unit tests from IBM's public dataset (`qiskit-community/qiskit-human-eval`, HuggingFace: `Qiskit/qiskit_humaneval`). The generated code is executed followed by the unit test assertions in an isolated subprocess with a 60-second timeout.

**Important caveat:** IBM's tests assume the generated code *defines a named function* matching the `entry_point` field. Our leaderboard prompts elicit standalone programs. This format mismatch means Approach A may undercount correct programs that implement the logic as standalone code rather than a named function.

### Approach B: KL divergence on measurement distributions

Both the model-generated code and IBM's `canonical_solution` are executed on `AerSimulator` with 1024 shots. The resulting measurement distributions are compared via KL divergence with additive smoothing (ε=1e-10).

**Pass criterion:** KL(P_gen ∥ P_ref) < τ = 0.05

Threshold and shot count per QuanBench+ (arXiv:2604.08570, ICLR 2026 Workshop).

**Limitation:** Approach B compares circuit output distributions, not algorithm correctness. A circuit that happens to produce the same distribution via a different algorithm will pass.

### Retroactive comparison

Both validators were run retroactively on stored `generated_code` from all v1.0 model runs (16 models × 151 examples = 2,416 data points). The comparison reveals:

| Case | Interpretation |
|------|---------------|
| Both pass | Definitively correct |
| A only (unit test) | Logically correct, but KL threshold may be too tight or different decomposition |
| B only (KL) | Distribution matches but fails unit test — likely due to format mismatch (standalone vs function) |
| Both fail | Definitively incorrect |

### v1.1 leaderboard columns

When v1.1 ships, two new columns will be added alongside the existing `semantic_pct`:
- `unit_test_pct`: Approach A pass rate
- `kl_semantic_pct`: Approach B pass rate

v1.0 `semantic_pct` (= execution pass) remains on the leaderboard for continuity.
```

- [ ] **Step 2: Commit METHODOLOGY.md**

```bash
git add METHODOLOGY.md
git commit -m "docs: METHODOLOGY.md v1.1 section (unit test + KL divergence approaches)"
```

- [ ] **Step 3: Update GitHub issue #1 with implementation status**

Mark the implementation tasks as in-progress or done in the issue body, and link to the plan file.

---

## Self-review

**Spec coverage check:**
- ✅ IBM dataset enrichment (Task 1)
- ✅ Unit test execution via `validate_test_code` — already in validator.py, wired via Task 2
- ✅ KL divergence validator (Task 3)
- ✅ Retroactive rescore of all 16 models (Task 4 + 5)
- ✅ Comparison analysis: agreement matrix, correlation, disagreement breakdown (Task 6)
- ✅ METHODOLOGY.md v1.1 (Task 7)
- ✅ Format mismatch documented (METHODOLOGY.md caveat)

**Placeholder scan:** None found.

**Type consistency:** `KLResult.passed` (bool), `KLResult.kl_divergence` (float | None), `unit_test_pass` (bool | None), `kl_pass` (bool | None) — consistent throughout Tasks 3–6.
