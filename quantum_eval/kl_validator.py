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
- DeprecationWarning suppressed only for qiskit modules.
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
    json.loads(full_stdout) would raise JSONDecodeError on those rows.
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
