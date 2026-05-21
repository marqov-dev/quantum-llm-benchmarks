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
                "extracted_code": ex.get("extracted_code"),  # full function def for KL runner
                "entry_point": ex.get("entry_point", ""),
            }
    return index


def compute_suite_hash(suite_path: Path) -> str:
    import hashlib
    return "sha256:" + hashlib.sha256(suite_path.read_bytes()).hexdigest()


def defines_entry_point_fn(generated_code: str, entry_point: str) -> bool:
    """Return True if generated_code defines entry_point as a TOP-LEVEL function.

    Uses tree.body (not ast.walk) to avoid matching nested definitions.
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
    """Return True if code has a top-level return statement."""
    try:
        tree = ast.parse(code)
        return any(isinstance(node, ast.Return) for node in tree.body)
    except SyntaxError:
        return False


def synthesise_wrapper(generated_code: str, entry_point: str) -> str:
    """Wrap standalone generated_code in a function named entry_point.

    Uses expandtabs(4) + textwrap.indent to avoid TabError.
    Caller must check: no top-level return, no 'from __future__' imports.
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
    from quantum_eval.validator import validate_test_code, ValidationLevel, extract_code
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
            raw_generated_code = record.get("generated_code", "")

            # Strip markdown fences: ALL stored generated_code fields contain fences
            generated_code = extract_code(raw_generated_code) if raw_generated_code else ""

            suite_entry = suite_index.get(task_id, {})
            test_code = suite_entry.get("test_code")
            entry_point = suite_entry.get("entry_point", "")
            extracted_code = suite_entry.get("extracted_code")  # full function def for KL

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
                    code_for_test = synthesise_wrapper(generated_code, entry_point)
                try:
                    ut_result = validate_test_code(code_for_test, test_code, timeout=60)
                    unit_test_pass = ut_result.level_passed.value >= ValidationLevel.SEMANTIC.value
                except Exception:
                    unit_test_pass = None  # None = couldn't evaluate
            record["unit_test_pass"] = unit_test_pass

            # --- Approach B: KL divergence ---
            kl_pass = None
            kl_divergence_val = None
            kl_error = None
            if extracted_code and generated_code:
                kl_result = validate_kl_divergence(
                    generated_code, extracted_code,
                    entry_point=entry_point,
                )
                kl_pass = kl_result.passed
                kl_divergence_val = kl_result.kl_divergence
                kl_error = kl_result.error
            elif not extracted_code:
                kl_error = "no extracted_code in suite"
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
