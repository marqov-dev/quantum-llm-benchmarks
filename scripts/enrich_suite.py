#!/usr/bin/env python3
"""Enrich humaneval.jsonl with IBM's test_code, canonical_solution, and entry_point.

Fetches from qiskit-community/qiskit-human-eval GitHub repo (no extra deps).
Matches on task_id. Writes in-place after backing up original.

Note: IBM dataset uses slash separator (qiskitHumanEval/0) while our suite
uses underscore (qiskitHumanEval_0). The fetch step normalises to underscore.
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


def _normalise_task_id(task_id: str) -> str:
    """Normalise IBM task_id to match our id format.

    IBM:  qiskitHumanEval/0  →  qiskitHumanEval_0
    Ours: qiskitHumanEval_0  (unchanged)
    """
    return task_id.replace("/", "_")


def fetch_ibm_data() -> dict[str, dict]:
    """Return {task_id: {canonical_solution, test_code, entry_point}}.

    Keys are normalised to underscore format to match our suite ids.
    """
    print(f"Fetching IBM dataset from GitHub...")
    with urllib.request.urlopen(IBM_GITHUB_URL, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
    data = json.loads(raw)
    problems = data if isinstance(data, list) else data.get("problems", [])
    result = {}
    for p in problems:
        task_id = _normalise_task_id(p.get("task_id", ""))
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
