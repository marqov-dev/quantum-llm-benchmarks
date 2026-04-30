import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class SuiteExample:
    id: str
    prompt: str
    category: str


def compute_suite_hash(suite_path: Path) -> str:
    """SHA-256 of the suite JSONL file bytes. Computed at runtime — not committed as a separate file."""
    content = suite_path.read_bytes()
    return "sha256:" + hashlib.sha256(content).hexdigest()


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
            ))
    return examples


def write_header(output_path: Path, *, suite: str, suite_hash: str, model: str) -> None:
    """Write the metadata header as line 1 of the output JSONL.

    All downstream consumers (results.py, third-party readers) must skip records
    where _header is True — it is metadata, not a result row.
    """
    header = {
        "_header": True,
        "suite": suite,
        "suite_hash": suite_hash,
        "model": model,
        "started": datetime.now(timezone.utc).isoformat(),
    }
    with output_path.open("w") as f:
        f.write(json.dumps(header) + "\n")


def append_result(
    output_path: Path,
    *,
    example_id: str,
    generated_code: str,
    syntax_pass: bool,
    execution_pass: bool,
    semantic_pass: bool,
    error: str | None,
) -> None:
    """Append one result record to the output JSONL."""
    record = {
        "id": example_id,
        "generated_code": generated_code,
        "syntax_pass": syntax_pass,
        "execution_pass": execution_pass,
        "semantic_pass": semantic_pass,
        "error": error,
    }
    with output_path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def get_completed_ids(output_path: Path) -> set[str]:
    """Return the set of example IDs already recorded in output_path.

    Skips _header records. Returns empty set if file doesn't exist or is empty.
    """
    if not output_path.exists():
        return set()
    completed: set[str] = set()
    with output_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip corrupt line, keep accumulating good IDs
            if record.get("_header"):
                continue
            if "id" in record:
                completed.add(record["id"])
    return completed


def get_header_suite_hash(output_path: Path) -> str | None:
    """Return the suite_hash from the _header record of output_path, or None if absent."""
    if not output_path.exists():
        return None
    with output_path.open() as f:
        first_line = f.readline().strip()
    if not first_line:
        return None
    try:
        record = json.loads(first_line)
    except json.JSONDecodeError:
        return None
    if record.get("_header"):
        return record.get("suite_hash")
    return None
