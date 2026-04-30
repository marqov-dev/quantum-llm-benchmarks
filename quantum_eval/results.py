import json
from datetime import datetime, timezone
from pathlib import Path

from scipy.stats import binomtest


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[int, int]:
    """Wilson 95% CI for a proportion, returned as integer percentages.

    Uses scipy.stats.binomtest(...).proportion_ci(method='wilson').
    """
    result = binomtest(k, n).proportion_ci(confidence_level=confidence, method="wilson")
    return round(result.low * 100), round(result.high * 100)


def aggregate_jsonl(jsonl_path: Path) -> dict:
    """Aggregate a results JSONL into pass counts and Wilson CIs.

    Skips records where _header is True.
    Raises ValueError if duplicate IDs are found (indicates a botched resume).
    Raises ValueError if no result records are found.
    """
    records = []
    seen_ids: set[str] = set()
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("_header"):
                continue
            example_id = record.get("id")
            if example_id in seen_ids:
                raise ValueError(
                    f"duplicate ID '{example_id}' in {jsonl_path}. "
                    "This may indicate a botched resume. Inspect the JSONL before publishing."
                )
            seen_ids.add(example_id)
            records.append(record)

    n = len(records)
    if n == 0:
        raise ValueError(f"No result records found in {jsonl_path}")

    def counts(field: str) -> int:
        return sum(1 for r in records if r.get(field, False))

    syntax_k = counts("syntax_pass")
    execution_k = counts("execution_pass")
    semantic_k = counts("semantic_pass")

    return {
        "n_examples": n,
        "syntax_pass": syntax_k,
        "syntax_pct": round(syntax_k / n * 100),
        "syntax_ci_low": wilson_ci(syntax_k, n)[0],
        "syntax_ci_high": wilson_ci(syntax_k, n)[1],
        "execution_pass": execution_k,
        "execution_pct": round(execution_k / n * 100),
        "execution_ci_low": wilson_ci(execution_k, n)[0],
        "execution_ci_high": wilson_ci(execution_k, n)[1],
        "semantic_pass": semantic_k,
        "semantic_pct": round(semantic_k / n * 100),
        "semantic_ci_low": wilson_ci(semantic_k, n)[0],
        "semantic_ci_high": wilson_ci(semantic_k, n)[1],
    }


def build_release_json(
    *,
    jsonl_path: Path,
    version: str,
    suite: str,
    suite_hash: str,
    model_id: str,
    model_label: str,
    provider: str,
    expected_n: int,
) -> dict:
    """Build the versioned release JSON from a single-model JSONL result file.

    Sets complete=True only if n_examples == expected_n.
    """
    stats = aggregate_jsonl(jsonl_path)
    complete = stats["n_examples"] == expected_n
    return {
        "version": version,
        "published": datetime.now(timezone.utc).isoformat(),
        "suite": suite,
        "suite_hash": suite_hash,
        "n_examples": stats["n_examples"],
        "complete": complete,
        "models": [
            {
                "id": model_id,
                "label": model_label,
                "provider": provider,
                **{k: v for k, v in stats.items() if k != "n_examples"},
                "n_examples": stats["n_examples"],
            }
        ],
    }
