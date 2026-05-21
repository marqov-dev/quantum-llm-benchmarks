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
MAX_EXAMPLES = 30  # calibrate on a random sample
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
    """Return the complete function definition for the canonical circuit runner.

    Uses extracted_code from the JSONL, which contains the full runnable function
    (imports + def line + docstring + body). The instruction field is natural-language
    text only and must NOT be used to construct Python code.

    Raises ValueError if extracted_code is absent so calibration fails loudly
    rather than silently producing a broken function body with no def line.
    """
    code = ex.get("extracted_code")
    if code:
        return code
    raise ValueError(
        f"Example {ex.get('id', '?')} has no extracted_code field. "
        "Cannot construct a runnable function for calibration."
    )


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

        # Run the same canonical circuit N times with different seeds.
        seeds = list(range(N_RUNS))
        distributions = {}
        for seed in seeds:
            try:
                distributions[seed] = run_canonical_circuit(
                    canonical_code, entry_point=entry_point, shots=SHOTS, seed=seed
                )
            except Exception as e:
                print(f"  {ex['id']} seed={seed}: FAILED ({e})")

        # Compute KL between every pair of independent runs.
        # Note: these pairs are not iid — pairs sharing a seed share an empirical
        # distribution. Fine for confirming tau is in the right ballpark.
        kl_values = []
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
