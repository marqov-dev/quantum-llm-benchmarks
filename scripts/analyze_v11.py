#!/usr/bin/env python3
"""
v1.1 comparison analysis: Approach A (unit tests) vs Approach B (KL divergence).

Produces:
  1. Per-model table: v1.0 semantic%, unit_test%, kl_semantic%, defines_entry_point%
  2. Overall 2x2 agreement matrix with Cohen's kappa
  3. McNemar's test on (A-only vs B-only) asymmetry
  4. Disagreement breakdown split by defines_entry_point flag
  5. Phi coefficients (per-example, binary Pearson)

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

    Handles version dots: `claude-sonnet-4-6` -> `Claude Sonnet 4.6`.
    Pattern: digit-digit sequences become digit.digit before title-casing.
    """
    import re
    stem = path.stem.replace("_humaneval", "")
    stem = re.sub(r"(\d)-(\d)", r"\1.\2", stem)  # 4-6 -> 4.6
    return stem.replace("_", " ").replace("-", " ").title()


def agreement_matrix(rows: list[dict]) -> dict:
    # Only include rows where BOTH methods produced a definitive result (True or False).
    # None = couldn't evaluate (error/timeout). Excluding these means "both_fail" in the
    # matrix genuinely means "ran and failed" -- not "one or both errored out."
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
    Use for the overall row (large n) where chi-squared is appropriate.
    """
    a_only, b_only = mat["a_only"], mat["b_only"]
    denom = a_only + b_only
    if denom == 0:
        return 1.0
    chi2 = (abs(a_only - b_only) - 1) ** 2 / denom
    from scipy.stats import chi2 as chi2_dist
    return float(chi2_dist.sf(chi2, 1))


def mcnemar_p_exact(mat: dict) -> float:
    """McNemar's exact binomial test. Use for per-model cells (small n)."""
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
    # Skip smoke test files
    files = [f for f in files if not f.name.startswith("smoke_")]
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
        p_mcn = mcnemar_p_exact(mat)  # exact binomial for per-model
        agree = mat["both_pass"] + mat["both_fail"]
        n_err = mat["n_errored"]

        print(
            f"{label:<38} {pct(v10, n):>6} {pct(ut, n):>6} {pct(kl, n):>6}"
            f" {pct(dep, n):>6} {pct(n_err, n):>6} {kappa:>6.3f} {pct(agree, mat['n']):>7} {p_mcn:>8.4f}"
        )
        all_rows.extend(rows)
        model_stats.append({
            "label": label, "n": n,
            "v10_pct": round(v10 / n * 100) if n else 0,
            "unit_test_pct": round(ut / n * 100) if n else 0,
            "kl_pct": round(kl / n * 100) if n else 0,
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
        label_str = "defines entry_point=True " if dep_val else "defines entry_point=False"
        print(f"    {label_str}: n={len(subset)}, A-only={a_only}, B-only={b_only}")

    # --- Phi coefficients (per-example, binary Pearson) ---
    # Phi coefficients require consistent denominators.
    # Only use rows where BOTH compared methods produced a definitive result.
    # Mixing None (couldn't evaluate) with False (evaluated, wrong) is a category error.

    # For v1.0 vs unit_test: require unit_test_pass is not None
    ut_rows = [r for r in all_rows if r.get("unit_test_pass") is not None]
    v10_for_ut = [1 if r.get("semantic_pass") else 0 for r in ut_rows]
    ut_vals = [1 if r.get("unit_test_pass") else 0 for r in ut_rows]
    r_v10_ut = pearsonr(v10_for_ut, ut_vals)

    # For v1.0 vs KL and unit_test vs KL: require BOTH methods ran
    kl_rows = [r for r in all_rows
               if r.get("unit_test_pass") is not None and r.get("kl_pass") is not None]
    v10_for_kl = [1 if r.get("semantic_pass") else 0 for r in kl_rows]
    ut_for_kl = [1 if r.get("unit_test_pass") else 0 for r in kl_rows]
    kl_vals = [1 if r.get("kl_pass") else 0 for r in kl_rows]
    r_v10_kl = pearsonr(v10_for_kl, kl_vals)
    r_ut_kl = pearsonr(ut_for_kl, kl_vals)

    print(f"\n  Per-example phi coefficient (binary Pearson):")
    print(f"    v1.0 semantic vs unit_test:  φ = {r_v10_ut:.3f}  (n={len(ut_rows)})")
    print(f"    v1.0 semantic vs KL:         φ = {r_v10_kl:.3f}  (n={len(kl_rows)}, both methods evaluable)")
    print(f"    unit_test vs KL:             φ = {r_ut_kl:.3f}  (n={len(kl_rows)}, both methods evaluable)")

    # --- Save JSON ---
    out = Path("results/v11_analysis.json")
    out.write_text(json.dumps({
        "model_stats": model_stats,
        "overall_matrix": overall,
        "cohens_kappa": kappa_all,
        "mcnemar_p": p_all,
        "phi_coefficients": {
            "v10_vs_unit_test": {"phi": r_v10_ut, "n": len(ut_rows)},
            "v10_vs_kl": {"phi": r_v10_kl, "n": len(kl_rows)},
            "unit_test_vs_kl": {"phi": r_ut_kl, "n": len(kl_rows)},
        },
    }, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
