#!/usr/bin/env python3
"""
v1.1 Behavioral-46 sub-benchmark analysis and Wilson 95% CIs on KL estimates.

Task A: Per-model pass rates on behavioral-46 vs structural-104.
Task B: Wilson 95% confidence intervals on all KL pass rates.

Usage: uv run python scripts/analyze_v11_behavioral.py
"""
import json
import math
from pathlib import Path

V11_DIR = Path("results/v11")
CLASSIFICATION_PATH = Path("results/v11_test_classification.json")
OUTPUT_PATH = Path("results/v11_behavioral_analysis.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_classification() -> dict[str, str]:
    """Return {problem_id: test_class} mapping."""
    data = json.loads(CLASSIFICATION_PATH.read_text())
    return {item["id"]: item["test_class"] for item in data}


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
    import re
    stem = path.stem.replace("_humaneval", "")
    stem = re.sub(r"(\d)-(\d)", r"\1.\2", stem)
    return stem.replace("_", " ").replace("-", " ").title()


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson 95% CI. Returns (pass_rate, ci_low, ci_high)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p_hat = k / n
    z2 = z * z
    center = (p_hat + z2 / (2 * n)) / (1 + z2 / n)
    half_width = z * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (p_hat, max(0.0, center - half_width), min(1.0, center + half_width))


def pct(k: int, n: int) -> str:
    return f"{k/n*100:.1f}%" if n else "N/A"


def fmt_ci(ci_low: float, ci_high: float) -> str:
    return f"[{ci_low*100:.1f}%, {ci_high*100:.1f}%]"


# ---------------------------------------------------------------------------
# Task A: Behavioral-46 / Structural-104 sub-benchmark stats
# ---------------------------------------------------------------------------

def compute_subgroup_stats(rows: list[dict], class_map: dict[str, str], target_class: str) -> dict:
    """Compute exec/unit/kl pass rates for rows belonging to target_class."""
    subset = [r for r in rows if class_map.get(r["id"]) == target_class]
    n = len(subset)

    exec_pass = sum(1 for r in subset if r.get("execution_pass"))
    unit_pass_nonnull = sum(1 for r in subset if r.get("unit_test_pass") is True)
    # kl: None = unevaluable, True/False = evaluable
    kl_evaluable = [r for r in subset if r.get("kl_pass") is not None]
    kl_pass_count = sum(1 for r in kl_evaluable if r["kl_pass"] is True)
    kl_n = len(kl_evaluable)

    return {
        "n": n,
        "exec_pass": exec_pass,
        "exec_rate": round(exec_pass / n * 100, 1) if n else 0.0,
        "unit_pass": unit_pass_nonnull,
        "unit_rate": round(unit_pass_nonnull / n * 100, 1) if n else 0.0,
        "kl_pass": kl_pass_count,
        "kl_n": kl_n,
        "kl_rate": round(kl_pass_count / kl_n * 100, 1) if kl_n else None,
    }


# ---------------------------------------------------------------------------
# Task B: Wilson CIs
# ---------------------------------------------------------------------------

def wilson_table_row(label: str, k: int, n: int) -> dict:
    if n == 0:
        return {"label": label, "pass_rate": None, "ci_low": None, "ci_high": None, "n": n, "k": k}
    p_hat, ci_low, ci_high = wilson_ci(k, n)
    return {
        "label": label,
        "pass_rate": round(p_hat * 100, 1),
        "ci_low": round(ci_low * 100, 1),
        "ci_high": round(ci_high * 100, 1),
        "n": n,
        "k": k,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    class_map = load_classification()

    # Count classes
    class_counts: dict[str, int] = {}
    for cls in class_map.values():
        class_counts[cls] = class_counts.get(cls, 0) + 1

    files = sorted(V11_DIR.glob("*_humaneval.jsonl"))
    files = [f for f in files if not f.name.startswith("smoke_")]

    if not files:
        print(f"No v11 files in {V11_DIR}")
        return

    # -----------------------------------------------------------------------
    # Collect per-model stats
    # -----------------------------------------------------------------------
    behavioral_stats: dict[str, dict] = {}
    structural_stats: dict[str, dict] = {}
    kl_wilson_cis: dict[str, dict] = {}

    # For overall and by-class KL CIs, accumulate across all models
    all_rows_by_class: dict[str, list[dict]] = {"behavioral": [], "structural": [], "interface": []}
    all_rows_all: list[dict] = []

    models_data: list[tuple[str, dict, dict]] = []

    for fpath in files:
        label = model_label(fpath)
        rows = load_results(fpath)

        beh = compute_subgroup_stats(rows, class_map, "behavioral")
        struct = compute_subgroup_stats(rows, class_map, "structural")

        # Per-model KL CI (all problems, not just behavioral/structural)
        kl_ev = [r for r in rows if r.get("kl_pass") is not None]
        kl_k = sum(1 for r in kl_ev if r["kl_pass"] is True)
        kl_n = len(kl_ev)
        row_kl = wilson_table_row(label, kl_k, kl_n)

        key = fpath.stem.replace("_humaneval", "")
        behavioral_stats[key] = {
            "exec_pass_rate": beh["exec_rate"],
            "unit_pass_rate": beh["unit_rate"],
            "kl_pass_rate": beh["kl_rate"],
            "kl_n": beh["kl_n"],
            "n": beh["n"],
        }
        structural_stats[key] = {
            "exec_pass_rate": struct["exec_rate"],
            "unit_pass_rate": struct["unit_rate"],
            "kl_pass_rate": struct["kl_rate"],
            "kl_n": struct["kl_n"],
            "n": struct["n"],
        }
        kl_wilson_cis[key] = {
            "pass_rate": row_kl["pass_rate"],
            "ci_low": row_kl["ci_low"],
            "ci_high": row_kl["ci_high"],
            "n": kl_n,
            "k": kl_k,
        }

        # Accumulate for overall
        for r in rows:
            cls = class_map.get(r["id"])
            if cls in all_rows_by_class:
                all_rows_by_class[cls].append(r)
        all_rows_all.extend(rows)

        models_data.append((label, beh, struct))

    # -----------------------------------------------------------------------
    # Overall KL CI (all evaluable pairs, all models)
    # -----------------------------------------------------------------------
    kl_all_ev = [r for r in all_rows_all if r.get("kl_pass") is not None]
    kl_all_k = sum(1 for r in kl_all_ev if r["kl_pass"] is True)
    kl_all_n = len(kl_all_ev)
    overall_p, overall_lo, overall_hi = wilson_ci(kl_all_k, kl_all_n)

    # -----------------------------------------------------------------------
    # By-class KL CIs
    # -----------------------------------------------------------------------
    kl_by_class: dict[str, dict] = {}
    for cls in ["behavioral", "structural"]:
        cls_rows = all_rows_by_class[cls]
        ev = [r for r in cls_rows if r.get("kl_pass") is not None]
        k = sum(1 for r in ev if r["kl_pass"] is True)
        n = len(ev)
        p, lo, hi = wilson_ci(k, n)
        kl_by_class[cls] = {
            "pass_rate": round(p * 100, 1) if n else None,
            "ci_low": round(lo * 100, 1) if n else None,
            "ci_high": round(hi * 100, 1) if n else None,
            "n": n,
            "k": k,
        }

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------

    print("\n" + "=" * 110)
    print("TASK A: BEHAVIORAL-46 vs STRUCTURAL-104 SUB-BENCHMARK ANALYSIS")
    print("=" * 110)
    print(f"\nTest class distribution: {class_counts}")
    print(f"(1 'interface' problem exists and is excluded from both sub-benchmarks)\n")

    # --- Behavioral-46 table ---
    print("-" * 100)
    print("BEHAVIORAL-46 (harder: state correctness checks via .equiv(), Statevector, state_fidelity())")
    print("-" * 100)
    print(f"{'Model':<42} {'Exec%':>7} {'Unit%':>7} {'KL%':>7} {'KL n':>6}")
    print("-" * 100)

    # Rank by unit_test pass rate (descending)
    beh_ranked = sorted(models_data, key=lambda x: x[1]["unit_rate"], reverse=True)
    beh_ranks: dict[str, int] = {}
    for rank, (lbl, beh, _) in enumerate(beh_ranked, 1):
        kl_str = f"{beh['kl_rate']:.1f}%" if beh["kl_rate"] is not None else "N/A"
        print(f"{rank:>2}. {lbl:<38} {beh['exec_rate']:>6.1f}% {beh['unit_rate']:>6.1f}% {kl_str:>7} {beh['kl_n']:>6}")
        beh_ranks[lbl] = rank

    # --- Structural-104 table ---
    print()
    print("-" * 100)
    print("STRUCTURAL-104 (easier: circuit shape/gate checks)")
    print("-" * 100)
    print(f"{'Model':<42} {'Exec%':>7} {'Unit%':>7} {'KL%':>7} {'KL n':>6}")
    print("-" * 100)

    struct_ranked = sorted(models_data, key=lambda x: x[2]["unit_rate"], reverse=True)
    struct_ranks: dict[str, int] = {}
    for rank, (lbl, _, struct) in enumerate(struct_ranked, 1):
        kl_str = f"{struct['kl_rate']:.1f}%" if struct["kl_rate"] is not None else "N/A"
        print(f"{rank:>2}. {lbl:<38} {struct['exec_rate']:>6.1f}% {struct['unit_rate']:>6.1f}% {kl_str:>7} {struct['kl_n']:>6}")
        struct_ranks[lbl] = rank

    # --- Ranking comparison ---
    print()
    print("-" * 90)
    print("RANKING SHIFT: Behavioral-46 vs Structural-104 (by unit test pass rate)")
    print("-" * 90)
    print(f"{'Model':<42} {'Beh rank':>9} {'Str rank':>9} {'Delta':>7}")
    print("-" * 90)

    # Sort by behavioral rank
    all_labels = [lbl for lbl, _, _ in beh_ranked]
    max_shift = 0
    for lbl in all_labels:
        b = beh_ranks.get(lbl, 0)
        s = struct_ranks.get(lbl, 0)
        delta = b - s  # positive = dropped in behavioral ranking
        marker = " <<" if abs(delta) >= 3 else ""
        print(f"  {lbl:<40} {b:>9} {s:>9} {delta:>+7}{marker}")
        max_shift = max(max_shift, abs(delta))

    print(f"\n  Max rank shift: {max_shift} positions")
    movers = [(lbl, beh_ranks[lbl], struct_ranks[lbl]) for lbl in all_labels
              if abs(beh_ranks[lbl] - struct_ranks[lbl]) >= 3]
    if movers:
        print(f"  Notable movers (>=3 positions):")
        for lbl, b, s in sorted(movers, key=lambda x: abs(x[1]-x[2]), reverse=True):
            direction = "up" if b < s else "down"
            print(f"    {lbl}: rank {s} (structural) -> rank {b} (behavioral), moved {direction} {abs(b-s)} places")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("TASK B: WILSON 95% CIs ON KL DIVERGENCE PASS RATES")
    print("=" * 110)

    # Overall
    print(f"\nOverall KL pass rate (all evaluable pairs pooled across all models):")
    print(f"  k={kl_all_k}, n={kl_all_n}, "
          f"pass%={overall_p*100:.1f}%, "
          f"95% CI {fmt_ci(overall_lo, overall_hi)}")

    # By test class
    print(f"\nKL pass rate by test_class:")
    for cls in ["structural", "behavioral"]:
        d = kl_by_class[cls]
        if d["n"]:
            p, lo, hi = wilson_ci(d["k"], d["n"])
            print(f"  {cls:<12}: k={d['k']:>3}, n={d['n']:>3}, "
                  f"pass%={p*100:.1f}%, 95% CI {fmt_ci(lo, hi)}")
        else:
            print(f"  {cls:<12}: no evaluable pairs")

    # Per-model
    print()
    print("-" * 90)
    print(f"{'Model':<42} {'KL pass%':>9} {'95% CI':>22} {'n eval':>7} {'k':>5}")
    print("-" * 90)

    # Rank by KL pass rate
    per_model_kl = []
    for fpath in files:
        lbl = model_label(fpath)
        key = fpath.stem.replace("_humaneval", "")
        d = kl_wilson_cis[key]
        per_model_kl.append((lbl, d))

    per_model_kl.sort(key=lambda x: (x[1]["pass_rate"] is not None, x[1]["pass_rate"] or 0), reverse=True)

    for lbl, d in per_model_kl:
        if d["n"] == 0:
            print(f"  {lbl:<40} {'N/A':>9} {'N/A':>22} {d['n']:>7} {d['k']:>5}")
        else:
            ci_str = fmt_ci(d["ci_low"] / 100, d["ci_high"] / 100)
            print(f"  {lbl:<40} {d['pass_rate']:>8.1f}% {ci_str:>22} {d['n']:>7} {d['k']:>5}")

    print()

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    output = {
        "behavioral_46_stats": behavioral_stats,
        "structural_104_stats": structural_stats,
        "kl_wilson_cis": kl_wilson_cis,
        "kl_by_class_cis": {
            "structural": kl_by_class["structural"],
            "behavioral": kl_by_class["behavioral"],
        },
        "overall_kl_ci": {
            "pass_rate": round(overall_p * 100, 1),
            "ci_low": round(overall_lo * 100, 1),
            "ci_high": round(overall_hi * 100, 1),
            "n": kl_all_n,
            "k": kl_all_k,
        },
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
