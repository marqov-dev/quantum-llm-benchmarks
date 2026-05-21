import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from quantum_eval.harness import (
    compute_suite_hash,
    get_completed_ids,
    get_header_suite_hash,
    load_suite,
    write_header,
    append_result,
)
from quantum_eval.registry import load_registry, get_model
from quantum_eval.validator import validate_example, ValidationLevel

_DATA_DIR = Path(__file__).parent / "_data"
SUITES = {
    "humaneval": _DATA_DIR / "benchmarks" / "humaneval" / "humaneval.jsonl",
}
SUITE_SIZES = {"humaneval": 151}

# Methodology-locked parameters applied when --leaderboard is set.
# These must match METHODOLOGY.md exactly. Do not change without updating the doc.
LEADERBOARD_TEMPERATURE = 0.0
LEADERBOARD_STOP = None  # No stop sequences: Qiskit HumanEval is generation not completion
LEADERBOARD_PROMPT_PREFIX = "Write executable Qiskit code only. Do not include explanations or markdown.\n\n"


def cmd_list(args: argparse.Namespace) -> None:
    configs = load_registry()
    print(f"{'ID':<50} {'Label':<35} {'Provider':<12} {'Key'}")
    print("-" * 105)
    for c in configs:
        if not c.api_key_env:
            key_status = "local"
        elif os.environ.get(c.api_key_env):
            key_status = "✓ set"
        else:
            key_status = f"✗ {c.api_key_env} missing"
        print(f"{c.id:<50} {c.label:<35} {c.provider_name:<12} {key_status}")


def cmd_run(args: argparse.Namespace) -> None:
    suite_path = SUITES.get(args.suite)
    if suite_path is None:
        print(f"Unknown suite '{args.suite}'. Available: {list(SUITES)}", file=sys.stderr)
        sys.exit(1)

    try:
        model_config = get_model(args.model)
    except KeyError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    suite_hash = compute_suite_hash(suite_path)
    examples = load_suite(suite_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume and output_path.exists():
        existing_hash = get_header_suite_hash(output_path)
        if existing_hash and existing_hash != suite_hash and not args.force_mismatch:
            print(
                f"Suite hash mismatch:\n  file:    {existing_hash}\n  current: {suite_hash}\n"
                "The suite changed since this run started. "
                "Add --force-mismatch to resume anyway (results may be inconsistent).",
                file=sys.stderr,
            )
            sys.exit(1)
        completed = get_completed_ids(output_path)
        print(f"Resuming — {len(completed)}/{len(examples)} examples already done.")
    else:
        completed = set()
        write_header(output_path, suite=args.suite, suite_hash=suite_hash, model=args.model)

    try:
        provider = model_config.build_provider()
    except (EnvironmentError, ValueError) as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    if args.leaderboard:
        temperature = LEADERBOARD_TEMPERATURE
        stop = LEADERBOARD_STOP
        prompt_prefix = LEADERBOARD_PROMPT_PREFIX
        print(
            f"Leaderboard mode: temperature={temperature}, "
            f"prompt_prefix applied, stop=None (see METHODOLOGY.md)"
        )
    else:
        temperature = model_config.temperature
        stop = None
        prompt_prefix = ""

    todo = [ex for ex in examples if ex.id not in completed]
    total = len(examples)
    done = len(completed)

    for ex in todo:
        done += 1
        print(f"[{done}/{total}] {ex.id} ... ", end="", flush=True)

        provider_error: str | None = None
        generated = ""
        for attempt in range(model_config.max_retries + 1):
            try:
                generated = provider.generate(
                    prompt_prefix + ex.prompt,
                    model_id=model_config.id,
                    temperature=temperature,
                    max_tokens=model_config.max_tokens,
                    stop=stop,
                )
                provider_error = None
                break
            except Exception as e:
                provider_error = str(e)
                if attempt < model_config.max_retries:
                    print(f"(retry {attempt + 1}) ", end="", flush=True)

        # validate_example returns (ValidationResult, extracted_code)
        result, extracted_code = validate_example({
            "response": generated,
            "category": ex.category,
            "test_code": ex.test_code,
            "entry_point": ex.entry_point,
        })
        # provider_error takes precedence: it tells us why we got empty output.
        error = provider_error or result.error

        append_result(
            output_path,
            example_id=ex.id,
            generated_code=generated,
            syntax_pass=result.level_passed.value >= ValidationLevel.SYNTAX.value,
            execution_pass=result.level_passed.value >= ValidationLevel.EXECUTION.value,
            semantic_pass=result.level_passed.value >= ValidationLevel.SEMANTIC.value,
            error=error,
        )
        print(result.level_passed.name)

    print(f"\nDone. Results written to {output_path}")


def cmd_publish(args: argparse.Namespace) -> None:
    import subprocess
    from quantum_eval.results import build_multi_release_json

    jsonl_paths = [Path(p) for p in args.jsonl]
    for p in jsonl_paths:
        if not p.exists():
            print(f"File not found: {p}", file=sys.stderr)
            sys.exit(1)

    # Infer suite from the first file's header; all files must share the same suite.
    with jsonl_paths[0].open() as f:
        first_header = json.loads(f.readline())
    suite_name = first_header.get("suite", "humaneval")
    suite_hash = get_header_suite_hash(jsonl_paths[0]) or "unknown"
    if suite_name not in SUITE_SIZES:
        print(f"Unknown suite '{suite_name}' in JSONL header. Cannot verify completeness.", file=sys.stderr)
        sys.exit(1)
    expected_n = SUITE_SIZES[suite_name]

    registry = load_registry()
    model_specs = []
    for p in jsonl_paths:
        with p.open() as f:
            header = json.loads(f.readline())
        model_id = header.get("model", "unknown")
        try:
            cfg = get_model(model_id, registry=registry)
            label = cfg.label
            provider = cfg.provider_name
        except KeyError:
            label = model_id
            provider = "unknown"
        model_specs.append(
            {"jsonl_path": p, "model_id": model_id, "model_label": label, "provider": provider}
        )

    release = build_multi_release_json(
        model_specs=model_specs,
        version=args.tag,
        suite=suite_name,
        suite_hash=suite_hash,
        expected_n=expected_n,
    )

    out_json = Path("results") / f"{args.tag}.json"
    out_json.parent.mkdir(exist_ok=True)
    out_json.write_text(json.dumps(release, indent=2))
    print(f"Written: {out_json}")

    latest_json = Path("results") / "latest.json"
    latest_json.write_text(json.dumps(release, indent=2))
    print(f"Written: {latest_json}")

    print(f"\n{'Model':<35} {'Semantic':>10} {'n':>6}")
    print("-" * 55)
    for m in release["models"]:
        print(
            f"{m['label']:<35} {m['semantic_pct']:>9}%  "
            f"({m['semantic_pass']}/{m['n_examples']}) "
            f"[{m['semantic_ci_low']}–{m['semantic_ci_high']}% CI]"
        )
    if not release["complete"]:
        print("\nWarning: One or more models have incomplete runs (complete=false).", file=sys.stderr)

    if not args.no_push:
        n_models = len(release["models"])
        top = release["models"][0]
        subprocess.run(["git", "add", str(out_json), str(latest_json)], check=True)
        subprocess.run(
            ["git", "commit", "-m",
             f"results: {args.tag} — {n_models} models, top {top['label']} {top['semantic_pct']}% semantic"],
            check=True,
        )
        subprocess.run(["git", "tag", "-f", args.tag], check=True)
        subprocess.run(["git", "push", "origin", "main", "--tags", "--force"], check=True)
        print(f"\nPublished: tag {args.tag} pushed to origin.")
    else:
        print(f"\n--no-push: skipped git commit/tag/push. JSON written to {out_json}.")


def cmd_merge(args: argparse.Namespace) -> None:
    print(
        "merge is not implemented in v1.0.\n"
        "To publish results for multiple models, run publish once per model.\n"
        "Multi-model aggregation is tracked in issue #1.",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="quantum-llm-benchmarks",
        description="Evaluate LLMs on quantum code generation benchmarks.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    sub.add_parser("list", help="List all registered models and API key status")

    # run
    run_p = sub.add_parser("run", help="Run the benchmark against a model")
    run_p.add_argument("--model", required=True, help="Model ID from registry")
    run_p.add_argument("--suite", default="humaneval", help="Benchmark suite (default: humaneval)")
    run_p.add_argument("--output", default=None, help="Output JSONL path (default: results/<model>_<suite>.jsonl)")
    run_p.add_argument("--resume", action="store_true", help="Resume interrupted run (ID-based)")
    run_p.add_argument("--force-mismatch", action="store_true",
                       help="Resume even if suite hash changed (results may be inconsistent)")
    run_p.add_argument("--leaderboard", action="store_true",
                       help="Enforce methodology-locked parameters (temperature=0, stop sequences). "
                            "Required for results to be comparable to published leaderboard scores.")

    # publish
    pub_p = sub.add_parser("publish", help="Aggregate JSONL(s), commit JSON to main, cut release tag")
    pub_p.add_argument("jsonl", nargs="+", help="Path(s) to result JSONL file(s) — one per model")
    pub_p.add_argument("--tag", required=True, help="Version tag e.g. 2026-05")
    pub_p.add_argument("--no-push", action="store_true", help="Write JSON but skip git commit/tag/push")

    # merge (stub)
    sub.add_parser("merge", help="[not implemented in v1] Combine per-model JSONLs into one release")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "run":
        if args.output is None:
            safe_model = args.model.replace("/", "_").replace(":", "_")
            args.output = f"results/{safe_model}_{args.suite}.jsonl"
        cmd_run(args)
    elif args.command == "publish":
        cmd_publish(args)
    elif args.command == "merge":
        cmd_merge(args)


if __name__ == "__main__":
    main()
