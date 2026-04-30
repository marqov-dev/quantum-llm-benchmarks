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
                    ex.prompt,
                    model_id=model_config.id,
                    temperature=model_config.temperature,
                    max_tokens=model_config.max_tokens,
                    stop=None,
                )
                provider_error = None
                break
            except Exception as e:
                provider_error = str(e)
                if attempt < model_config.max_retries:
                    print(f"(retry {attempt + 1}) ", end="", flush=True)

        # validate_example returns (ValidationResult, extracted_code)
        result, extracted_code = validate_example({"response": generated, "category": ex.category})
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
    from quantum_eval.results import build_release_json

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"File not found: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    with jsonl_path.open() as f:
        header = json.loads(f.readline())

    model_id = header.get("model", args.model_id or "unknown")
    suite_name = header.get("suite", "humaneval")
    suite_hash = get_header_suite_hash(jsonl_path) or "unknown"
    if suite_name not in SUITE_SIZES:
        print(f"Unknown suite '{suite_name}' in JSONL header. Cannot verify completeness.", file=sys.stderr)
        sys.exit(1)
    expected_n = SUITE_SIZES[suite_name]

    try:
        model_config = get_model(model_id)
        label = model_config.label
        provider = model_config.provider_name
    except KeyError:
        label = model_id
        provider = "unknown"

    release = build_release_json(
        jsonl_path=jsonl_path,
        version=args.tag,
        suite=suite_name,
        suite_hash=suite_hash,
        model_id=model_id,
        model_label=label,
        provider=provider,
        expected_n=expected_n,
    )

    out_json = Path("results") / f"{args.tag}.json"
    out_json.parent.mkdir(exist_ok=True)
    out_json.write_text(json.dumps(release, indent=2))
    print(f"Written: {out_json}")

    stats = release["models"][0]
    print(
        f"Semantic pass: {stats['semantic_pct']}% ({stats['semantic_pass']}/{stats['n_examples']}) "
        f"[{stats['semantic_ci_low']}%–{stats['semantic_ci_high']}% CI]"
    )
    if not release["complete"]:
        print(
            f"Warning: Partial run: {stats['n_examples']}/{expected_n} examples. "
            "release JSON has complete=false.",
            file=sys.stderr,
        )

    if not args.no_push:
        subprocess.run(["git", "add", str(out_json)], check=True)
        subprocess.run(
            ["git", "commit", "-m",
             f"results: {args.tag} — {model_id} {stats['semantic_pct']}% semantic"],
            check=True,
        )
        subprocess.run(["git", "tag", args.tag], check=True)
        subprocess.run(["git", "push", "origin", "main", "--tags"], check=True)
        print(f"Published: tag {args.tag} pushed to origin.")
    else:
        print(f"--no-push: skipped git commit/tag/push. JSON written to {out_json}.")


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

    # publish
    pub_p = sub.add_parser("publish", help="Aggregate JSONL, commit JSON to main, cut release tag")
    pub_p.add_argument("jsonl", help="Path to the result JSONL file")
    pub_p.add_argument("--tag", required=True, help="Version tag e.g. 2026-04")
    pub_p.add_argument("--model-id", dest="model_id", help="Override model ID if not in JSONL header")
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
