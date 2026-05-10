# QuantumEval Methodology

**Version:** v1.0  
**Locked:** 2026-05-11  
**Suite:** Qiskit HumanEval (151 examples, `qiskitHumanEval_0` … `qiskitHumanEval_150`)

This document defines the evaluation methodology for all results published to the leaderboard at polystacks.dev/benchmarks. Results are only comparable across models when this methodology is applied exactly. Any deviation must be disclosed.

---

## Prompt protocol

| Parameter | Value |
|-----------|-------|
| System prompt | None |
| User prompt prefix | `"Write executable Qiskit code only. Do not include explanations or markdown."` |
| Shot count | Zero-shot |
| Temperature | `0.0` (greedy decoding) |
| max_tokens | `4096` |
| Stop sequences | `["\nclass ", "\ndef ", "\n#", "\nif __name__"]` |
| Self-correction retries | None (methodology). API error retries: 2 — these are infrastructure, not methodology. |

**Rationale:**

- **No system prompt.** System prompt handling differs between providers (Anthropic's `system` parameter, OpenAI's system role message, Gemini's `system_instruction`). Embedding the instruction in the user turn ensures identical handling across all three native SDKs.
- **Zero-shot.** All quantum-specific benchmarks in the literature (Qiskit HumanEval, QuanBench, QuanBench+) use zero-shot. Few-shot performance depends heavily on example selection, making cross-model comparison harder to defend.
- **Temperature = 0.0 (greedy).** The field consensus for leaderboard comparison as of 2025–2026: BigCodeBench, EvalPlus, and QuanBench+ all use greedy decoding for their primary Pass@1 metric. Greedy is fully deterministic — running the same model twice produces the same score, which is the minimum bar for a reproducible public benchmark.
- **Stop sequences.** Inherited from the original HumanEval benchmark (Chen et al., 2021). Prevents generation from continuing past the function body into unrelated top-level constructs.

---

## Validation pipeline

Generated code is evaluated through three levels in sequence. A result is recorded at the highest level reached.

### Level 1 — Syntax

The generated code is parsed with `ast.parse()`. Failure at this level means the model produced code that is not valid Python.

### Level 2 — Execution

The code is executed in an isolated subprocess with a 30-second timeout. The subprocess runs with `MPLBACKEND=Agg` to suppress display backends. Failure at this level means the code crashes at runtime.

Deprecated Qiskit API patterns (e.g. `Aer.get_backend()`, `execute()`, `BasicAer`) are recorded as **warnings** on the result but do not prevent execution or semantic scoring. Code that runs correctly using an older API earns its points.

Test stubs (functions whose body contains only `pass`, `...`, or `assert` statements without implementation) and dead code (functions defined but never called) are detected between syntax and execution and scored at Level 1 only.

### Level 3 — Semantic

**v1.0:** Semantic pass equals execution pass for all examples in the Qiskit HumanEval suite. The suite's category is `"humaneval"`, for which no category-specific semantic check is implemented. A result that reaches Level 2 also reaches Level 3.

**This is a known limitation.** Code that runs without crashing but produces an incorrect quantum state (wrong gate sequence, wrong measurement, wrong algorithm) passes Level 3 in v1.0. This is disclosed here so leaderboard readers interpret "semantic pass rate" correctly: it measures execution correctness, not circuit correctness.

**v1.1 target:** KL divergence on measurement distributions against IBM's reference implementations, threshold τ = 0.05 calibrated from repeated canonical executions (following QuanBench+, 2026). Implementation is in progress. v1.1 results will not be directly comparable to v1.0 results — a new column will be added to the leaderboard when the methodology upgrade ships.

---

## Pass metric

**Primary:** Pass@1 (greedy). The fraction of the 151 examples where the generated code reaches each validation level.

**Confidence intervals:** Wilson 95% CI, computed via `scipy.stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method='wilson')`. All percentages are integers; pass counts are published alongside so readers can compute their own intervals.

**Supplementary:** Pass@5 at temperature=0.8 may be reported in future releases as a ceiling measure. Not reported in v1.0.

---

## What is not on the leaderboard

- **Latency / cost.** Not measured.
- **Instruction-following quality.** Not measured beyond syntax/execution/semantic pass.
- **Multi-turn / agentic use.** Not measured. Config B (Claude + MCP tools) is a separate internal experiment.
- **Frameworks other than Qiskit.** Cirq and PennyLane are out of scope for v1.

---

## Reproducibility

Every published result includes:
- The suite hash (`sha256:...` of `humaneval.jsonl`) — confirms the suite was not modified between runs
- The `complete` flag — `true` only when all 151 examples were evaluated in a single uninterrupted or cleanly resumed run
- The version tag (e.g. `2026-05`) — links to the GitHub release, which contains the raw JSONL output

Anyone can reproduce a result by running:

```bash
export <PROVIDER_API_KEY>=...
quantum-llm-benchmarks run --model <model-id> --suite humaneval
```

with temperature overridden to 0.0 and the stop sequences above applied. The CLI's `--resume` flag uses ID-based resume with suite hash verification, so interrupted runs can be continued safely.
