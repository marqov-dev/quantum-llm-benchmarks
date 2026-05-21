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
| Stop sequences | None |
| Self-correction retries | None (methodology). API error retries: 2 — these are infrastructure, not methodology. |

**Rationale:**

- **No system prompt.** System prompt handling differs between providers (Anthropic's `system` parameter, OpenAI's system role message, Gemini's `system_instruction`). Embedding the instruction in the user turn ensures identical handling across all three native SDKs.
- **Zero-shot.** All quantum-specific benchmarks in the literature (Qiskit HumanEval, QuanBench, QuanBench+) use zero-shot. Few-shot performance depends heavily on example selection, making cross-model comparison harder to defend.
- **Temperature = 0.0 (greedy).** The field consensus for leaderboard comparison as of 2025–2026: BigCodeBench, EvalPlus, and QuanBench+ all use greedy decoding for their primary Pass@1 metric. Greedy is fully deterministic — running the same model twice produces the same score, which is the minimum bar for a reproducible public benchmark.
- **No stop sequences.** The original HumanEval stop sequences (`\ndef `, `\nclass `, `\n#`, `\nif __name__`) were designed for *code completion* tasks where the model continues inside a pre-given function signature. The Qiskit HumanEval uses natural-language prompts requiring complete programs — models must write `def` to produce any callable code. Applying `\ndef ` as a stop sequence prevents function definitions entirely and produces truncated imports that always fail execution. Stop sequences are replaced by the prompt prefix (instructs the model to write code only) and `max_tokens=4096` as the natural generation bound.

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

---

## v1.1 Semantic Validation (in progress)

**Status:** Results from v1.1 are not directly comparable to v1.0.

### Approach A: Unit test execution

Each problem includes hand-authored unit tests from IBM's public dataset
(`qiskit-community/qiskit-human-eval`).
The generated code is executed, followed by the unit test assertions, in an
isolated subprocess with a 60-second timeout.

**Format mismatch:** IBM's tests call a function named `entry_point`. Our prompts
elicit standalone programs. If `entry_point` is not defined in the generated code,
a synthesised wrapper is used that runs the standalone code and captures the last
QuantumCircuit. Results are disaggregated by `defines_entry_point` (whether the
model happened to define the expected function) in the comparison analysis.

### Approach B: KL divergence on measurement distributions

The generated code and IBM's `canonical_solution` (invoked via `entry_point()`)
are each run on `AerSimulator` with 1024 shots and a fixed seed
(`seed_simulator=42`). The KL divergence between output distributions is computed
with additive smoothing (ε=1e-10).

**Pass criterion:** KL(P_gen ∥ P_ref) < τ = 0.05

Noise-floor calibration (canonical vs canonical, 30 examples, 5 independent runs)
confirmed the 95th percentile sampling KL is 0.0075, well below τ=0.05.
See `scripts/calibrate_kl.py`.

**Scope limitation:** Approach B is only applicable to problems where the canonical
solution returns a `QuantumCircuit` (approximately 13% of the benchmark). For the
remaining problems (returning Statevectors, Operators, counts dicts, etc.),
KL divergence records `kl_pass=null` (couldn't evaluate).

Citation: QuanBench+ (arXiv:2604.08570, ICLR 2026 Workshop on "I Can't Believe It's
Not Better"). τ and shot count follow QuanBench+ methodology (not stated explicitly
in the paper).

### Retroactive comparison

Both validators were run retroactively on stored `generated_code` from all 16 model
result sets (2,416 data points). Disagreements are analysed with:
- Cohen's κ (agreement beyond chance)
- McNemar's test (asymmetry between A-only and B-only cases)
- `defines_entry_point` split (isolates format-mismatch effect from genuine disagreement)

**Pass/fail/error semantics:** Each validator records three states: `True` (passed),
`False` (ran successfully but answer is wrong), `None` (could not evaluate — execution
error, timeout, or no circuit found). Agreement matrix denominators include only rows
where both methods produced a definitive result. Error rates are reported separately.

**Known edge case (Approach A):** If generated code contains defensive `assert`
statements that fail before the IBM test code runs, the result is categorised as
`unit_test_pass=False` ("test failed") rather than `None` ("couldn't evaluate"),
because both cases surface as `AssertionError`. This will slightly inflate the False
bucket for Approach A.

**Partial measurement (Approach B):** If generated code measures fewer qubits than the
canonical solution, the KL divergence is computed over different key spaces. With
additive smoothing, this typically produces a high divergence and a `False` result.

### v1.1 leaderboard columns

Two new columns alongside the existing `semantic_pct`:
- `unit_test_pct` — Approach A pass rate
- `kl_semantic_pct` — Approach B pass rate

v1.0 `semantic_pct` (execution pass) is retained for continuity.
