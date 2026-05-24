# Reproducing the Results

This document lets an external collaborator verify the key claims in our benchmark
paper in under two hours, without re-running any models. All results are pre-computed
and checked into the repository.

**Repo:** https://github.com/marqov-dev/quantum-llm-benchmarks  
**Results tag:** `v1.1`  
**Dataset:** IBM Qiskit HumanEval (151 problems, Apache 2.0 — see [qiskit-community/qiskit-human-eval](https://github.com/qiskit-community/qiskit-human-eval))  
**Paper citation for dataset:** arXiv:2406.14712

---

## Setup (~5 minutes)

```bash
git clone https://github.com/marqov-dev/quantum-llm-benchmarks.git
cd quantum-llm-benchmarks
git checkout v1.1

# Python 3.12+ required
pip install uv
uv sync
```

Verify the install:

```bash
uv run quantum-llm-benchmarks list
# Should print all 18 registered models
```

---

## Claim 1: Test classification (104 structural / 46 behavioral / 1 interface)

**What this checks:** IBM's 151 HumanEval test files were classified by AST analysis into
structural tests (check circuit shape only), behavioral tests (verify quantum state/operator
correctness), and interface tests (isinstance check only).

```bash
uv run python scripts/classify_tests.py
```

Expected output (last few lines):

```
=== Test classification ===
  structural                104
  behavioral                 46
  interface                   1

=== Canonical output types ===
  QuantumCircuit             73
  other                      14
  dict                       13
  ...

Classification written to results/v11_test_classification.json
```

The classification rules are documented at the top of `scripts/classify_tests.py`.
The key design decisions (max-tier rule: behavioral > structural > interface;
behavioral signals include `.equiv(`, `AerSimulator`, `state_fidelity(`) are written
out explicitly and reviewable without running any code.

---

## Claim 2: Unit tests never fail — only pass or null

**What this checks:** Across all 2,416 model-problem pairs (16 models × 151 problems),
`unit_test_pass` is either `True` or `null` — never `False`. The wrapper is so permissive
that any code producing a `QuantumCircuit` object passes; the IBM tests cannot catch
semantic errors via this path.

```bash
# Count unit_test_pass values across all v1.1 result files
python3 -c "
import json, glob
counts = {'True': 0, 'False': 0, 'null': 0}
for f in glob.glob('results/v11/*_humaneval.jsonl'):
    for line in open(f):
        row = json.loads(line)
        if row.get('_header'):
            continue
        v = row.get('unit_test_pass')
        if v is True: counts['True'] += 1
        elif v is False: counts['False'] += 1
        else: counts['null'] += 1
print(counts)
print(f'Total pairs: {sum(counts.values())}')
"
```

Expected output:

```
{'True': 2213, 'False': 0, 'null': 203}
Total pairs: 2416
```

---

## Claim 3: KL divergence is evaluable for only 6% of pairs (153 / 2,416)

**What this checks:** KL divergence on measurement distributions requires that the
canonical solution returns a `QuantumCircuit` that can be simulated and measured.
Only 20 of 151 problems meet this criterion. The other 131 return `Statevector`,
`Operator`, `SparsePauliOp`, `float`, or other types.

```bash
python3 -c "
import json, glob
kl_pass = kl_null = kl_fail = 0
for f in glob.glob('results/v11/*_humaneval.jsonl'):
    for line in open(f):
        row = json.loads(line)
        if row.get('_header'):
            continue
        v = row.get('kl_pass')
        if v is True: kl_pass += 1
        elif v is False: kl_fail += 1
        else: kl_null += 1
total = kl_pass + kl_fail + kl_null
print(f'kl_pass={kl_pass}, kl_fail={kl_fail}, kl_null={kl_null}')
print(f'Evaluable: {kl_pass + kl_fail}/{total} ({100*(kl_pass+kl_fail)/total:.1f}%)')
print(f'Of evaluable, pass rate: {kl_pass}/{kl_pass+kl_fail} ({100*kl_pass/(kl_pass+kl_fail):.1f}%)')
"
```

Expected output:

```
kl_pass=97, kl_fail=56, kl_null=2263
Evaluable: 153/2416 (6.3%)
Of evaluable, pass rate: 97/153 (63.4%)
```

The 20 evaluable problems are those with `output_type = QuantumCircuit` in
`results/v11_test_classification.json`. Inspect with:

```bash
python3 -c "
import json
data = json.load(open('results/v11_test_classification.json'))
qc = [r for r in data if r['output_type'] == 'QuantumCircuit']
print(f'{len(qc)} problems return QuantumCircuit (KL-evaluable)')
"
```

---

## Claim 4: Behavioral KL pass rate (48%) is significantly lower than structural (77%)

**What this checks:** Among the 153 evaluable pairs, those from behavioral-test problems
pass KL at 47.9% [37%, 59%] vs structural at 76.8% [67%, 85%]. The Wilson 95% CIs
do not overlap — the gap is statistically defensible.

```bash
uv run python scripts/analyze_v11.py
```

Look for the `kl_by_test_class` section in the output, or inspect the pre-computed result:

```bash
python3 -c "
import json
d = json.load(open('results/v11_behavioral_analysis.json'))
for cls, v in d['kl_by_class_cis'].items():
    print(f\"{cls}: {v['pass_rate']:.1f}% [{v['ci_low']:.1f}%, {v['ci_high']:.1f}%] (n={v['n']})\")
overall = d.get('overall_kl_ci') or json.load(open('results/v11_analysis.json')).get('overall_kl')
"
```

Expected output:

```
structural: 76.8% [66.6%, 84.6%] (n=82)
behavioral: 47.9% [36.7%, 59.3%] (n=71)
```

The Wilson CI formula used: center = (p̂ + z²/2n) / (1 + z²/n), half-width =
z·√(p̂(1−p̂)/n + z²/4n²) / (1 + z²/n), z = 1.96.

---

## Claim 5: Unit tests and KL divergence are uncorrelated (Cohen's κ = 0.000)

**What this checks:** Among the 153 evaluable pairs, `unit_test_pass` and `kl_pass`
have Cohen's κ = 0.000 and McNemar's p < 0.0001. Unit tests never fail — so the only
observable cells are (unit_pass=True, kl_pass=True) and (unit_pass=True, kl_pass=False).

```bash
uv run python scripts/analyze_v11.py
```

Look for the confusion matrix and κ in the output, or inspect:

```bash
python3 -c "
import json
d = json.load(open('results/v11_analysis.json'))
print(json.dumps(d, indent=2))
"
```

Expected: `cohen_kappa` ≈ 0.000, `mcnemar_p` < 0.001.

---

## Claim 6: IBM fine-tuned 14B model matches frontier 400B+ models on execution pass

**What this checks:** `IBM Qwen2.5-Coder-14B-Qiskit` achieves 33.8% execution pass —
matching DeepSeek V4 Pro (35%), Gemini 2.5 Pro (36%), and Qwen3 Coder 480B (37%).

```bash
python3 -c "
import json
models = {
    'IBM Qwen2.5-Coder-14B-Qiskit': 'results/qwen-14b-qiskit_latest_humaneval.jsonl',
    'IBM Mistral-Small-3.2-24B-Qiskit': 'results/mistral-24b-qiskit_latest_humaneval.jsonl',
}
for label, path in models.items():
    rows = [json.loads(l) for l in open(path) if not json.loads(l).get('_header')]
    exec_pass = sum(1 for r in rows if r.get('execution_pass')) / len(rows)
    print(f'{label}: exec={exec_pass:.1%} (n={len(rows)})')
"
```

Expected:

```
IBM Qwen2.5-Coder-14B-Qiskit: exec=33.8% (n=151)
IBM Mistral-Small-3.2-24B-Qiskit: exec=31.8% (n=151)
```

The published v1.0 leaderboard for comparison:

```bash
python3 -c "
import json
d = json.load(open('results/latest.json'))
for m in sorted(d['models'], key=lambda x: -x['semantic_pass_count']):
    print(f\"{m['label']:40s} exec={m['execution_pass_count']}/{m['total']} ({m['execution_pass_count']/m['total']:.0%})\")
"
```

---

## Re-running a model yourself

To independently verify results for any model in the registry:

```bash
# Requires the relevant API key as an environment variable
export ANTHROPIC_API_KEY=...

uv run quantum-llm-benchmarks run \
  --model claude-opus-4-6 \
  --suite humaneval \
  --leaderboard \
  --output results/my_run_claude-opus-4-6_humaneval.jsonl
```

At temperature=0.0 (greedy), results are deterministic — re-running the same model
should produce the same scores. The suite hash in the output JSONL header should match
`sha256:0e46ce03acb2985af1ccfc64174a818e8753a5d3f875e17ee72dedb4e2f1a21c`.

For local Ollama models (IBM fine-tuned), ensure Ollama is running and the model is pulled:

```bash
ollama pull hf.co/Qiskit/Qwen2.5-Coder-14B-Qiskit-GGUF:Q4_K_M
uv run quantum-llm-benchmarks run \
  --model qwen-14b-qiskit:latest \
  --suite humaneval \
  --leaderboard
```

---

## Adding a new model

To benchmark a model not in the registry, add an entry to
`quantum_eval/_data/models/registry.yaml`:

```yaml
- id: your-model-id
  label: "Display Name"
  provider: compat                              # or anthropic, google
  base_url: "https://api.yourprovider.com/v1"
  api_key_env: YOUR_API_KEY_ENV_VAR
  temperature: 0.0
  max_tokens: 4096
  max_retries: 2
```

Then run as above. Results are welcome as a pull request — include the output JSONL and
a brief note on the model (parameters, training data if known).

---

## Questions and contact

Open an issue on GitHub or reach out directly. We welcome:
- Bug reports on the methodology or analysis scripts
- New model additions via pull request
- Extensions to the semantic validation framework (especially statevector/unitary comparison for the 131 non-QuantumCircuit problems)
