# Collaboration Pitch — Draft

*This is a template. Personalise the opener, [THEIR NAME], [THEIR INSTITUTION], and the
"extension we'd propose" section based on what you know about the collaborator.*

---

**Subject:** Co-authorship opportunity — quantum code generation benchmark (18 models, open data)

Hi [THEIR NAME],

I've been building an open benchmark for evaluating LLMs on Qiskit quantum code generation,
and I think you'd find the results interesting — and might want to co-author the paper.

**What we've built:**

We evaluated 18 models on IBM's 151-problem Qiskit HumanEval suite — frontier models
(Claude Opus 4.7, Gemini 2.5 Pro, DeepSeek V4, Qwen3 480B, etc.) plus IBM's own
domain-specific fine-tuned models (Qwen2.5-Coder-14B-Qiskit, Mistral-Small-3.2-24B-Qiskit)
run locally via Ollama. We also built two semantic validators beyond execution pass:
IBM's own unit tests, and KL divergence on measurement distributions.

The headline results are genuinely surprising:

- IBM's 14B fine-tuned model competes on execution pass with 400B+ frontier models —
  strongest efficiency signal we've seen in the dataset
- IBM's unit tests pass at 76–99% across all models, but are nearly uninformative:
  the wrapper is so permissive that `unit_test_pass` is never `False`, only `True` or null
- KL divergence is semantically rigorous but only applies to 6% of evaluation pairs —
  most of IBM's benchmark problems return `Statevector`, `Operator`, or `float`, not a
  measurable `QuantumCircuit`
- Among those 153 evaluable pairs, behavioral tests (quantum state verification) catch
  failures that structural tests (circuit shape checks) miss entirely (48% vs 77% KL pass,
  non-overlapping Wilson CIs)

The code, all model outputs (2,416 data points), and reproduction scripts are public at:
https://github.com/marqov-dev/quantum-llm-benchmarks

**Verification in an afternoon:**

REPRODUCING.md walks through all six key claims with exact commands and expected output.
No API keys needed — everything runs against pre-computed results. You could verify or
falsify our main findings in under two hours.

**The paper:**

We're writing this up as a benchmark paper. The core contribution is:

1. A methodology-locked open leaderboard (18 models, reproducible at temperature=0)
2. An empirical analysis of two semantic validation approaches and why both are incomplete
3. A static AST classification of IBM's test suite (104 structural / 46 behavioral / 1 interface)
   and what that reveals about what the benchmark actually measures
4. The IBM fine-tuned model efficiency result

**Where we'd like your help:**

[Choose one of the following and personalise:]

*Option A — Verification + independent model run:*
Run the benchmark on [MODEL FROM THEIR INFRASTRUCTURE] and verify our methodology
independently. We'd co-author the paper together, with your independent run strengthening
the reproducibility claim.

*Option B — Statevector/unitary comparison (the open problem):*
The honest gap in our work: 131 of 151 problems return types other than QuantumCircuit,
so KL divergence can't evaluate them. Statevector comparison, unitary equivalence checking,
and output value comparison for those problems is the next research step — and the one that
would make this a rigorous full-coverage semantic evaluation. If this aligns with your
group's interests, co-authoring that extension would make a strong paper in its own right.

*Option C — Domain expertise review:*
We'd value a review of our test classification rules (behavioral vs structural) from
someone with deep Qiskit domain knowledge, to confirm the classification is correct
quantum-mechanically, not just syntactically.

**Co-authorship:**

I'm proposing co-first-authorship for a substantive contribution (independent model run,
extension implementation, or domain review that shapes the methodology section).

Happy to jump on a call this week to discuss. The repo is fully public and REPRODUCING.md
is the fastest way to get a feel for what we've done.

Best,
[YOUR NAME]

---

## Suggested targets (in priority order)

### Tier 1 — Highest fit

**IBM Research (Zurich / Yorktown Heights)**
- They made qiskit-human-eval; they'd be highly motivated
- Angle: "We used your benchmark rigorously and found things your own paper didn't"
- Contact path: paper authors on arXiv:2406.14712
- Best pitch: Option C (domain review) or Option B (statevector extension)

**Q-CTRL**
- AI-native quantum company, active in quantum error correction + ML
- Angle: code generation quality matters for their product pipeline
- Contact path: their research team (research@q-ctrl.com or LinkedIn)
- Best pitch: Option A (run their models) or Option B

**ETH Zurich — Quantum Computing Group**
- Active in quantum software and benchmarking
- Best pitch: Option B (statevector extension as a research project)

### Tier 2 — Good fit

**MIT — Quantum Photonics / CS groups**
- Less Qiskit-specific but strong ML evaluation expertise
- Best pitch: methodology review + co-author on the benchmark design paper

**Classiq**
- Quantum software company; benchmark results directly affect their product positioning
- Best pitch: run their compiler output through the benchmark; business angle

**TU Delft — QuTech**
- European quantum computing hub, active in quantum software
- Best pitch: Option B or domain review

### Tier 3 — Worth a try

**Quantinuum** — quantum software, might want to benchmark their models  
**Pasqal / Quera** — neutral atom companies with growing software stacks  
**Any academic group working on LLM code generation** — quantum is a differentiating angle
