# quantum-llm-benchmarks

Open benchmark for evaluating LLMs on quantum code generation. Run by [Marqov](https://marqov.ai). Results published monthly at [polystacks.dev/benchmarks](https://polystacks.dev/benchmarks).

## Install

```bash
pip install quantum-llm-benchmarks
# or without installing:
uvx quantum-llm-benchmarks --help
```

## Quick start

```bash
export ANTHROPIC_API_KEY=sk-ant-...
quantum-llm-benchmarks run --model claude-sonnet-4-6 --suite humaneval
```

Results are written to `results/claude-sonnet-4-6_humaneval.jsonl` as they go. If the run is interrupted, resume with `--resume` (ID-based, safe across suite updates).

## Registered models

```
quantum-llm-benchmarks list
```

Shows all 15 registered models and which API keys are set. Covers Anthropic, Google, DeepSeek, Kimi, Qwen, Mistral, Groq, and local Ollama (IBM Qiskit fine-tuned).

## Provider adapters

Native SDKs are used for the three major providers to avoid compat-layer methodology artifacts:

| Provider | Adapter | Models |
|----------|---------|--------|
| Anthropic | native `anthropic` SDK | Claude |
| OpenAI | native `openai` SDK | GPT |
| Google | native `google-genai` SDK | Gemini |
| All others | OpenAI-compat via `openai` SDK | DeepSeek, Ollama, Groq, Mistral, Kimi, Qwen, IBM GGUF |

## Benchmark suite

151-example [Qiskit HumanEval](https://github.com/Qiskit/qiskit-code-assistant-benchmark) benchmark. IDs: `qiskitHumanEval_0` … `qiskitHumanEval_150`.

Three-level validation: **syntax** → **execution** → **semantic**. Only semantic pass counts on the leaderboard.

## Resume

```bash
quantum-llm-benchmarks run --model gemini-2.5-flash --suite humaneval --resume
```

Resume skips already-completed example IDs. If the suite file changed since the run started, the CLI refuses and requires `--force-mismatch`.

## Publish results

```bash
quantum-llm-benchmarks publish results/claude-sonnet-4-6_humaneval.jsonl --tag 2026-04
```

Aggregates results, writes `results/2026-04.json`, commits to main, and cuts a release tag. The JSON is fetched by polystacks.dev for the live leaderboard.

> ⚠️ **Methodology prerequisite:** Complete `METHODOLOGY.md` before running `publish` for the first time.

## Attribution

Maintained by [Marqov](https://marqov.ai). Community PRs to add models to the registry are welcome.
