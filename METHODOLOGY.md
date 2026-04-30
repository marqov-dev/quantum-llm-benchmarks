# QuantumEval Methodology

> ⚠️ **This document is incomplete.** It must be finalized before publishing the first leaderboard results. Results published without a locked methodology cannot be reproduced by third parties.

## Semantic equivalence

**Decision required:** What does it mean for a generated circuit to be semantically correct?

Options:
- **Statevector L2 distance** — compare statevectors of generated and reference circuits, pass if L2 < threshold
- **Measurement distribution TVD** — sample both circuits, compare distributions via Total Variation Distance
- **Unitary equivalence up to global phase** — compare unitary matrices, ignoring global phase

*TBD — locked here before first publish.*

## Prompt protocol

| Parameter | Value |
|-----------|-------|
| System prompt | TBD |
| Chat template | TBD (unified or per-provider) |
| Shot count | Zero-shot (current default) |
| Stop sequences | TBD per provider |
| Self-correction retries | 2 (current default) |
| Retry prompt | TBD |

## Provider generate() call parameters

| Parameter | Value |
|-----------|-------|
| temperature | TBD (per-model registry value or methodology-locked constant) |
| max_tokens | TBD |
| stop | TBD |

*These values must match across all models for results to be comparable.*
