# Verification model

| Status | Meaning |
| --- | --- |
| `Checked` | The trace replays from the exact input and every step was independently checked. |
| `Heuristic` | The trace replays, but at least one step relies on weaker evidence such as rule replay or numerical sampling. |
| `Unverified` | A required replay or verification check failed. |
| `Unsupported` | The requested verification mode is not implemented. |

Verification means equivalence checking by the implemented symbolic checks, with numerical
sampling as a weaker fallback where applicable. It is **not** a machine-checked formal proof,
and the project currently has no SMT backend.

## What "rule replay" trust actually covers

An expression containing a derivative or integral can't be numerically sampled, so
`Verifier::verify_step` falls back to trusting the rule that produced it — but only when the
derivative or integral being rewritten is the thing `before`/`after` *themselves* are, not
merely present somewhere inside a larger expression. That distinction was the difference
between a real defect and a fix: see `is_calculus_rewrite` in `crates/mm-verifier/src/lib.rs`
and the adversarial test pinning it in `crates/mm-verifier/tests/adversarial.rs`. Read that
doc comment before changing how `mm-search` applies rules to sub-terms.

## Numeric sampling is fail-closed by construction

`Expr::approx_equals` and `numerical::is_zero` sample random points and compare. If every
sample fails to evaluate on both sides (for example, both expressions are un-evaluable), the
result is `false`, not `true` — a comparison that never actually ran must not report as
"equivalent". See the same file for `is_calculus_rewrite`'s neighbor, the vacuous-truth fix.

## What is not implemented

`VerificationLevel::Formal` reports `Unsupported` rather than pretending a proof-checking
backend ran. There is no SMT solver, no full CAS, and no natural-language parser in this
project. See [`../README.md`](../README.md#project-status) for the current boundary of what
works end to end.
