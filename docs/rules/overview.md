# Rule system overview

`mm-rules` holds LEMMA's registry of mathematical transformation rules. Each `Rule` pairs an
`is_applicable(expr) -> bool` check with an `apply(expr) -> Vec<RuleApplication>` transform, a
stable `RuleId`, a `RuleCategory`, and a `domains` tag the guardrail (`mm-boink`) uses to
pre-filter which rules the search even tries.

Search (`mm-search`'s beam/MCTS) proposes a rule application; `mm-verifier` independently
checks whether the claimed result actually follows before the search is allowed to use it. A
rule existing in the registry does not mean it does anything, and a rule doing something does
not mean the verifier will accept it — both are measured, not assumed. See
[`../verification.md`](../verification.md) for how that check works.

## Ground truth, not this document

Rule counts and the "does it actually work" question drift as the registry changes. The
authoritative source is always the live test output, not prose:

```bash
# Per-module transform / no-op / no-witness breakdown
cargo test -p mm-rules --test rule_census -- --nocapture

# Which of those the verifier will actually accept
cargo test -p mm-verifier --test rule_acceptance -- --nocapture

# Which accepted rules the search guardrail can actually reach
cargo test -p mm-search --test guardrail_reachability -- --nocapture
```

As of the last measured run: 572 rules registered, 138 transform at least one witness in the
228-expression corpus, 244 are applicable but never change anything (stubs — many of them
explicitly informational, e.g. theorem statements with no rewrite target), and 190 are never
reached by the current corpus at all. Of the 138 that transform, 119 are accepted by the
verifier on at least one witness; the rest are refused everywhere and cannot be used by
search regardless of how the guardrail routes them.

## Catalog

[`catalog.md`](catalog.md) is a by-topic map of what's registered, organized by ID range and
family (algebra, trigonometry, calculus, number theory, combinatorics). It is useful for
finding where a rule *should* live and what it claims to do. It is not a substitute for the
census above — a rule appearing in the catalog does not mean it transforms anything, is
verifier-accepted, or is reachable through the guardrail. Where the catalog and the test
output disagree, the test output is correct.

## Adding a rule

See [`CONTRIBUTING.md`](../../CONTRIBUTING.md).
