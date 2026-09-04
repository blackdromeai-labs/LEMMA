# Evaluation

Earlier headline benchmark scores have been withdrawn: the old scripts could print passes
while still exiting successfully on failures, and accepted overly broad output shapes (see
`crates/mm-solver/tests/evaluation.rs`'s module doc for the specifics). Current evaluation is
designed to fail closed.

```bash
# Exact solver outcomes and replayable traces
cargo test -p mm-solver --test evaluation -- --nocapture

# Registry reachability and no-op census
cargo test -p mm-rules --test rule_census -- --nocapture

# Acceptance by the verifier
cargo test -p mm-verifier --test rule_acceptance -- --nocapture

# Guardrail reachability (which verifier-accepted rules the search can actually reach)
cargo test -p mm-search --test guardrail_reachability -- --nocapture
```

The current pinned witness census contains **572 registered rules**:

| Verdict in the 228-expression witness corpus | Rules |
| --- | ---: |
| Transforms at least one witness | 138 |
| Applicable but produces no changed expression | 244 |
| Not reached by this corpus | 190 |

These numbers describe this specific corpus, not complete mathematical coverage. Read the
latest test output when evaluating a revision — this table goes stale the moment the rule
registry or the witness corpus changes.

## Seeded random evaluation

`crates/mm-solver/examples/seeded_eval.rs` runs a reproducible, randomly generated problem set
against the real solver and checks each result two independent ways: against an expected
value computed without calling into LEMMA, and against `assess_trace` to confirm the recorded
steps actually replay. It prints the seed it used, so a run is reproducible:

```bash
cargo run --release --example seeded_eval -p mm-solver -- <seed>
```

This is how the `canonicalize()` gap on nested constant folding (`2*(x*4)` vs `8*x`) was
found — it's a stress test for correctness, not a demo, and failures it reports are real
findings, not noise to explain away.

## Research integrity

Report results with the command, revision, configuration, model provenance, and exact test
output used to produce them. Do not describe a registered or applicable rule as working
unless it changes a representative expression and the result passes the intended verification
path.
