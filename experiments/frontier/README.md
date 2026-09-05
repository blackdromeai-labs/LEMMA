# LEMMA Frontier Search Benchmark

This is LEMMA's long-horizon, adversarial rewriting benchmark. It borrows three useful
properties from frontier evaluations: tasks should be difficult, have unambiguous end states,
and be checked by an outcome verifier. It is **not** Humanity's Last Exam, FrontierMath, or a
benchmark of broad mathematical reasoning. LEMMA currently accepts symbolic expressions rather
than expert natural-language problems, so making that claim would be misleading.

## Frozen suite

- 300 deterministic problems; all 300 are evaluated in every reported run.
- Five exact construction-depth tiers: 8, 12, 16, 24, and 32 steps.
- Two balanced tracks, `wrapped-ID` and `wrapped-OOD`, with 30 problems per track at each depth.
- Ten source families (five per track), each represented by exactly 30 problems. The source tasks
  preserve the original algebra, equation, logarithmic, power, and Pythagorean targets.
- Every source task is extended to its target depth with alternating two-step collision layers.
  Each layer exposes a verifier-accepted distributive distractor alongside the shorter identity
  rewrite.
- Search budget: 400 simulations per problem; maximum search depth: 40.
- Corpus SHA-256: `BDB0E0CCA74EE09E99734452A525AF6116CF8D5575B01319BCAA9453C97E0E4E`.

The checked reference sequence is construction metadata, not an input to search. It establishes
solvability but does not assert that the path is unique. A problem counts as solved only when the
solver reaches the exact parsed target and returns a replayable evidence status.

## Validation gates

`generate` refuses to freeze the corpus unless all of the following hold:

1. exactly 300 unique input-target pairs and IDs exist;
2. every track/depth cell has exactly 30 problems;
3. every expression survives parse-format round-tripping;
4. every named reference rule exists and has exactly one non-identity, verifier-accepted output;
5. replaying the complete reference sequence reaches the exact target;
6. the recorded construction depth equals the number of checked steps.

On the frozen corpus, reference states expose 1--3 verifier-accepted successors (mean 1.46,
also printed by `validate`).
That small branching factor is a limitation of LEMMA's current root-rewrite search space; the
benchmark compounds the branch collisions over long horizons instead of pretending the engine
supports broader mathematical tasks.

## Reproduce

From the repository root:

```powershell
cargo run -p mm-solver --example frontier_search_benchmark --release -- generate
cargo run -p mm-solver --example frontier_search_benchmark --release -- validate
cargo run -p mm-solver --example frontier_search_benchmark --release --features cuda -- run experiments/frontier/corpus.jsonl 400
```

The final command evaluates uniform, shallow-policy, and compositional-policy search over the
same full suite and writes `experiments/frontier/results.json`.

## Interpretation boundary

The two policy files predate this corpus, so the present run is a clean held-out comparison for
those fixed models. Once this public suite is used for model selection or retraining, it becomes a
development benchmark. A publication-grade successor should use independently authored or
privately held problems and disclose model access, attempts, compute, and confidence intervals.

Design references: [Humanity's Last Exam](https://www.nature.com/articles/s41586-025-09962-4),
[FrontierMath](https://arxiv.org/abs/2411.04872), and
[Terminal-Bench](https://github.com/laude-institute/terminal-bench).
