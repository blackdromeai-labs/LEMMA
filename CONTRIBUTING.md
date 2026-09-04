# Contributing to LEMMA

Thank you for considering contributing to LEMMA! This document reflects the current codebase
— if something here disagrees with the source, the source wins; open an issue or fix this
file in the same PR.

---

## Ways to Contribute

### 1. Add New Mathematical Rules

LEMMA's power comes from its rule library, but a rule only counts once it's reachable and
verifiable — see [`docs/rules/overview.md`](docs/rules/overview.md) before adding one.

### 2. Report Bugs

Found an expression LEMMA simplifies incorrectly, or a rule the census says transforms
something but the verifier never accepts? Open an issue with:
- Input expression
- Expected output
- Actual output
- The relevant `cargo test` command and its output (see [Evaluation](docs/evaluation.md))

### 3. Add Test Cases

More test coverage = more confidence. Add cases to:
- `crates/mm-solver/tests/evaluation.rs` — exact expected values, checked by canonical form
  and by replayable trace (`assess_trace`). This is the fail-closed correctness suite; see its
  module doc for what "fail closed" means here.
- `crates/mm-solver/examples/seeded_eval.rs` — a seeded random-problem generator, useful for
  finding cases hand-picked tests miss. Extend a problem family or add a new one; keep the
  expected-answer computation independent of LEMMA (never derive ground truth by calling the
  solver and trusting its own answer).
- `crates/mm-rules/tests/rule_census.rs` and `crates/mm-verifier/tests/rule_acceptance.rs` —
  pinned counts. If your change moves a rule between "transforms" / "no-op" / "not reached" or
  between "accepted" / "rejected", these tests will fail until you update the pinned constant
  — that's the intended signal, not a bug in the test.

### 4. Improve Documentation

- Fix typos
- Add examples
- Clarify confusing sections
- If you find a factual error in `docs/rules/catalog.md`, fix it in place — that document was
  assembled from older notes and hasn't been re-verified rule by rule.

---

## Development Setup

```bash
git clone https://github.com/blackdromeai-labs/LEMMA.git
cd LEMMA

# Build every workspace target
cargo build --workspace

# Launch the interactive workbench
cargo run -p mm-tui

# Run the same gates CI runs
cargo fmt --all -- --check
cargo check --workspace --all-targets --locked -j 1
cargo test --workspace --lib --tests --locked -j 1
cargo test --workspace --doc --locked -j 1
```

---

## Adding a New Rule

### Step 1: Choose the Right File

Rules live under `crates/mm-rules/src/`, organized by topic:

| Rule Type | Location |
|-----------|----------|
| Algebra | `algebra/` |
| Calculus | `calculus/` |
| Trigonometry | `trig/` |
| Geometry | `geometry/` |
| Equation solving | `equations.rs` |
| Inequalities | `inequalities.rs`, `inequality_chain.rs` |
| Number theory | `number_theory/` |
| Combinatorics | `combinatorics.rs` |
| Polynomials | `polynomial.rs`, `polynomials.rs` |
| Integration | `integration.rs` |

### Step 2: Write the Rule

```rust
fn your_rule_name() -> Rule {
    Rule {
        id: RuleId(/* pick the next unused ID; the registry test rejects duplicates */),
        name: "your_rule_name",
        category: RuleCategory::Simplification, // see the full list below
        description: "Human readable: pattern -> result",
        domains: &[], // empty = applicable regardless of detected problem domain;
                      // see docs/rules/overview.md before restricting this
        requires: &[],

        is_applicable: |expr, _ctx| {
            match expr {
                Expr::Add(a, b) => {
                    // your condition, matching the *specific* shape this rule handles —
                    // a rule that matches too broadly (e.g. "any Add") and returns a value
                    // unrelated to the match has caused real, live bugs in this codebase.
                    false
                }
                _ => false,
            }
        },

        apply: |expr, _ctx| {
            if let Expr::Add(a, b) = expr {
                return vec![RuleApplication {
                    result: /* your transformed expression, built from a and b */,
                    justification: "explanation".to_string(),
                }];
            }
            vec![]
        },

        reversible: false, // true if the rule can be validly applied backwards
        cost: 1,           // lower = preferred by search
    }
}
```

A rule's `apply` output must actually depend on the matched sub-structure. If a rule can't
compute a real value yet (a theorem statement without an executable rewrite), have it return
`expr.clone()` unchanged rather than a placeholder value — that keeps it correctly classified
as a no-op by the census instead of miscounted as "transforms".

### Step 3: Register the Rule

Add it to the module's collector function (e.g. `algebra_rules()`, `trig_rules()`) alongside
the other rules in that file.

### Step 4: Add a Witness and a Test

- Add an expression to `crates/mm-rules/src/witness.rs`'s corpus that actually exercises your
  rule, if none of the existing witnesses do — `rule_census` will otherwise report it as "not
  reached by this corpus".
- Add a case to `crates/mm-solver/tests/evaluation.rs` with the exact expected result.

### Step 5: Run the checks

```bash
cargo test --workspace --lib --tests --locked -j 1
cargo test -p mm-rules --test rule_census -- --nocapture
cargo test -p mm-verifier --test rule_acceptance -- --nocapture
```

If your rule moved the pinned counts in either census test, update the constant in the same
PR with a comment saying what moved and why.

---

## Expression Types

`Expr` (`crates/mm-core/src/expr.rs`) is the AST every rule matches against. It's larger than
a quick sketch can usefully show — read the source. Notable groups: arithmetic
(`Add`/`Sub`/`Mul`/`Div`/`Pow`/`Neg`), transcendental functions (`Sin`/`Cos`/`Tan`/`Exp`/`Ln`/
inverse trig), calculus (`Derivative`/`Integral`), number theory (`GCD`/`LCM`/`Mod`/
`Binomial`/`Factorial`), comparisons and logic (`Equation`/`Gt`/`Gte`/`Lt`/`Lte`/`And`/`Or`/
`Not`/`Implies`), and quantifiers (`ForAll`/`Exists`).

## Rule Categories

```rust
pub enum RuleCategory {
    Simplification,   // Makes expression simpler: x+0 -> x
    Factoring,         // Factors: ab+ac -> a(b+c)
    Expansion,         // Expands: a(b+c) -> ab+ac
    AlgebraicSolving,  // General algebraic manipulation
    EquationSolving,   // Solves: x+3=7 -> x=4
    TrigIdentity,      // Trig: sin^2+cos^2 -> 1
    Derivative,        // Differentiation
    Integral,          // Integration
    Limit,             // Limit evaluation
    Inequality,        // AM-GM, Cauchy-Schwarz, bounds
    Complex,           // Complex number rules
    LogExp,            // Logarithm and exponential rules
    Sequence,          // Sequence and series rules
    NumberTheory,      // Number theory rules
}
```

---

## Common Pitfalls

### 1. A rule that matches too broadly

If `is_applicable` matches a generic shape (any `Add`, any `Gt`) instead of the specific
pattern the rule's math actually requires, and `apply` returns a value that doesn't depend on
the match, the rule can silently hijack unrelated search states. This has happened and was a
real, live bug — see `is_calculus_rewrite` in `crates/mm-verifier/src/lib.rs` for the incident
and the fix. Match the narrowest shape that's actually correct.

### 2. Infinite Loops

If a rule's output can trigger another rule that produces the original input:
```
distribute: a(b+c) -> ab+ac
factor_common: ab+ac -> a(b+c)
```
Prefer simpler outputs and set an appropriate `cost` to avoid this.

### 3. Forgetting `Box`

```rust
// Wrong
Expr::Add(a, b)
// Right
Expr::Add(Box::new(a), Box::new(b))
```

### 4. Not Cloning

```rust
// Wrong
result: base
// Right
result: base.clone()
```

---

## Quality Checklist

Before submitting a PR:

- [ ] `cargo fmt --all -- --check` passes
- [ ] `cargo test --workspace --lib --tests --locked -j 1` passes
- [ ] New rule has a witness that exercises it and a test with an exact expected value
- [ ] `is_applicable` matches the specific shape the rule's math requires, not a broad category
- [ ] Rule ID is unique
- [ ] Any pinned-count test that moved (`rule_census`, `rule_acceptance`,
      `guardrail_reachability`) has its constant updated, with a comment saying what moved

---

## Questions?

Open an issue with the `question` label.

## Code of Conduct

Be respectful. We're all here to learn and build something useful.

---

Thank you for contributing to LEMMA!
