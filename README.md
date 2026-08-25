<div align="center">

# LEMMA

**Logical Engine for Multi-domain Mathematical Analysis**

A research prototype exploring neural-guided symbolic mathematics in Rust. Inspired by AlphaProof and AlphaZero.

[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg?style=for-the-badge)](https://opensource.org/licenses/MPL-2.0)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg?style=for-the-badge)](https://www.rust-lang.org/)

</div>

---

## What This Project IS and IS NOT

### What LEMMA IS:
- A **research prototype** exploring hybrid neural-symbolic reasoning
- A **proof of concept** for AlphaProof-style mathematical search
- **572 registered transformation rules**, of which a measured 146 actually transform an
  expression and 117 produce a result the verifier accepts (see Evaluation)
- An **MCTS engine** over those rules; it can be guided by a policy network, but no
  trained model ships with the repository
- A **learning project** for anyone interested in symbolic AI

### What LEMMA is NOT:
- **Not a Wolfram Alpha replacement** - we handle basic calculus, not arbitrary math
- **Not production-ready** - this is research code
- **Not a complete CAS** - missing integration, limits, ODEs, and advanced features
- **Not magic** - it applies explicit rules, nothing more

---

## Current Capabilities

### Working Features (Tested)

| Category | Examples | Status |
|----------|----------|--------|
| **Arithmetic** | `(2+3)*(4+5) -> 45` | Working |
| **Identities** | `((x+0)*1)+0 -> x` | Working |
| **Power Rules** | `x^2 * x^3 * x^4 -> x^9` | Working |
| **Basic Derivatives** | `d/dx(x^3) -> 3x^2` | Working |
| **Sum Rule** | `d/dx(x^2 + x^3) -> 2x + 3x^2` | Working |
| **Trig Derivatives** | `d/dx(sin x) -> cos x` | Working |
| **Linear Equations** | `3x + 5 = 17 -> x = 4` | Working |
| **Pythagorean** | `sin^2(x) + cos^2(x) -> 1` | Working |
| **Like Terms** | `2(x+y) + 3(x+y) -> 5(x+y)` | Working |

### New Features (Added Jan 2026)

| Category | Count | Examples |
|----------|-------|----------|
| **Integration** | 9 | `∫x^n dx → x^(n+1)/(n+1)` |
| **Number Theory** | 80+ | Divisibility, GCD, modular arithmetic |
| **Inequalities** | 40+ | AM-GM, Cauchy-Schwarz, Triangle |
| **Combinatorics** | 50+ | Binomial, Pascal, Catalan, generating functions |
| **Polynomials** | 40+ | Vieta's, symmetric polys, factoring |

---

## Architecture

```
+------------------------------------------------------------------+
|                      Neural Policy Network                       |
|              (Transformer, suggests which rule to try)           |
+------------------------------+-----------------------------------+
                               |
                               v
+-----------+     +-----------------------------+     +-----------+
|  Problem  |---->|      MCTS Search Engine     |---->| Solution  |
|   Expr    |     |   (AlphaZero-style UCB)     |     | + Proof   |
+-----------+     +--------------+--------------+     +-----------+
                                 |
                                 v
                  +-----------------------------+
                  |       Rule Library          |
                  |     (572 registered rules)  |
                  +--------------+--------------+
                                 |
                                 v
                  +-----------------------------+
                  |         Verifier            |
                  |  (Numerical + Symbolic)     |
                  +-----------------------------+
```

### Approach

Unlike LLMs that predict text statistically, LEMMA:
1. **Only applies rules from an explicit registry** - it cannot invent a transformation
2. **Records a trace** - every result carries the steps that produced it, and a
   `VerificationStatus` saying what was actually checked (see below)
3. **Can be guided by a policy network** - with no trained model present the search uses
   uniform priors and says so, rather than reading randomly initialised weights

Verification here means equivalence checking (canonical form, with numeric sampling as a
fallback), not machine-checked proof. There is no SMT backend; `VerificationLevel::Formal`
reports that it is unsupported instead of returning a verdict.

---

## Crate Structure

| Crate | Purpose | Lines of Code |
|-------|---------|---------------|
| `mm-core` | Expression AST, parsing, evaluation | ~3,700 |
| `mm-rules` | 550+ transformation rules | ~22,700 |
| `mm-verifier` | Numerical and symbolic verification | ~600 |
| `mm-search` | Beam search, Neural MCTS | ~1,800 |
| `mm-brain` | Transformer network (Candle) | ~2,400 |
| `mm-solver` | Unified API | ~1,400 |

---

## Quick Start

### Requirements
- Rust 1.75+
- ~500MB disk space for dependencies

### Build and Test

```bash
git clone https://github.com/Pushp-Kharat1/LEMMA.git
cd LEMMA

# Build everything
cargo build --release

# Run the correctness evaluation (fail-closed: a regression fails the run)
cargo test -p mm-solver --test evaluation -- --nocapture

# Run everything
cargo test --workspace --lib --tests
```

### Train the Neural Network

```bash
# Generates ~17k synthetic examples, trains for 50 epochs
cargo run --release --example train_network
```

---

## Terminal Workbench

```bash
cargo run -p mm-tui
```

An interactive front end for the three operations the solver actually supports: simplify an
expression, differentiate it with respect to a variable, or check a candidate value against an
equation. It takes formal LEMMA syntax, not prose.

The point of it is to make verification legible. Every result carries its
`VerificationStatus` as a labelled badge with the verifier's own reason, and the trace lists
each recorded step with its before/after expressions, its stable `module::name` rule identity,
the rule's justification, and the evidence that step rests on — so a step accepted by rule
replay is visibly different from one checked symbolically. Only `CHECKED` is described as
verified.

```
┌ Result ─────────────────────────┐┌ Trace · 2 steps ─────────────────────────┐
│x                                ││> 01 algebra::identity_add_zero SYMBOLIC  │
│                                 ││  02 algebra::identity_mul_one SYMBOLIC   │
│ ++ CHECKED                      ││                                          │
│Trace replays from the input to  ││                                          │
│the result and every step was    ││                                          │
│independently checked.           ││                                          │
│2 steps · 14 ms                  ││                                          │
└─────────────────────────────────┘└──────────────────────────────────────────┘
```

Needs an 80x24 terminal; below that it shows a size notice rather than a broken layout. Press
`?` for keys. Equation solving and natural-language input are not offered, because the APIs
behind them report themselves unimplemented.

---

## Usage Example

```rust
use mm_core::{Expr, SymbolTable};
use mm_rules::rule::standard_rules;
use mm_search::{MCTSConfig, NeuralMCTS};
use mm_verifier::Verifier;

fn main() {
    let mut symbols = SymbolTable::new();
    let x = symbols.intern("x");
    
    // Setup the solver
    let rules = standard_rules();
    let verifier = Verifier::new();
    let mcts = NeuralMCTS::new(rules, verifier);
    
    // Solve: x + 3 = 7
    let equation = Expr::Equation {
        lhs: Box::new(Expr::Add(Box::new(Expr::Var(x)), Box::new(Expr::int(3)))),
        rhs: Box::new(Expr::int(7)),
    };

    let solution = mcts.simplify(equation);

    // solution.result  -> x = 4
    // solution.steps   -> the transitions that produced it, each with its evidence
    // solution.status  -> what is actually known about the result. Only
    //                     `VerificationStatus::Checked` means the trace replays from the
    //                     exact input and every step was independently checked.
    println!("{:?} ({})", solution.result, solution.status);
}
```

Not every problem in this shape is solved. `3x + 5 = 17` is one the search does not currently
finish; the evaluation suite lists such cases explicitly rather than omitting them.

## Evaluation

## Evaluation Notice (Research Integrity Update){Edited on 6 Feb 2026}
An internal audit identified issues in earlier evaluation scripts and neural-rule integration that invalidated previous competitive benchmark claims.
Those historical results are deprecated.
See Issue #8 for technical details and remediation work.

The `benchmark`, `benchmark_advanced` and `stress_test` examples that produced the previously
quoted 20/21, 10/10 and 10/10 figures have been removed. They printed pass counts but exited
zero whether or not a case failed, accepted output *shapes* instead of values (one case
accepted every possible output), and gated on a `verified` flag that several code paths set
unconditionally. Those numbers are withdrawn and are not replaced here.

Correctness is now measured by a fail-closed test:

```bash
cargo test -p mm-solver --test evaluation -- --nocapture
```

Every case states an exact expected expression, cases the engine does not solve are listed by
name in the suite itself, and a solved case must also produce a trace that replays from the
input to the reported result. The run prints the rule count, the action-vocabulary digest and
the model provenance it was produced under. Read the numbers from a run, not from this file.

---

## Design Philosophy

### Why Not Just Use an LLM?

LLMs are amazing at many tasks, but they can:
- Produce plausible-looking but incorrect derivations
- Skip steps or make sign errors
- Not explain why a transformation is valid

LEMMA trades generality for **reliability**:
- Every step is a provable transformation
- The verifier catches errors
- Complete proof traces for debugging

### Why Rust?

1. **Performance** - MCTS explores thousands of nodes; speed matters
2. **Memory Safety** - No GC pauses during search
3. **Type System** - Expression trees are naturally typed
4. **Ecosystem** - Candle for neural networks, excellent tooling

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

**Quick version:**
1. Fork the repo
2. Add a rule to `mm-rules/src/algebra.rs`
3. Add a test case
4. Submit a PR

We especially welcome:
- New mathematical rules
- Bug reports with reproducible examples
- Documentation improvements
- Benchmark problems that fail

---

## License

Mozilla Public License 2.0 - see [LICENSE](LICENSE)

---

## Acknowledgments

- [AlphaZero](https://www.nature.com/articles/nature24270) - MCTS + Neural guidance
- [AlphaProof](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/) - Inspiration for math reasoning
- [Candle](https://github.com/huggingface/candle) - Rust ML framework
- [SimSIMD](https://github.com/ashvardanian/SimSIMD) / [USearch](https://github.com/unum-cloud/usearch) - Ash Vardanian's ecosystem

---

## Insights From the Creator :-
### LEMMA is a neuro-symbolic reasoning architecture prototype that *explores*:

1) rule-based symbolic transformation
2) domain-aware rule gating
3) credit-constrained search control
4) reinforcement-style meta-reward feedback
5) hybrid symbolic + learned policy guidance

It is *NOT*:
1) a CAS
2) an LLM
3) a IMO/JEE/MATH paper solver
4) a theorem prover 
5) a Magic Box
.... yet

It’s a research platform/Prototype for studying/Understanding how symbolic systems can be guided, constrained, and optimized Through Neural Guidance.

---

## Contact

- **Author**: Pushp Kharat
- **Email**: kharatpushp16@outlook.com
- **GitHub**: [@Pushp-Kharat1](https://github.com/Pushp-Kharat1)

---

LEMMA is a research project. Use it to learn, experiment, and contribute - not as your only source of mathematical truth.

bUY ME A COFFEE : https://buymeacoffee.com/kharatpushg
