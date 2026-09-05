# Problem set specification (for Codex)

## Why this is needed

The current evaluation set cannot answer the question the paper is built on. With uniform-prior
search, eight of nine families already score 100%:

```
algebra-expand 1/8 | algebra-identity 8/8 | algebra-power 8/8 | arithmetic 8/8
calculus 8/8 | combinatorics 7/7 | equation-solving 8/8 | number-theory 16/16 | trigonometry 8/8
```

Trained policy and uniform priors both score **72/79 strict + 7 POLY, 0 wrong** — byte-identical.
That is not evidence the policy fails; it is evidence the benchmark has **no discriminating
power**. A saturated benchmark cannot measure guidance.

We need problems where uniform-prior search *actually struggles*, so that better guidance has
room to show up.

## The one constraint that matters most

**Difficulty must come from search, not from missing coverage.**

If a problem needs a rule LEMMA does not have, or needs one the verifier rejects, it fails for
*both* arms and measures nothing. Difficulty must come from:

- **Depth** — 3 to 8 correct rewrite steps in sequence.
- **Branching** — many rules applicable at each state, so uniform priors waste budget.
- **Deceptive moves** — locally applicable rewrites that lead away from the goal.

Not from exotic mathematics.

### Rules the verifier REJECTS — do not require these

Problems needing any of these are unsolvable regardless of guidance:

```
algebra::exp_ln              algebra::log_power           algebra::sqrt_product
inequalities::ln_comparison  inequalities::ln_monotonic   number_theory::euler_criterion
number_theory::euler_theorem number_theory::legendre_symbol_compute
number_theory::sqrt_quotient polynomials::diff_nth_power  polynomials::sum_cubes
```

## Format

JSONL, one problem per line. Strings are parsed by `mm_core::parse::Parser`.

```json
{"id":"id-distribute-collect-d4-001","family":"distribute-collect","split":"ID","depth":4,
 "input":"2*(3*x + 4) + 5*(x + 1)","expected":"11*x + 13"}
```

| Field | Meaning |
|---|---|
| `id` | unique, stable, encodes family + depth |
| `family` | generator family name |
| `split` | `ID` or `OOD` (defined below) |
| `depth` | intended minimum number of rewrite steps |
| `input` | expression to simplify/solve |
| `expected` | the correct fully-simplified result |

### Parser syntax available

Numbers `42`, `3.14`, `1/2`; variables `x`, `y`, `theta`; operators `+ - * / ^ %`, `!`, `=`;
parentheses; functions `sin cos tan ln exp sqrt abs floor ceil`, `gcd(a,b)`, `lcm(a,b)`,
`binomial(n,k)`, `diff(expr,var)`, `int(expr,var)`, `sum(var,from,to,body)`,
`prod(var,from,to,body)`.

**Every `input` and `expected` must round-trip through the parser.** Please validate this
programmatically before shipping the file — a problem that does not parse is silently useless.

## The ID / OOD split

The policy network is trained on **only 24 of 572 rules**. This split is what makes the
generalization claim meaningful; please respect it exactly.

### ID — problems requiring ONLY these 24 trained rules

```
algebra:   const_fold, identity_add_zero, identity_mul_one, zero_mul,
           collect_like_terms, distribute, factor_common, difference_of_squares
calculus:  power_rule, constant_rule, sum_rule, product_rule, quotient_rule,
           sin_chain_rule, cos_chain_rule, exp_derivative, ln_derivative
equations: isolate_variable, cancel_addition, cancel_subtraction,
           cancel_multiplication, cancel_division, linear_solve, quadratic_formula
```

### OOD — problems requiring rules with NO training labels

Anything outside that list: trig identities (e.g. Pythagorean), GCD/LCM, modular arithmetic,
binomial/factorial identities, polynomial expansion beyond difference-of-squares.

## Templates already used — do NOT reuse these shapes

Reusing them reintroduces the leakage documented in `experiments/SPLITS.md`. Vary the
**structure**, not just the constants.

**Training templates (`crates/mm-brain/src/data.rs`):**
`a+b`, `a-b`, `a*b`, `a/b`, `2^n`; `x+0`, `0+x`, `x*1`, `1*x`, `x*0`, `0*x`;
`a*(x+y)`, `(x+y)*a`, `a*(x-y)`; `ax+ay`; `x^2-y^2`, `x^2-a^2`; `ax+bx`, `x+x`;
`diff(x^n,x)`, `diff(x,x)`, `diff(c,x)`, `diff(x^n + x^m, x)`, `diff(x+c,x)`,
`diff(x*x^n,x)`, `diff(x*sin(x),x)`, `diff(x/x^n,x)`, `diff(1/x,x)`, `diff(x/(x+1),x)`,
`diff(sin(x),x)`, `diff(cos(x),x)`, `diff(exp(x),x)`, `diff(ln(x),x)`;
`x+a=b`, `a+x=b`, `x-a=b`, `ax=b`, `x/a=b`, `ax+b=c`, `ax-b=c`, `ax^2+bx+c=0`, `x^2=c`

**Current eval templates (`crates/mm-solver/examples/seeded_eval.rs`):**
`(a op b) op c`; `x+0`/`x*1`/`x-0`; `x^a * x^b`; `sin(x)^2 + cos(x)^2`; `diff(x^n,x)`;
`gcd(a,b)`; `a^b % m`; `ax+b=c`; `(x+a)^2`; `binomial(n,k)`

## Difficulty ladder

Do **not** try to hit a single difficulty. Produce a **ladder** at depths **2, 3, 4, 6, 8** so we
can empirically select the band where uniform-prior search lands between roughly 20% and 70%.
That band is where the experiment has power; we will measure and pick, rather than guess.

## Quantities

Per split (ID, OOD), per depth (2, 3, 4, 6, 8), aim for **≥ 20 problems** across **≥ 4 distinct
families**. That is ~200 per split, ~400 total minimum. More is welcome — the statistical plan
uses paired tests over 30 seeds, so larger cells give tighter intervals.

## Deliverables

1. **A deterministic, seeded generator** (Rust preferred, matching the existing `Problem`
   pattern in `seeded_eval.rs`; Python acceptable if it emits the JSONL). Seeded so we can
   regenerate disjoint instances per split without overlap.
2. **A materialized JSONL corpus** at `experiments/corpus/problems.jsonl`.
3. **A validation report** confirming, for every problem:
   - `input` and `expected` both parse;
   - `expected` is genuinely the correct answer (computed independently — please do **not**
     derive it by running LEMMA, which would make the corpus circular);
   - no problem requires a rule from the verifier-rejected list;
   - no `(input, expected)` pair duplicates another, and none reuses a forbidden template.

## What makes this corpus succeed or fail

**Succeeds** if uniform-prior solve rate at some depth lands well below 100% while remaining
above 0% — that is the only condition under which trained-vs-uniform is measurable.

**Fails** if either everything is solved (no headroom, same as today) or nothing is solved
(usually means it needs missing or verifier-rejected rules, which measures coverage, not
guidance).

If in doubt, prefer **deeper chains of simple, well-supported rules** over shallow uses of
exotic ones.
