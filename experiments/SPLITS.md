# Dataset splits and leakage controls

## Why the split is by rule family, not by instance

Every problem in the evaluation corpora is template-generated. Splitting by *instance* would
put `3x + 5 = 11` in train and `7x + 2 = 23` in test — different constants, identical template,
identical target rule. That is not a held-out set; it measures memorization of a constant
range. The split is therefore by **rule family**: which target rule the family's problems
require.

## What the policy is actually trained on

`crates/mm-brain/src/data.rs` writes labels for **24 rules out of 572 registered** (~4%). Its
own module documentation states this: *"The remaining actions in the vocabulary get no synthetic
examples at all; this data is not coverage of the rule corpus."*

Trained rules, by module:

- **algebra** — `const_fold`, `identity_add_zero`, `identity_mul_one`, `zero_mul`,
  `collect_like_terms`, `distribute`, `factor_common`, `difference_of_squares`
- **calculus** — `power_rule`, `constant_rule`, `sum_rule`, `product_rule`, `quotient_rule`,
  `sin_chain_rule`, `cos_chain_rule`, `exp_derivative`, `ln_derivative`
- **equations** — `isolate_variable`, `cancel_addition`, `cancel_subtraction`,
  `cancel_multiplication`, `cancel_division`, `linear_solve`, `quadratic_formula`
- plus the reserved `no_op` terminal class

## The leakage that would have happened

The evaluation harness (`crates/mm-solver/examples/seeded_eval.rs`) generates nine families.
Four of them are the *same templates* the trainer emits:

| Eval family | Eval template | Training generator | Overlap |
|---|---|---|---|
| arithmetic | `(a op b) op c` | `generate_constant_folding` | yes — `const_fold` |
| algebra-identity | `x+0`, `x*1`, `x-0` | `generate_identity_rules` | yes — near-identical |
| calculus | `d/dx(x^n)` | `generate_power_rule` | yes — near-identical |
| equation-solving | `ax + b = c` | `generate_linear_equations` | yes — identical template |

Training on `data.rs` and evaluating on `seeded_eval` unsplit would report a trained-vs-uniform
gain that is substantially template memorization.

## The splits

| Split | Families | Purpose |
|---|---|---|
| **Train** | The 24 labeled rules above | Fit policy + value heads |
| **Val** | Train families, disjoint generator seed | Early stopping only; never reported |
| **Test-ID** | arithmetic, algebra-identity, calculus (power rule), equation-solving | In-distribution gain |
| **Test-OOD** | algebra-power (`x^a·x^b`), trigonometry (Pythagorean identity), number-theory (GCD), number-theory (modpow), algebra-expand (binomial square), combinatorics | Generalization to rules with **zero** training labels |

Test-OOD families were checked individually against the trained-rule list above: none of them
has a training label for the rule its problems require. The Pythagorean identity is OOD even
though *trig derivatives* are trained — `trig::pythagorean_identity` is a different rule from
`calculus::sin_chain_rule`.

## Controls

1. **Disjoint generator seeds** per split; seeds recorded in each run's output.
2. **Collision assertion** — before training, assert no exact encoded-token-sequence appears in
   both train and any test split. Fail the run rather than warn.
3. **Frozen manifests** — each split written to disk and hashed; the hash is recorded alongside
   every result produced from it.
4. **ID and OOD reported separately, never pooled.** A single headline number over both would
   hide exactly the effect this split exists to expose.
5. **Sampling seed pinned** — `mm_core::sampling::seed_sampling_rng` makes the verifier's
   numeric checks reproducible, so trained and uniform arms see identical sample points on
   identical problems and can be compared pairwise.

## Pre-registered expectations

- Gain on **Test-ID** is expected: the policy has labels for those rules.
- Little or no gain on **Test-OOD** is expected: the policy has never seen a label for those
  rules, and its priors there come from an untrained region of the output head.
- **A large OOD gain should be treated as suspected leakage**, not as a result, until the
  collision assertion and the family-to-rule mapping have been re-checked.
