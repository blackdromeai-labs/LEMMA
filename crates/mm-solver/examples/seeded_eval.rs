//! Seeded evaluation: a reproducible, randomly generated problem set solved by `NeuralMCTS`,
//! checked against ground truth that never calls into LEMMA's own equality machinery, at three
//! decreasing tiers of evidence strength.
//!
//! Every problem's expected answer is computed here by a method independent of the solver
//! under test -- direct Rust arithmetic for the arithmetic families, an `Expr` built by hand
//! from the rule being tested for the algebraic ones. What a "pass" requires depends on the
//! tier, since not every check here is equally strong evidence:
//!
//! 1. **PASS** (strict): `solution.result.canonicalize() == expected.canonicalize()`. The
//!    strongest tier -- exact structural agreement after LEMMA's own normalization.
//! 2. **POLY** (independently confirmed): only reached when tier 1 fails. Both `result` and
//!    `expected` are decomposed into a map from power of the problem's variable to exact
//!    rational coefficient by [`poly_coeffs`] -- a small routine written fresh in this file,
//!    using neither `canonicalize()` nor `approx_equals()`, that recurses over `Add`/`Sub`/
//!    `Mul`/`Neg`/integer-`Pow`/`Const`/`Var` and bails out (`None`) on anything else. If both
//!    sides decompose and the coefficient maps are exactly equal, the two expressions are the
//!    same polynomial regardless of tree shape -- confirmed by independent symbolic
//!    recomputation, not sampling.
//! 3. **SAMPLE** (sampling-equivalent only, NOT independently confirmed): only reached when
//!    both tier 1 and tier 2 fail or do not apply (e.g. an expression `poly_coeffs` cannot
//!    decompose). Falls back to `solution.result.approx_equals(&expected, ...)`, LEMMA's own
//!    numeric-sampling routine. This is deliberately the weakest, last-resort tier: it uses the
//!    system under test to check itself, and is reported as such rather than folded into the
//!    stronger tiers above.
//!
//! Every tier also requires `assess_trace` to confirm the recorded steps replay from the exact
//! input to the exact output through checked or heuristic evidence -- landing on the right
//! answer via a trace that does not actually connect input to output is not a pass at any tier.
//!
//! This example is fail-closed: it exits with a non-zero status if any problem's answer is
//! wrong, its trace does not replay, OR it only reached the SAMPLE tier. SAMPLE is explicitly
//! not independently confirmed evidence -- accepting it with a zero exit code would make
//! "fail-closed" only as strict as the weakest tier available. POLY does not fail the process:
//! coefficient equality is established independently of both `canonicalize()` and
//! `approx_equals()`, so it is real evidence of correctness, just not exact structural
//! agreement.
//!
//! Usage: cargo run --release --example seeded_eval -p mm-solver [seed] [model.safetensors]
//! Rerunning with the printed seed reproduces the same problem set exactly.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use candle_core::Device;
use mm_brain::PolicyNetwork;
use mm_core::{Expr, Rational, Symbol, SymbolTable};
use mm_rules::{standard_rules, ActionVocabulary};
use mm_search::{assess_trace, MCTSConfig, NeuralMCTS};
use mm_verifier::Verifier;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Decompose `expr` into a map from power of `var` to exact rational coefficient, or `None` if
/// `expr` contains a node this routine does not handle (division, roots, transcendental
/// functions, derivatives, equations, a second free variable, ...). Written independently of
/// `Expr::canonicalize` and `Expr::approx_equals` so that agreement between two decompositions
/// is evidence neither of those two routines could manufacture by sharing a bug.
fn poly_coeffs(expr: &Expr, var: Symbol) -> Option<BTreeMap<u32, Rational>> {
    match expr {
        Expr::Const(c) => Some(singleton_map(0, *c)),
        Expr::Var(v) if *v == var => Some(singleton_map(1, Rational::from_integer(1))),
        Expr::Var(_) => None,
        Expr::Neg(a) => {
            let ma = poly_coeffs(a, var)?;
            Some(negate_map(&ma))
        }
        Expr::Add(a, b) => {
            let ma = poly_coeffs(a, var)?;
            let mb = poly_coeffs(b, var)?;
            Some(add_maps(&ma, &mb))
        }
        Expr::Sub(a, b) => {
            let ma = poly_coeffs(a, var)?;
            let mb = poly_coeffs(b, var)?;
            Some(add_maps(&ma, &negate_map(&mb)))
        }
        Expr::Mul(a, b) => {
            let ma = poly_coeffs(a, var)?;
            let mb = poly_coeffs(b, var)?;
            Some(mul_maps(&ma, &mb))
        }
        Expr::Pow(base, exp) => {
            let Expr::Const(e) = exp.as_ref() else {
                return None;
            };
            if e.denom() != 1 || e.numer() < 0 {
                return None; // only non-negative integer exponents are a polynomial power
            }
            let mb = poly_coeffs(base, var)?;
            let mut result = singleton_map(0, Rational::from_integer(1));
            for _ in 0..e.numer() {
                result = mul_maps(&result, &mb);
            }
            Some(result)
        }
        _ => None,
    }
}

fn singleton_map(power: u32, coeff: Rational) -> BTreeMap<u32, Rational> {
    let mut m = BTreeMap::new();
    if !coeff.is_zero() {
        m.insert(power, coeff);
    }
    m
}

fn negate_map(m: &BTreeMap<u32, Rational>) -> BTreeMap<u32, Rational> {
    m.iter().map(|(&k, &v)| (k, -v)).collect()
}

fn add_maps(a: &BTreeMap<u32, Rational>, b: &BTreeMap<u32, Rational>) -> BTreeMap<u32, Rational> {
    let mut result = a.clone();
    for (&k, &v) in b {
        let entry = result.entry(k).or_insert_with(|| Rational::from_integer(0));
        *entry = *entry + v;
    }
    result.retain(|_, v| !v.is_zero());
    result
}

fn mul_maps(a: &BTreeMap<u32, Rational>, b: &BTreeMap<u32, Rational>) -> BTreeMap<u32, Rational> {
    let mut result: BTreeMap<u32, Rational> = BTreeMap::new();
    for (&ka, &va) in a {
        for (&kb, &vb) in b {
            let entry = result
                .entry(ka + kb)
                .or_insert_with(|| Rational::from_integer(0));
            *entry = *entry + va * vb;
        }
    }
    result.retain(|_, v| !v.is_zero());
    result
}

/// Whether `a` and `b` are the same polynomial in `var`, decided by independent coefficient
/// extraction. `None` (not `Some(false)`) means at least one side could not be decomposed --
/// callers must not treat that as "not equal", only as "this check does not apply here".
fn poly_equal(a: &Expr, b: &Expr, var: Symbol) -> Option<bool> {
    let ca = poly_coeffs(a, var)?;
    let cb = poly_coeffs(b, var)?;
    Some(ca == cb)
}

struct Problem {
    id: String,
    category: &'static str,
    input: Expr,
    expected: Expr,
}

fn gcd_i64(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.abs()
}

fn mod_pow(mut base: i64, mut exp: i64, m: i64) -> i64 {
    let mut result = 1i64;
    base = base.rem_euclid(m);
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * base).rem_euclid(m);
        }
        exp >>= 1;
        base = (base * base).rem_euclid(m);
    }
    result
}

fn binomial_i64(n: i64, k: i64) -> i64 {
    if k < 0 || k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut result: i64 = 1;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

fn generate_problems(rng: &mut ChaCha8Rng, x: Symbol, y: Symbol, count: usize) -> Vec<Problem> {
    let mut problems = Vec::new();
    let mut n = 0usize;

    while problems.len() < count {
        n += 1;
        let category_pick = n % 9;

        match category_pick {
            // Arithmetic: (a op b) op c for small integers, ground truth from i64 math.
            0 => {
                let a = rng.gen_range(2..12);
                let b = rng.gen_range(2..12);
                let c = rng.gen_range(2..12);
                let (input, value) = match rng.gen_range(0..3) {
                    0 => (
                        Expr::Mul(
                            Box::new(Expr::Add(Box::new(Expr::int(a)), Box::new(Expr::int(b)))),
                            Box::new(Expr::int(c)),
                        ),
                        (a + b) * c,
                    ),
                    1 => (
                        Expr::Sub(
                            Box::new(Expr::Mul(Box::new(Expr::int(a)), Box::new(Expr::int(b)))),
                            Box::new(Expr::int(c)),
                        ),
                        a * b - c,
                    ),
                    _ => (
                        Expr::Mul(
                            Box::new(Expr::Mul(Box::new(Expr::int(a)), Box::new(Expr::int(b)))),
                            Box::new(Expr::int(c)),
                        ),
                        a * b * c,
                    ),
                };
                problems.push(Problem {
                    id: format!("arith-{a}-{b}-{c}"),
                    category: "arithmetic",
                    input,
                    expected: Expr::int(value),
                });
            }
            // Identity noise: wrap a base expression in +0 / *1 / -0; base is unchanged.
            1 => {
                let base = if rng.gen_bool(0.5) {
                    Expr::Var(x)
                } else {
                    Expr::Add(Box::new(Expr::Var(x)), Box::new(Expr::Var(y)))
                };
                let wrapped = match rng.gen_range(0..3) {
                    0 => Expr::Add(Box::new(base.clone()), Box::new(Expr::int(0))),
                    1 => Expr::Mul(Box::new(base.clone()), Box::new(Expr::int(1))),
                    _ => Expr::Sub(Box::new(base.clone()), Box::new(Expr::int(0))),
                };
                problems.push(Problem {
                    id: format!("identity-{n}"),
                    category: "algebra-identity",
                    input: wrapped,
                    expected: base,
                });
            }
            // Power product: x^a * x^b -> x^(a+b).
            2 => {
                let a = rng.gen_range(1..6);
                let b = rng.gen_range(1..6);
                let input = Expr::Mul(
                    Box::new(Expr::Pow(Box::new(Expr::Var(x)), Box::new(Expr::int(a)))),
                    Box::new(Expr::Pow(Box::new(Expr::Var(x)), Box::new(Expr::int(b)))),
                );
                let expected = Expr::Pow(Box::new(Expr::Var(x)), Box::new(Expr::int(a + b)));
                problems.push(Problem {
                    id: format!("power-product-x^{a}-x^{b}"),
                    category: "algebra-power",
                    input,
                    expected,
                });
            }
            // Pythagorean identity, on whichever variable this round picked.
            3 => {
                let var = if rng.gen_bool(0.5) { x } else { y };
                let input = Expr::Add(
                    Box::new(Expr::Pow(
                        Box::new(Expr::Sin(Box::new(Expr::Var(var)))),
                        Box::new(Expr::int(2)),
                    )),
                    Box::new(Expr::Pow(
                        Box::new(Expr::Cos(Box::new(Expr::Var(var)))),
                        Box::new(Expr::int(2)),
                    )),
                );
                problems.push(Problem {
                    id: format!("pythagorean-{n}"),
                    category: "trigonometry",
                    input,
                    expected: Expr::int(1),
                });
            }
            // Derivative power rule: d/dx(x^n) -> n * x^(n-1).
            4 => {
                let deg = rng.gen_range(2..8);
                let input = Expr::Derivative {
                    expr: Box::new(Expr::Pow(Box::new(Expr::Var(x)), Box::new(Expr::int(deg)))),
                    var: x,
                };
                let expected = Expr::Mul(
                    Box::new(Expr::int(deg)),
                    Box::new(Expr::Pow(
                        Box::new(Expr::Var(x)),
                        Box::new(Expr::int(deg - 1)),
                    )),
                );
                problems.push(Problem {
                    id: format!("derivative-x^{deg}"),
                    category: "calculus",
                    input,
                    expected,
                });
            }
            // GCD of two random integers, ground truth from an independent Euclidean gcd.
            5 => {
                let a = rng.gen_range(4..500);
                let b = rng.gen_range(4..500);
                problems.push(Problem {
                    id: format!("gcd-{a}-{b}"),
                    category: "number-theory",
                    input: Expr::GCD(Box::new(Expr::int(a)), Box::new(Expr::int(b))),
                    expected: Expr::int(gcd_i64(a, b)),
                });
            }
            // a^b mod m, ground truth from an independent modular-exponentiation routine.
            6 => {
                let base = rng.gen_range(2..20);
                let exp = rng.gen_range(2..10);
                let m = rng.gen_range(3..97);
                problems.push(Problem {
                    id: format!("modpow-{base}-{exp}-{m}"),
                    category: "number-theory",
                    input: Expr::Mod(
                        Box::new(Expr::Pow(
                            Box::new(Expr::int(base)),
                            Box::new(Expr::int(exp)),
                        )),
                        Box::new(Expr::int(m)),
                    ),
                    expected: Expr::int(mod_pow(base, exp, m)),
                });
            }
            // Linear equation a*x + b = c, ground truth x = (c-b)/a chosen to divide evenly.
            // Harder than the arithmetic family: needs equation-aware rewriting
            // (`cancel_addition`/`cancel_multiplication`) rather than plain constant folding.
            7 => {
                let a = rng.gen_range(2..9);
                let root = rng.gen_range(-9..9);
                let b = rng.gen_range(-9..9);
                let c = a * root + b;
                let input = Expr::Equation {
                    lhs: Box::new(Expr::Add(
                        Box::new(Expr::Mul(Box::new(Expr::int(a)), Box::new(Expr::Var(x)))),
                        Box::new(Expr::int(b)),
                    )),
                    rhs: Box::new(Expr::int(c)),
                };
                let expected = Expr::Equation {
                    lhs: Box::new(Expr::Var(x)),
                    rhs: Box::new(Expr::int(root)),
                };
                problems.push(Problem {
                    id: format!("linear-eq-{a}x+{b}={c}"),
                    category: "equation-solving",
                    input,
                    expected,
                });
            }
            // Binomial square expansion: (x + a)^2 -> x^2 + 2ax + a^2. Harder than the power-
            // product family: needs full expansion and like-term collection, not one rule.
            _ => {
                let a = rng.gen_range(1..9);
                let input = Expr::Pow(
                    Box::new(Expr::Add(Box::new(Expr::Var(x)), Box::new(Expr::int(a)))),
                    Box::new(Expr::int(2)),
                );
                let expected = Expr::Add(
                    Box::new(Expr::Add(
                        Box::new(Expr::Pow(Box::new(Expr::Var(x)), Box::new(Expr::int(2)))),
                        Box::new(Expr::Mul(
                            Box::new(Expr::int(2 * a)),
                            Box::new(Expr::Var(x)),
                        )),
                    )),
                    Box::new(Expr::int(a * a)),
                );
                problems.push(Problem {
                    id: format!("binomial-square-(x+{a})^2"),
                    category: "algebra-expand",
                    input,
                    expected,
                });
            }
        }
    }

    // A handful of combinatorics cases mixed in, ground truth from an independent binomial
    // computation (this family is small on purpose: LEMMA's binomial rules are narrow).
    for _ in 0..(count / 10).max(1) {
        let n_val = rng.gen_range(4..15);
        let k_val = rng.gen_range(1..n_val);
        problems.push(Problem {
            id: format!("binomial-{n_val}-{k_val}"),
            category: "combinatorics",
            input: Expr::Binomial(Box::new(Expr::int(n_val)), Box::new(Expr::int(k_val))),
            expected: Expr::int(binomial_i64(n_val, k_val)),
        });
    }

    problems
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| rand::thread_rng().gen());
    let model_path = args.get(2);

    println!(
        "seed = {seed}  (rerun with `cargo run --release --example seeded_eval -p mm-solver -- {seed}` to reproduce this exact set)\n"
    );

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut symbols = SymbolTable::new();
    let x = symbols.intern("x");
    let y = symbols.intern("y");

    let problems = generate_problems(&mut rng, x, y, 72);

    let rules = standard_rules();
    let verifier = Verifier::new();
    let config = MCTSConfig {
        simulations: 150,
        ..Default::default()
    };
    let mut mcts = NeuralMCTS::with_config(rules, verifier, config);
    if let Some(model_path) = model_path {
        let vocabulary = ActionVocabulary::standard();
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        let policy = match PolicyNetwork::load(Path::new(model_path), vocabulary, device) {
            Ok(policy) => policy,
            Err(error) => {
                eprintln!("failed to load trained policy {model_path}: {error}");
                return ExitCode::FAILURE;
            }
        };
        mcts = match mcts.with_policy(policy) {
            Ok(mcts) => mcts,
            Err(error) => {
                eprintln!("trained policy vocabulary mismatch: {error}");
                return ExitCode::FAILURE;
            }
        };
    }
    println!("policy = {}\n", mcts.provenance());

    let mut by_category: std::collections::BTreeMap<&str, (usize, usize)> =
        std::collections::BTreeMap::new();
    let mut failures: Vec<String> = Vec::new();
    let mut total_pass = 0usize;
    let mut total_poly_verified = 0usize;
    let mut total_sample_only = 0usize;
    let run_start = Instant::now();

    for (i, p) in problems.iter().enumerate() {
        let start = Instant::now();
        let solution = mcts.simplify(p.input.clone());
        let elapsed = start.elapsed();

        let trace_status = assess_trace(&solution.problem, &solution.result, &solution.steps);
        let replays = trace_status.replays();

        // Tier 1 (strict): exact structural agreement after LEMMA's own canonicalization.
        let canonical_match = solution.result.canonicalize() == p.expected.canonicalize();

        // Tier 2 (independently confirmed): only tried when tier 1 fails. `poly_equal` uses
        // neither `canonicalize()` nor `approx_equals()` -- it recomputes coefficients from
        // scratch, so agreement here is not something a shared bug in those two routines could
        // manufacture. `None` means the check doesn't apply (a non-polynomial node appeared),
        // not that the expressions disagree.
        let poly_verdict = if canonical_match {
            None
        } else {
            poly_equal(&solution.result, &p.expected, x)
        };
        let poly_confirmed = poly_verdict == Some(true);

        // Tier 3 (sampling-equivalent only, NOT independently confirmed): last resort, only
        // reached when tiers 1 and 2 both fail or don't apply. Uses the system under test
        // (`approx_equals`) to check itself, so it is reported as the weakest tier rather than
        // folded into "PASS".
        let sample_equivalent = !canonical_match
            && !poly_confirmed
            && solution.result.approx_equals(&p.expected, 20, 1e-9);

        let success = canonical_match && replays;
        let poly_pass = !success && poly_confirmed && replays;
        let sample_pass = !success && !poly_pass && sample_equivalent && replays;

        let slot = by_category.entry(p.category).or_insert((0, 0));
        slot.0 += 1;
        if success {
            slot.1 += 1;
            total_pass += 1;
        } else if poly_pass {
            total_poly_verified += 1;
        } else if sample_pass {
            total_sample_only += 1;
        }

        let flag = if success {
            "PASS"
        } else if poly_pass {
            "POLY" // not canonically identical, but independently confirmed as the same polynomial
        } else if sample_pass {
            "SAMPLE" // sampling-equivalent only -- NOT independently confirmed, weakest tier
        } else {
            "FAIL"
        };

        println!(
            "[{:>2}/{}] {:<26} {}  got {:?}  expected {:?}  [{}]  {:.1}ms",
            i + 1,
            problems.len(),
            p.id,
            flag,
            solution.result,
            p.expected,
            solution.status.label(),
            elapsed.as_secs_f64() * 1000.0,
        );

        if !success && !poly_pass && !sample_pass {
            let value_ok = poly_confirmed || sample_equivalent;
            let why = if !value_ok && !replays {
                format!("wrong answer AND trace does not replay ({trace_status})")
            } else if !value_ok {
                "wrong answer (checked canonical form, independent polynomial coefficients \
                 where applicable, and numeric sampling as a last resort)"
                    .to_string()
            } else {
                format!("value checks out but trace does not replay ({trace_status})")
            };
            failures.push(format!("{}: {}", p.id, why));
        }
    }

    let total_elapsed = run_start.elapsed();
    let genuinely_wrong = problems.len() - total_pass - total_poly_verified - total_sample_only;

    println!("\n=== Results by category ===");
    for (cat, (total, pass)) in &by_category {
        println!(
            "  {cat:<18} {pass:>3}/{total:<3} ({:.0}%)",
            *pass as f64 / *total as f64 * 100.0
        );
    }

    println!(
        "\n=== Overall: {}/{} passed strictly ({:.1}%); {} more independently confirmed via \
         polynomial coefficients (not canonically identical); {} more sampling-equivalent only \
         (NOT independently confirmed -- weakest tier); {} genuinely wrong or unreplayed. Ran \
         in {:.2}s ===",
        total_pass,
        problems.len(),
        total_pass as f64 / problems.len() as f64 * 100.0,
        total_poly_verified,
        total_sample_only,
        genuinely_wrong,
        total_elapsed.as_secs_f64()
    );

    if !failures.is_empty() {
        println!("\nFailures:");
        for f in &failures {
            println!("  - {f}");
        }
    }

    // Fail-closed on anything short of independently-confirmed correctness: a wrong answer, a
    // trace that does not replay, OR a case that only reached the SAMPLE tier. SAMPLE is
    // explicitly not independently confirmed -- it is `approx_equals`, the system under test
    // checking itself -- so accepting it with a zero exit code would make "fail-closed" only
    // as strict as the weakest evidence tier. POLY remains non-fatal: coefficient equality is
    // established independently of both `canonicalize()` and `approx_equals()`, so it is real
    // evidence, just not exact structural agreement.
    if genuinely_wrong > 0 || total_sample_only > 0 {
        if genuinely_wrong > 0 {
            eprintln!(
                "\n{genuinely_wrong} case(s) were genuinely wrong or unreplayed -- see Failures above."
            );
        }
        if total_sample_only > 0 {
            eprintln!(
                "\n{total_sample_only} case(s) only reached the SAMPLE tier (sampling-equivalent \
                 only, NOT independently confirmed) -- treated as a failure by this fail-closed \
                 harness even though a value was not flagged as wrong."
            );
        }
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
