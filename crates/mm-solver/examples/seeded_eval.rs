//! Seeded evaluation: a reproducible, randomly generated problem set solved by `NeuralMCTS`
//! and checked two independent ways.
//!
//! Every problem's expected answer is computed here, by a method that never calls into
//! LEMMA -- direct Rust arithmetic for the arithmetic families, an `Expr` built by hand from
//! the rule being tested for the algebraic ones. That is what makes this a check and not a
//! demo: a "pass" requires the returned expression to canonicalize equal to that
//! independently-computed answer, *and* `assess_trace` to confirm the recorded steps actually
//! replay from the exact input to the exact output through checked or heuristic evidence, not
//! just that some path happened to land on the right number.
//!
//! Usage: cargo run --release --example seeded_eval -p mm-solver [seed]
//! Rerunning with the printed seed reproduces the same problem set exactly.

use mm_core::{Expr, Symbol, SymbolTable};
use mm_rules::standard_rules;
use mm_search::{assess_trace, MCTSConfig, NeuralMCTS};
use mm_verifier::Verifier;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

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

fn main() {
    let seed: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| rand::thread_rng().gen());

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
    let mcts = NeuralMCTS::with_config(rules, verifier, config);

    let mut by_category: std::collections::BTreeMap<&str, (usize, usize)> =
        std::collections::BTreeMap::new();
    let mut failures: Vec<String> = Vec::new();
    let mut total_pass = 0usize;
    let mut total_numerically_equivalent = 0usize;
    let run_start = Instant::now();

    for (i, p) in problems.iter().enumerate() {
        let start = Instant::now();
        let solution = mcts.simplify(p.input.clone());
        let elapsed = start.elapsed();

        let trace_status = assess_trace(&solution.problem, &solution.result, &solution.steps);
        let replays = trace_status.replays();
        let canonical_match = solution.result.canonicalize() == p.expected.canonicalize();
        // A second, independent check for anything the strict canonical comparison rejects:
        // numeric sampling at several random points (`approx_equals`, fixed earlier this
        // session to return false rather than vacuously true when nothing evaluates). This
        // tells apart "mathematically wrong" from "correct but canonicalize() didn't collect
        // it the same way" -- e.g. `2*(x*4)` vs `8*x` are the same function but different
        // trees, and only the second is on canonicalize() rather than on the search.
        let numerically_equivalent =
            canonical_match || solution.result.approx_equals(&p.expected, 20, 1e-9);
        let answer_matches = canonical_match;
        let success = answer_matches && replays;

        let slot = by_category.entry(p.category).or_insert((0, 0));
        slot.0 += 1;
        if success {
            slot.1 += 1;
            total_pass += 1;
        } else if numerically_equivalent && replays {
            total_numerically_equivalent += 1;
        }

        let flag = if success {
            "PASS"
        } else if numerically_equivalent && replays {
            "PASS*" // equivalent value, not canonically identical -- see note below
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

        if !success {
            let why = if !numerically_equivalent && !replays {
                format!("wrong answer AND trace does not replay ({trace_status})")
            } else if !numerically_equivalent {
                "wrong answer (checked both canonical form and numeric sampling)".to_string()
            } else if !replays {
                format!("trace does not replay ({trace_status}), though the value is right")
            } else {
                "numerically equivalent but canonicalize() does not treat it as identical \
                 (a gap in canonicalize, not in what the search found)"
                    .to_string()
            };
            failures.push(format!("{}: {}", p.id, why));
        }
    }

    let total_elapsed = run_start.elapsed();

    println!("\n=== Results by category ===");
    for (cat, (total, pass)) in &by_category {
        println!(
            "  {cat:<18} {pass:>3}/{total:<3} ({:.0}%)",
            *pass as f64 / *total as f64 * 100.0
        );
    }

    println!(
        "\n=== Overall: {}/{} passed strictly ({:.1}%); {} more numerically equivalent but not \
         canonically identical; {} genuinely wrong or unreplayed. Ran in {:.2}s ===",
        total_pass,
        problems.len(),
        total_pass as f64 / problems.len() as f64 * 100.0,
        total_numerically_equivalent,
        problems.len() - total_pass - total_numerically_equivalent,
        total_elapsed.as_secs_f64()
    );

    if !failures.is_empty() {
        println!("\nFailures:");
        for f in &failures {
            println!("  - {f}");
        }
    }
}
