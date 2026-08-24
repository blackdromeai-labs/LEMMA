//! Fail-closed correctness evaluation.
//!
//! Replaces the print-only `benchmark`, `benchmark_advanced` and `stress_test` examples. Those
//! reported pass counts to stdout and exited zero regardless, accepted broad output *shapes*
//! rather than values, and in two places accepted anything at all (`|_e| true`, and "any
//! `Expr::Add`" for a derivative). They also gated on `Solution::verified`, which several code
//! paths set unconditionally.
//!
//! What this suite does instead:
//!
//! - Every case states an exact expected expression, compared by canonical form. There are no
//!   shape-only predicates.
//! - Cases the current engine does not solve are listed explicitly in [`KNOWN_UNSOLVED`] and
//!   asserted to *not* produce the expected answer, so a case that starts working is a test
//!   failure telling you to move it, and a passing case that regresses is a failure too.
//! - Every solved case must also produce a trace that replays from the input to the reported
//!   result. Reaching the right answer by an unrecorded route is a failure.
//! - Provenance (rule count, action vocabulary digest, model provenance, configuration) is
//!   printed once, so a run can be tied to the corpus it was produced from.
//!
//! This measures correctness, not speed. There are no timings here.

use mm_core::{Expr, SymbolTable};
use mm_rules::{standard_rules, ActionVocabulary};
use mm_search::{assess_trace, MCTSConfig, NeuralMCTS, Solution};
use mm_verifier::Verifier;

/// Configuration every case in this suite runs under.
fn config() -> MCTSConfig {
    MCTSConfig {
        simulations: 100,
        exploration_weight: 1.41,
        max_depth: 15,
        max_simplify_iterations: 50,
    }
}

/// Cases the engine is not expected to solve today.
///
/// Listed by case name. Each is still executed, and each is asserted to *fail*, so this list
/// cannot quietly hide a regression in the other direction either.
const KNOWN_UNSOLVED: &[&str] = &[
    // `power_add` fires once on the inner product and then the outer product no longer
    // matches, leaving x^5 * x^4 unmerged.
    "x^2 * x^3 * x^4 -> x^9",
    // Requires distributing a binomial product and collecting the middle terms.
    "(x+1)(x+1) -> x^2 + 2x + 1",
    // Blocked by the verifier, not by a missing rule. `equations::cancel_multiplication`
    // correctly rewrites `2x = 10` to `x = 10/2`, but `verify_step` compares the two
    // equations as expressions, they are not value-equal, and the step is refused before the
    // search can take it. See crates/mm-verifier/tests/equation_semantics.rs.
    "2x = 10 -> x = 5",
    // Same cause. The search subtracts 5 from both sides, then needs the division step that
    // the verifier refuses.
    "3x + 5 = 17 -> x = 4",
];

struct Case {
    name: &'static str,
    input: Expr,
    expected: Expr,
}

fn solver() -> NeuralMCTS {
    NeuralMCTS::with_config(standard_rules(), Verifier::new(), config())
}

/// Print the provenance of this run once, so results can be tied to what produced them.
fn print_provenance(mcts: &NeuralMCTS) {
    let vocab = ActionVocabulary::standard();
    println!("evaluation provenance:");
    println!("  rules registered        : {}", standard_rules().len());
    println!("  action vocabulary       : {} actions", vocab.len());
    println!("  action vocabulary digest: {:#018x}", vocab.digest());
    println!("  policy provenance       : {}", mcts.provenance());
    println!("  config                  : {:?}", config());
    println!(
        "  priors                  : {}",
        if mcts.provenance().is_trained() {
            "from the loaded model"
        } else {
            "uniform; the untrained head is not consulted"
        }
    );
}

/// Run a case and report whether it produced the expected answer through a replayable trace.
fn run(mcts: &NeuralMCTS, case: &Case) -> (bool, Solution) {
    let solution = mcts.simplify(case.input.clone());

    let correct = solution.result.canonicalize() == case.expected.canonicalize();
    let replays = assess_trace(&solution.problem, &solution.result, &solution.steps).replays();

    (correct && replays, solution)
}

/// Execute a group of cases, failing on any mismatch with the expected/known-unsolved split.
fn evaluate(group: &str, cases: Vec<Case>) {
    let mcts = solver();
    let mut failures: Vec<String> = Vec::new();
    let mut solved = 0usize;

    println!("\n== {group} ==");
    for case in &cases {
        let expected_to_solve = !KNOWN_UNSOLVED.contains(&case.name);
        let (passed, solution) = run(&mcts, case);

        if passed {
            solved += 1;
        }

        println!(
            "  [{}] {} -> {:?} ({})",
            if passed { "ok" } else { "--" },
            case.name,
            solution.result,
            solution.status.label()
        );

        if expected_to_solve && !passed {
            let status = assess_trace(&solution.problem, &solution.result, &solution.steps);
            failures.push(format!(
                "{}: expected {:?}, got {:?} (trace: {})",
                case.name, case.expected, solution.result, status
            ));
        }
        if !expected_to_solve && passed {
            failures.push(format!(
                "{}: is listed in KNOWN_UNSOLVED but now succeeds; move it out of that list",
                case.name
            ));
        }
    }

    println!("  {solved}/{} solved", cases.len());
    assert!(
        failures.is_empty(),
        "{group} regressions:\n  {}",
        failures.join("\n  ")
    );
}

#[test]
fn provenance_is_recorded() {
    print_provenance(&solver());

    // The untrained default must never be presented as trained guidance.
    assert!(
        !solver().provenance().is_trained(),
        "this suite runs with an untrained policy; a trained one changes what the numbers mean"
    );
}

#[test]
fn algebraic_identities() {
    let mut symbols = SymbolTable::new();
    let x = symbols.intern("x");
    let var = Expr::Var(x);

    evaluate(
        "algebraic identities",
        vec![
            Case {
                name: "x + 0 -> x",
                input: Expr::Add(Box::new(var.clone()), Box::new(Expr::int(0))),
                expected: var.clone(),
            },
            Case {
                name: "x * 1 -> x",
                input: Expr::Mul(Box::new(var.clone()), Box::new(Expr::int(1))),
                expected: var.clone(),
            },
            Case {
                name: "x * 0 -> 0",
                input: Expr::Mul(Box::new(var.clone()), Box::new(Expr::int(0))),
                expected: Expr::int(0),
            },
            Case {
                name: "x^1 -> x",
                input: Expr::Pow(Box::new(var.clone()), Box::new(Expr::int(1))),
                expected: var.clone(),
            },
            Case {
                name: "x^0 -> 1",
                input: Expr::Pow(Box::new(var.clone()), Box::new(Expr::int(0))),
                expected: Expr::int(1),
            },
            Case {
                name: "(x + 0) * 1 -> x",
                input: Expr::Mul(
                    Box::new(Expr::Add(Box::new(var.clone()), Box::new(Expr::int(0)))),
                    Box::new(Expr::int(1)),
                ),
                expected: var.clone(),
            },
            Case {
                name: "((x + 0) * 1) + 0 -> x",
                input: Expr::Add(
                    Box::new(Expr::Mul(
                        Box::new(Expr::Add(Box::new(var.clone()), Box::new(Expr::int(0)))),
                        Box::new(Expr::int(1)),
                    )),
                    Box::new(Expr::int(0)),
                ),
                expected: var,
            },
        ],
    );
}

#[test]
fn constant_arithmetic() {
    evaluate(
        "constant arithmetic",
        vec![
            Case {
                name: "2 + 3 -> 5",
                input: Expr::Add(Box::new(Expr::int(2)), Box::new(Expr::int(3))),
                expected: Expr::int(5),
            },
            Case {
                name: "7 * 8 -> 56",
                input: Expr::Mul(Box::new(Expr::int(7)), Box::new(Expr::int(8))),
                expected: Expr::int(56),
            },
            Case {
                name: "10 - 4 -> 6",
                input: Expr::Sub(Box::new(Expr::int(10)), Box::new(Expr::int(4))),
                expected: Expr::int(6),
            },
            Case {
                name: "12 / 4 -> 3",
                input: Expr::Div(Box::new(Expr::int(12)), Box::new(Expr::int(4))),
                expected: Expr::int(3),
            },
            Case {
                name: "2^3 -> 8",
                input: Expr::Pow(Box::new(Expr::int(2)), Box::new(Expr::int(3))),
                expected: Expr::int(8),
            },
            Case {
                name: "(2+3)*(4+5) -> 45",
                input: Expr::Mul(
                    Box::new(Expr::Add(Box::new(Expr::int(2)), Box::new(Expr::int(3)))),
                    Box::new(Expr::Add(Box::new(Expr::int(4)), Box::new(Expr::int(5)))),
                ),
                expected: Expr::int(45),
            },
            Case {
                name: "(100-50)*2 + 10 -> 110",
                input: Expr::Add(
                    Box::new(Expr::Mul(
                        Box::new(Expr::Sub(Box::new(Expr::int(100)), Box::new(Expr::int(50)))),
                        Box::new(Expr::int(2)),
                    )),
                    Box::new(Expr::int(10)),
                ),
                expected: Expr::int(110),
            },
            Case {
                name: "2*(3+4) -> 14",
                input: Expr::Mul(
                    Box::new(Expr::int(2)),
                    Box::new(Expr::Add(Box::new(Expr::int(3)), Box::new(Expr::int(4)))),
                ),
                expected: Expr::int(14),
            },
        ],
    );
}

#[test]
fn trigonometric_identities() {
    let mut symbols = SymbolTable::new();
    let x = symbols.intern("x");
    let sin_sq = Expr::Pow(
        Box::new(Expr::Sin(Box::new(Expr::Var(x)))),
        Box::new(Expr::int(2)),
    );
    let cos_sq = Expr::Pow(
        Box::new(Expr::Cos(Box::new(Expr::Var(x)))),
        Box::new(Expr::int(2)),
    );

    evaluate(
        "trigonometric identities",
        vec![
            Case {
                name: "sin^2 x + cos^2 x -> 1",
                input: Expr::Add(Box::new(sin_sq.clone()), Box::new(cos_sq.clone())),
                expected: Expr::int(1),
            },
            Case {
                name: "sin^2 x + cos^2 x - 1 -> 0",
                input: Expr::Sub(
                    Box::new(Expr::Add(Box::new(sin_sq), Box::new(cos_sq))),
                    Box::new(Expr::int(1)),
                ),
                expected: Expr::int(0),
            },
            Case {
                name: "sin(0) -> 0",
                input: Expr::Sin(Box::new(Expr::int(0))),
                expected: Expr::int(0),
            },
            Case {
                name: "cos(0) -> 1",
                input: Expr::Cos(Box::new(Expr::int(0))),
                expected: Expr::int(1),
            },
        ],
    );
}

#[test]
fn derivatives() {
    let mut symbols = SymbolTable::new();
    let x = symbols.intern("x");
    let var = Expr::Var(x);
    let d = |inner: Expr| Expr::Derivative {
        expr: Box::new(inner),
        var: x,
    };

    evaluate(
        "derivatives",
        vec![
            Case {
                name: "d/dx(5) -> 0",
                input: d(Expr::int(5)),
                expected: Expr::int(0),
            },
            Case {
                name: "d/dx(x) -> 1",
                input: d(var.clone()),
                expected: Expr::int(1),
            },
            Case {
                name: "d/dx(x^2) -> 2x",
                input: d(Expr::Pow(Box::new(var.clone()), Box::new(Expr::int(2)))),
                expected: Expr::Mul(Box::new(Expr::int(2)), Box::new(var.clone())),
            },
            Case {
                name: "d/dx(x^3) -> 3x^2",
                input: d(Expr::Pow(Box::new(var.clone()), Box::new(Expr::int(3)))),
                expected: Expr::Mul(
                    Box::new(Expr::int(3)),
                    Box::new(Expr::Pow(Box::new(var.clone()), Box::new(Expr::int(2)))),
                ),
            },
            Case {
                name: "d/dx(x^4) -> 4x^3",
                input: d(Expr::Pow(Box::new(var.clone()), Box::new(Expr::int(4)))),
                expected: Expr::Mul(
                    Box::new(Expr::int(4)),
                    Box::new(Expr::Pow(Box::new(var.clone()), Box::new(Expr::int(3)))),
                ),
            },
            Case {
                name: "d/dx(sin x) -> cos x",
                input: d(Expr::Sin(Box::new(var.clone()))),
                expected: Expr::Cos(Box::new(var.clone())),
            },
            Case {
                name: "d/dx(cos x) -> -sin x",
                input: d(Expr::Cos(Box::new(var.clone()))),
                expected: Expr::Neg(Box::new(Expr::Sin(Box::new(var.clone())))),
            },
            Case {
                name: "d/dx(x + 5) -> 1",
                input: d(Expr::Add(Box::new(var.clone()), Box::new(Expr::int(5)))),
                expected: Expr::int(1),
            },
            Case {
                name: "d/dx(2x) -> 2",
                input: d(Expr::Mul(Box::new(Expr::int(2)), Box::new(var.clone()))),
                expected: Expr::int(2),
            },
            // Exact expected value, rather than the old "any Expr::Add" predicate.
            Case {
                name: "d/dx(x^2 + x^3) -> 2x + 3x^2",
                input: d(Expr::Add(
                    Box::new(Expr::Pow(Box::new(var.clone()), Box::new(Expr::int(2)))),
                    Box::new(Expr::Pow(Box::new(var.clone()), Box::new(Expr::int(3)))),
                )),
                expected: Expr::Add(
                    Box::new(Expr::Mul(Box::new(Expr::int(2)), Box::new(var.clone()))),
                    Box::new(Expr::Mul(
                        Box::new(Expr::int(3)),
                        Box::new(Expr::Pow(Box::new(var.clone()), Box::new(Expr::int(2)))),
                    )),
                ),
            },
            Case {
                name: "d/dx(sin x + cos x) -> cos x - sin x",
                input: d(Expr::Add(
                    Box::new(Expr::Sin(Box::new(var.clone()))),
                    Box::new(Expr::Cos(Box::new(var.clone()))),
                )),
                expected: Expr::Add(
                    Box::new(Expr::Cos(Box::new(var.clone()))),
                    Box::new(Expr::Neg(Box::new(Expr::Sin(Box::new(var))))),
                ),
            },
        ],
    );
}

#[test]
fn multi_step_algebra() {
    let mut symbols = SymbolTable::new();
    let x = symbols.intern("x");
    let y = symbols.intern("y");
    let (vx, vy) = (Expr::Var(x), Expr::Var(y));
    let pow = |base: Expr, n: i64| Expr::Pow(Box::new(base), Box::new(Expr::int(n)));

    evaluate(
        "multi-step algebra",
        vec![
            Case {
                name: "x^2 * x^3 -> x^5",
                input: Expr::Mul(Box::new(pow(vx.clone(), 2)), Box::new(pow(vx.clone(), 3))),
                expected: pow(vx.clone(), 5),
            },
            Case {
                name: "x^2 * x^3 * x^4 -> x^9",
                input: Expr::Mul(
                    Box::new(Expr::Mul(
                        Box::new(pow(vx.clone(), 2)),
                        Box::new(pow(vx.clone(), 3)),
                    )),
                    Box::new(pow(vx.clone(), 4)),
                ),
                expected: pow(vx.clone(), 9),
            },
            Case {
                name: "x + y + 0 -> x + y",
                input: Expr::Add(
                    Box::new(Expr::Add(Box::new(vx.clone()), Box::new(vy.clone()))),
                    Box::new(Expr::int(0)),
                ),
                expected: Expr::Add(Box::new(vx.clone()), Box::new(vy.clone())),
            },
            Case {
                name: "x * y * 1 -> x * y",
                input: Expr::Mul(
                    Box::new(Expr::Mul(Box::new(vx.clone()), Box::new(vy.clone()))),
                    Box::new(Expr::int(1)),
                ),
                expected: Expr::Mul(Box::new(vx.clone()), Box::new(vy.clone())),
            },
            // Was accepted by `|_e| true`. Now it states the answer it should reach.
            Case {
                name: "(x+1)(x+1) -> x^2 + 2x + 1",
                input: Expr::Mul(
                    Box::new(Expr::Add(Box::new(vx.clone()), Box::new(Expr::int(1)))),
                    Box::new(Expr::Add(Box::new(vx.clone()), Box::new(Expr::int(1)))),
                ),
                expected: Expr::Add(
                    Box::new(Expr::Add(
                        Box::new(pow(vx.clone(), 2)),
                        Box::new(Expr::Mul(Box::new(Expr::int(2)), Box::new(vx.clone()))),
                    )),
                    Box::new(Expr::int(1)),
                ),
            },
            Case {
                name: "2(x+y) + 3(x+y) -> 5(x+y)",
                input: Expr::Add(
                    Box::new(Expr::Mul(
                        Box::new(Expr::int(2)),
                        Box::new(Expr::Add(Box::new(vx.clone()), Box::new(vy.clone()))),
                    )),
                    Box::new(Expr::Mul(
                        Box::new(Expr::int(3)),
                        Box::new(Expr::Add(Box::new(vx.clone()), Box::new(vy.clone()))),
                    )),
                ),
                expected: Expr::Mul(
                    Box::new(Expr::int(5)),
                    Box::new(Expr::Add(Box::new(vx), Box::new(vy))),
                ),
            },
        ],
    );
}

#[test]
fn equation_solving() {
    let mut symbols = SymbolTable::new();
    let x = symbols.intern("x");
    let var = Expr::Var(x);
    let eq = |lhs: Expr, rhs: Expr| Expr::Equation {
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    };

    evaluate(
        "equation solving",
        vec![
            Case {
                name: "x + 3 = 7 -> x = 4",
                input: eq(
                    Expr::Add(Box::new(var.clone()), Box::new(Expr::int(3))),
                    Expr::int(7),
                ),
                expected: eq(var.clone(), Expr::int(4)),
            },
            Case {
                name: "2x = 10 -> x = 5",
                input: eq(
                    Expr::Mul(Box::new(Expr::int(2)), Box::new(var.clone())),
                    Expr::int(10),
                ),
                expected: eq(var.clone(), Expr::int(5)),
            },
            Case {
                name: "3x + 5 = 17 -> x = 4",
                input: eq(
                    Expr::Add(
                        Box::new(Expr::Mul(Box::new(Expr::int(3)), Box::new(var.clone()))),
                        Box::new(Expr::int(5)),
                    ),
                    Expr::int(17),
                ),
                expected: eq(var, Expr::int(4)),
            },
        ],
    );
}

#[test]
fn every_solved_case_has_a_replayable_trace() {
    // Guards the property the old runners could not check: reaching the right answer is not
    // enough, the recorded steps have to get there from the exact input.
    let mcts = solver();
    let mut symbols = SymbolTable::new();
    let x = symbols.intern("x");

    let inputs = vec![
        Expr::Add(Box::new(Expr::int(2)), Box::new(Expr::int(3))),
        Expr::Mul(Box::new(Expr::Var(x)), Box::new(Expr::int(1))),
        Expr::Derivative {
            expr: Box::new(Expr::Pow(Box::new(Expr::Var(x)), Box::new(Expr::int(2)))),
            var: x,
        },
    ];

    for input in inputs {
        let solution = mcts.simplify(input.clone());
        let derived = assess_trace(&solution.problem, &solution.result, &solution.steps);
        assert_eq!(
            derived, solution.status,
            "reported status disagrees with the recorded trace for {input:?}"
        );
        assert!(
            derived.replays(),
            "no replayable path from {:?} to {:?} ({derived})",
            input,
            solution.result
        );
    }
}
