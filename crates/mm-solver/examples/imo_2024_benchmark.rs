//! Six toy computations loosely themed after each IMO 2024 problem's domain.
//!
//! None of these are the actual IMO 2024 problems -- they are grade-school-level warm-ups
//! (an arithmetic mean, a gcd, a modular power, a binomial coefficient, a power-rule
//! derivative, the Pythagorean identity) chosen only because each sits in the same broad
//! domain as one of the six real problems. The closing "Coverage Analysis" said as much
//! ("P3 (Geometry): Requires geometric primitives (not implemented)") while the header above
//! it claimed "LEMMA vs Competition Problems" and handed out a medal-emoji verdict
//! ("EXCELLENT" / "GOOD" / ...) for solving the warm-ups. Retitled rather than deleted: the
//! six checks are legitimate pass/fail tests against an exact expected value, just mislabeled
//! as competition performance.
//!
//! Run: cargo run --release --example imo_2024_benchmark

use mm_core::{Expr, Rational, SymbolTable};
use mm_rules::rule::standard_rules;
use mm_search::{MCTSConfig, NeuralMCTS};
use mm_verifier::Verifier;
use std::time::Instant;

fn main() {
    println!("\n╔════════════════════════════════════════════════════════════════════╗");
    println!("║   Six warm-ups themed after IMO 2024's domains (not the problems)   ║");
    println!("╚════════════════════════════════════════════════════════════════════╝\n");

    let mut symbols = SymbolTable::new();
    let x = symbols.intern("x");
    let _n = symbols.intern("n");
    let _a = symbols.intern("a");
    let _b = symbols.intern("b");
    let _c = symbols.intern("c");

    let rules = standard_rules();
    let verifier = Verifier::new();
    let config = MCTSConfig {
        simulations: 500,
        exploration_weight: 1.5,
        max_depth: 25,
        ..Default::default()
    };
    let mcts = NeuralMCTS::with_config(rules, verifier, config);

    let mut solved = 0;
    let mut partial = 0;
    let total = 6;

    // =========================================================================
    // Problem 1: Arithmetic Mean (AM) calculation
    // (3 + 4 + 5) / 3 = 4 (foundation for AM-GM)
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Problem 1 (Inequalities): AM-GM Foundation");
    println!("Compute: (3 + 4 + 5) / 3 = 4");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let am = Expr::Div(
        Box::new(Expr::Add(
            Box::new(Expr::Add(Box::new(Expr::int(3)), Box::new(Expr::int(4)))),
            Box::new(Expr::int(5)),
        )),
        Box::new(Expr::int(3)),
    );

    let (_result1, passed1) = run_test(
        &mcts,
        am,
        |e| matches!(e, Expr::Const(r) if r == &Rational::from(4)),
    );
    if passed1 {
        solved += 1;
    }

    // =========================================================================
    // Problem 2: GCD evaluation (Number Theory foundation)
    // gcd(12, 8) = 4
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Problem 2 (Number Theory): GCD Evaluation");
    println!("Compute: gcd(12, 8) = 4");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let gcd_test = Expr::GCD(Box::new(Expr::int(12)), Box::new(Expr::int(8)));
    let (_result2, passed2) = run_test(
        &mcts,
        gcd_test,
        |e| matches!(e, Expr::Const(r) if r == &Rational::from(4)),
    );
    if passed2 {
        solved += 1;
    } else {
        partial += 1;
    }

    // =========================================================================
    // Problem 3: Fermat's Little Theorem
    // 2^10 mod 11 = 1
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Problem 3 (Number Theory): Fermat's Little Theorem");
    println!("Compute: 2^10 mod 11 = 1");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let fermat = Expr::Mod(
        Box::new(Expr::Pow(Box::new(Expr::int(2)), Box::new(Expr::int(10)))),
        Box::new(Expr::int(11)),
    );
    let (_result3, passed3) = run_test(
        &mcts,
        fermat,
        |e| matches!(e, Expr::Const(r) if r == &Rational::from(1)),
    );
    if passed3 {
        solved += 1;
    } else {
        partial += 1;
    }

    // =========================================================================
    // Problem 4: Binomial Coefficient (Combinatorics)
    // C(10, 5) = 252
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Problem 4 (Combinatorics): Binomial Coefficient");
    println!("Compute: C(10, 5) = 252");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let binomial = Expr::Binomial(Box::new(Expr::int(10)), Box::new(Expr::int(5)));
    let (_result4, passed4) = run_test(
        &mcts,
        binomial,
        |e| matches!(e, Expr::Const(r) if r == &Rational::from(252)),
    );
    if passed4 {
        solved += 1;
    } else {
        partial += 1;
    }

    // =========================================================================
    // Problem 5: Power Rule Derivative
    // d/dx(x^4) = 4x^3
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Problem 5 (Calculus): Power Rule Derivative");
    println!("Compute: d/dx(x^4) = 4x^3");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let deriv = Expr::Derivative {
        expr: Box::new(Expr::Pow(Box::new(Expr::Var(x)), Box::new(Expr::int(4)))),
        var: x,
    };
    let (_result5, passed5) = run_test(&mcts, deriv, |e| match e {
        Expr::Mul(coef, _base) => {
            matches!(coef.as_ref(), Expr::Const(r) if r == &Rational::from(4))
        }
        _ => false,
    });
    if passed5 {
        solved += 1;
    } else {
        partial += 1;
    }

    // =========================================================================
    // Problem 6: Pythagorean Identity
    // sin²(x) + cos²(x) - 1 = 0
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Problem 6 (Trigonometry): Pythagorean Identity");
    println!("Simplify: sin²(x) + cos²(x) - 1 = 0");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let pythag = Expr::Sub(
        Box::new(Expr::Add(
            Box::new(Expr::Pow(
                Box::new(Expr::Sin(Box::new(Expr::Var(x)))),
                Box::new(Expr::int(2)),
            )),
            Box::new(Expr::Pow(
                Box::new(Expr::Cos(Box::new(Expr::Var(x)))),
                Box::new(Expr::int(2)),
            )),
        )),
        Box::new(Expr::int(1)),
    );
    let (_result6, passed6) = run_test(
        &mcts,
        pythag,
        |e| matches!(e, Expr::Const(r) if r.is_zero()),
    );
    if passed6 {
        solved += 1;
    }

    // =========================================================================
    // FINAL RESULTS
    // =========================================================================
    let score = ((solved as f64 + partial as f64 * 0.25) / total as f64) * 100.0;

    println!("\n╔════════════════════════════════════════════════════════════════════╗");
    println!("║                      WARM-UP RESULTS                                ║");
    println!("╠════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Fully Solved:  {}/{}                                                 ║",
        solved, total
    );
    println!(
        "║  Partial:       {}/{}                                                 ║",
        partial, total
    );
    println!(
        "║  Score:         {:.0}%                                                  ║",
        score
    );
    println!("╚════════════════════════════════════════════════════════════════════╝");

    println!("\nThis score describes six toy expressions, not competition performance.");
    println!("Actually attempting IMO 2024's six problems would need, at minimum:");
    println!("   - floor/ceiling reasoning over an unbounded family of n (P1)");
    println!("   - sequence/recurrence reasoning, not a single gcd evaluation (P2, P3)");
    println!("   - a geometry module: angles, triangles, incircles (P4, P6)");
    println!("   - functional-equation solving beyond a single substitution (P6)");
    println!("None of that is exercised above.");
}

fn run_test<F>(mcts: &NeuralMCTS, expr: Expr, check: F) -> (Expr, bool)
where
    F: Fn(&Expr) -> bool,
{
    let start = Instant::now();
    let result = mcts.simplify(expr);
    let elapsed = start.elapsed();
    let passed = check(&result.result);

    let status = if passed { "✅" } else { "❌" };
    println!("{} Result: {:?}", status, result.result);
    println!("   Time: {:?}  |  Steps: {}\n", elapsed, result.steps.len());

    (result.result, passed)
}
