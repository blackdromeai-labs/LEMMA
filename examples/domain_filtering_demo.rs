//! What `mm_boink::analyze` actually does: a coarse domain profile of an expression, used to
//! pre-filter which rules `NeuralMCTS` even tries.
//!
//! This replaces a former `boink_demo` that demonstrated a credit-tracking, budget-allocating
//! "BOINK supervisor" layer with a bank of credits and a "premium" unlock at 20,000 credits.
//! None of that was real: nothing in `mm-solver` or `mm-tui` ever consulted the bank or the
//! budget, "premium" gated no actual feature, and the wrapper that reported those numbers
//! (`BoinkMCTS`) had no test anywhere in its own crate. It has been removed; what remains in
//! `mm-boink` is only the domain-profile analyzer this example actually exercises.
//!
//! Run: cargo run --example domain_filtering_demo

use mm_boink::analyze;
use mm_core::{Expr, SymbolTable};
use mm_rules::standard_rules;
use mm_search::NeuralMCTS;
use mm_verifier::Verifier;

fn main() {
    let mut symbols = SymbolTable::new();
    let x = symbols.intern("x");
    let mcts = NeuralMCTS::new(standard_rules(), Verifier::new());

    let examples: Vec<(&str, Expr)> = vec![
        (
            "x + 0",
            Expr::Add(Box::new(Expr::Var(x)), Box::new(Expr::int(0))),
        ),
        (
            "sin(x)^2 + cos(x)^2",
            Expr::Add(
                Box::new(Expr::Pow(
                    Box::new(Expr::Sin(Box::new(Expr::Var(x)))),
                    Box::new(Expr::int(2)),
                )),
                Box::new(Expr::Pow(
                    Box::new(Expr::Cos(Box::new(Expr::Var(x)))),
                    Box::new(Expr::int(2)),
                )),
            ),
        ),
        (
            "d/dx(x^2)",
            Expr::Derivative {
                expr: Box::new(Expr::Pow(Box::new(Expr::Var(x)), Box::new(Expr::int(2)))),
                var: x,
            },
        ),
    ];

    for (label, expr) in examples {
        let profile = analyze(&expr);
        println!("{label}");
        println!(
            "  domains detected: {:?} (complexity {})",
            profile.domains, profile.complexity
        );

        let solution = mcts.simplify(expr);
        println!(
            "  simplify -> {:?}  [{}]",
            solution.result,
            solution.status.label()
        );
        println!();
    }
}
