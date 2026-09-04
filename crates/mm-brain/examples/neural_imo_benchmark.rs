//! What `SubstitutionPredictor` does with an IMO problem statement.
//!
//! It scores a fixed 20-entry vocabulary of common IMO substitutions (`x = 0`, `Apply AM-GM`,
//! `WLOG assume ordering`, ...) against pattern matches in the problem text and returns the
//! top-k by score. Despite "Neural" in this file's old name, it is not a trained model --
//! there is no learned weight anywhere in the path -- and it does not solve, search, or prove
//! anything. This example used to also run `mm_core::search::ProofSearchEngine` over an empty
//! `ProofState` (two declared variables, no encoding of the problem at all) and print a
//! "hit rate" scorecard comparing the predicted strings against a hand-picked hint list by
//! substring match. Both steps were decoration: the "search" explored nothing related to the
//! problem, and the "hit rate" measured whether two free-text lists happened to overlap, not
//! whether anything was proven. Removed rather than fixed, since there is no real search or
//! verification step to report on here -- see `mm-solver`'s `imo_integrated` example and its
//! `IMOSolver::solve_text`, which is the actual (and honestly unsupported) entry point for
//! text problems.
//!
//! Usage: cargo run --example neural_imo_benchmark --release -p mm-brain

use mm_brain::SubstitutionPredictor;

/// A real IMO problem statement, used only as text for the predictor to score.
struct IMOProblem {
    year: u32,
    number: u32,
    domain_type: &'static str,
    statement: &'static str,
}

fn main() {
    let predictor = SubstitutionPredictor::new();

    let problems = vec![
        IMOProblem {
            year: 2019,
            number: 1,
            domain_type: "Functional Equation",
            statement: r#"
Let f: Z -> Z be a function such that for all integers a and b,
f(2a) + 2f(b) = f(f(a + b)).
Determine all such functions f.
"#,
        },
        IMOProblem {
            year: 2017,
            number: 2,
            domain_type: "Functional Equation",
            statement: r#"
Let R be the set of real numbers. Find all functions f: R -> R such that
for all x, y in R, f(f(x)f(y)) + f(x + y) = f(xy).
"#,
        },
        IMOProblem {
            year: 2008,
            number: 2,
            domain_type: "Algebra/Inequality",
            statement: r#"
Let x, y, z be distinct real numbers different from 1 such that xyz = 1.
Prove that x^2/(x-1)^2 + y^2/(y-1)^2 + z^2/(z-1)^2 >= 1.
"#,
        },
        IMOProblem {
            year: 2024,
            number: 1,
            domain_type: "Number Theory",
            statement: r#"
Determine all real numbers alpha such that, for every positive integer n,
the integer floor(alpha) + floor(2*alpha) + ... + floor(n*alpha) is a multiple of n.
"#,
        },
    ];

    println!(
        "Scoring {} IMO problem statements against a fixed {}-entry substitution vocabulary.",
        problems.len(),
        mm_brain::substitution::SUBSTITUTION_VOCAB.len()
    );
    println!("Pattern-matched keyword scoring, not a trained model. Nothing is solved.\n");

    for p in &problems {
        println!("IMO {} Problem {} [{}]", p.year, p.number, p.domain_type);

        let predictions = predictor.predict(p.statement, 5);
        for (i, pred) in predictions.iter().enumerate() {
            println!(
                "  {}. {} (score {:.0}%)",
                i + 1,
                pred.substitution,
                pred.confidence * 100.0
            );
        }
        println!();
    }
}
