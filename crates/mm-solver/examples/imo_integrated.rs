//! What `IMOSolver` does with a problem statement.
//!
//! It reports that text input is unsupported and prints the keyword-derived substitution
//! hints. Nothing is searched and nothing is solved. This example exists to make that
//! boundary visible: it previously printed search statistics for a hard-coded algebraic
//! identity that had no relation to any of the statements below.
//!
//! Usage: cargo run --example imo_integrated --release -p mm-solver

use mm_solver::{IMOOutcome, IMOSolveResult, IMOSolver, IMOSolverConfig};

fn main() {
    let solver = IMOSolver::with_config(IMOSolverConfig {
        max_nodes: 500_000,
        time_limit_secs: 30,
        top_k_substitutions: 5,
        verbose: false,
    });

    println!("IMOSolver");
    println!("  Rules loaded: {}", solver.num_rules());
    println!(
        "  Substitution vocabulary: {} strategies",
        solver.vocab_size()
    );
    println!();

    let problems = [
        (
            "Functional equation",
            "Find all functions f: Z -> Z such that f(2a + f(b)) = a + b + f(a) for all \
             integers a, b.",
        ),
        (
            "Inequality with constraint",
            "Let a, b, c be positive real numbers with abc = 1. Prove that a + b + c >= 3.",
        ),
        (
            "Number theory",
            "Find all prime numbers p such that p divides 2^p - 2.",
        ),
        (
            "Iterated function",
            "Find all functions f: R -> R such that f(f(x)) = x for all x in R.",
        ),
    ];

    for (name, text) in problems {
        print_result(name, &solver.solve_text(text));
    }

    println!("To search a problem, build it as an `Expr` and call `IMOSolver::solve_expr`.");
}

fn print_result(name: &str, result: &IMOSolveResult) {
    println!("Problem: {name}");

    match &result.outcome {
        IMOOutcome::Unsupported(info) => {
            println!("  Outcome: unsupported input - {}", info.reason);
        }
        IMOOutcome::NotFound => println!("  Outcome: searched, no goal path found"),
        IMOOutcome::Solved(steps) => println!("  Outcome: solved in {} steps", steps.len()),
    }

    println!("  Substitution hints (keyword-derived, not applied):");
    for (i, sub) in result.substitutions_suggested.iter().enumerate().take(3) {
        println!(
            "    {}. {} (confidence: {:.0}%)",
            i + 1,
            sub.substitution,
            sub.confidence * 100.0
        );
    }

    println!("  Nodes explored: {}", result.stats.nodes_explored);
    println!("  Time: {:.3}s", result.elapsed.as_secs_f64());
    println!();
}
