//! Rule-availability report for problem statements.
//!
//! Prints the keyword classifier's topic guess for each statement and lists which rules the
//! registry would offer for a matching expression. No search runs and nothing is solved: the
//! classifier is keyword-based, not a trained model.
//!
//! Usage: cargo run --example neural_mcts_demo --release -p mm-solver

use mm_brain::KeywordProblemClassifier;
use mm_core::{Expr, Rational, SymbolTable};
use mm_rules::rule::standard_rules;
use mm_rules::{RuleCategory, RuleContext};
use std::path::Path;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       Neural-Guided MCTS - IMO Problem Solver                ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // The classifier only needs a token vocabulary; its scoring is keyword-based.
    let vocab_path = Path::new("models/vocab.txt");

    let classifier: Option<KeywordProblemClassifier> =
        match KeywordProblemClassifier::from_vocab(vocab_path) {
            Ok(m) => {
                println!("Keyword classifier ready: {} vocab tokens\n", m.vocab_len());
                Some(m)
            }
            Err(e) => {
                println!("No vocabulary file ({e}); skipping topic classification\n");
                None
            }
        };

    // Load rules
    let rules = standard_rules();
    println!("✓ Loaded {} rules\n", rules.len());

    // Test problems
    let problems = vec![
        ("Prove x² + y² ≥ 2xy for all real x, y", "inequality"),
        (
            "Find all functions f: ℝ → ℝ such that f(x+y) = f(x) + f(y)",
            "functional",
        ),
        (
            "Show that if p is prime and p | ab, then p | a or p | b",
            "number_theory",
        ),
        ("Prove that (a+b+c)/3 ≥ ∛(abc) for positive a,b,c", "am_gm"),
        ("Factor x³ - 1", "algebra"),
    ];

    println!("═══════════════════════════════════════════════════════════════");
    println!("                   PROBLEM ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════\n");

    for (problem, expected_type) in &problems {
        println!("Problem: {}", problem);
        println!("Expected: {}", expected_type);

        if let Some(ref m) = classifier {
            println!("Keyword topic scores:");
            for (idx, prob) in m.predict_top_k(problem, 3) {
                println!(
                    "  -> {} ({:.1}%)",
                    KeywordProblemClassifier::class_name(idx),
                    prob * 100.0
                );
            }
        }

        // Find applicable rules
        let ctx = RuleContext::default();

        // Create a simple expression to test rules
        let mut symbols = SymbolTable::new();
        let x = symbols.intern("x");
        let y = symbols.intern("y");

        let test_expr = Expr::Add(
            Box::new(Expr::Pow(Box::new(Expr::Var(x)), Box::new(Expr::int(2)))),
            Box::new(Expr::Pow(Box::new(Expr::Var(y)), Box::new(Expr::int(2)))),
        );

        let applicable = rules.applicable(&test_expr, &ctx);
        println!("Applicable rules to x²+y²: {} rules", applicable.len());

        // Show top 5 by category relevance
        let prioritized = prioritize_rules_for_problem(problem, &applicable);
        println!("Top prioritized rules:");
        for (i, rule) in prioritized.iter().take(5).enumerate() {
            println!("  {}. {} ({:?})", i + 1, rule.name, rule.category);
        }

        println!();
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("                   MCTS SEARCH DEMO");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Demo: Expand (a+b)² using MCTS with rule prioritization
    let mut symbols = SymbolTable::new();
    let a = symbols.intern("a");
    let b = symbols.intern("b");

    let expr = Expr::Pow(
        Box::new(Expr::Add(Box::new(Expr::Var(a)), Box::new(Expr::Var(b)))),
        Box::new(Expr::Const(Rational::from(2))),
    );

    println!("Starting expression: (a + b)²");
    println!("Goal: Expand to a² + 2ab + b²\n");

    let start = Instant::now();
    let ctx = RuleContext::default();
    let applicable = rules.applicable(&expr, &ctx);

    println!("Found {} applicable rules:", applicable.len());
    for rule in &applicable {
        let results = (rule.apply)(&expr, &ctx);
        if !results.is_empty() {
            let result_str = format!("{:?}", results[0].result);
            let truncated = if result_str.len() > 50 {
                format!("{}...", &result_str[..50])
            } else {
                result_str
            };
            println!("  {} → {}", rule.name, truncated);
        }
    }

    println!(
        "\nSearch time: {:.2}ms",
        start.elapsed().as_secs_f64() * 1000.0
    );
    println!("\n✅ Neural-Guided MCTS Demo Complete!");
}

/// Prioritize rules based on problem text analysis
fn prioritize_rules_for_problem<'a>(
    problem: &str,
    rules: &[&'a mm_rules::Rule],
) -> Vec<&'a mm_rules::Rule> {
    let problem_lower = problem.to_lowercase();
    let mut scored: Vec<(&mm_rules::Rule, i32)> = rules
        .iter()
        .map(|&r| {
            let mut score = 0i32;

            // Boost based on problem keywords vs rule category
            if problem_lower.contains("prove") || problem_lower.contains("show") {
                if matches!(
                    r.category,
                    RuleCategory::AlgebraicSolving | RuleCategory::Simplification
                ) {
                    score += 10;
                }
            }
            if problem_lower.contains("inequality")
                || problem_lower.contains("≥")
                || problem_lower.contains(">=")
            {
                if r.name.contains("am_gm")
                    || r.name.contains("cauchy")
                    || r.name.contains("inequality")
                {
                    score += 20;
                }
            }
            if problem_lower.contains("factor") {
                if matches!(r.category, RuleCategory::Factoring) {
                    score += 15;
                }
            }
            if problem_lower.contains("prime") || problem_lower.contains("divisor") {
                if r.name.contains("fermat") || r.name.contains("euler") || r.name.contains("prime")
                {
                    score += 20;
                }
            }
            if problem_lower.contains("function") || problem_lower.contains("f(") {
                if r.name.contains("function") || r.name.contains("substitut") {
                    score += 15;
                }
            }

            (r, score)
        })
        .collect();

    // Sort by score descending
    scored.sort_by(|a, b| b.1.cmp(&a.1));

    scored.into_iter().map(|(r, _)| r).collect()
}
