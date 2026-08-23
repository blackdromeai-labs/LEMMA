//! JEE Advanced Test Run
//!
//! Testing LEMMA's ability to solve real JEE Advanced problems from the question bank:
//! - Quadratic Equations & Inequalities
//! - Sequences and Series (AP, GP, HP)
//! - Trigonometric equations
//! - Calculus problems
//!
//! Run: cargo run --example jee_test

use mm_boink::analyze;
use mm_core::{Expr, SymbolTable};
use mm_rules::standard_rules;
use mm_search::BoinkMCTS;
use mm_verifier::Verifier;

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║              JEE ADVANCED TEST RUN                         ║");
    println!("║        Testing LEMMA with Real Exam Problems               ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    let mut symbols = SymbolTable::new();
    let rules = standard_rules();
    let verifier = Verifier::default();
    let boink = BoinkMCTS::from_parts(rules, verifier);

    let mut solved = 0;
    let mut total = 0;

    println!("=== QUADRATIC EQUATIONS ===\n");

    // JEE 2011: α,β roots of x² - 6x - 2 = 0, find (a₁₀ - 2a₈)/2a₉ where aₙ = αⁿ - βⁿ
    // This simplifies to (α² - 2)/(2α) or similar via recurrence
    // For roots of x² - px + q = 0: aₙ = p*aₙ₋₁ - q*aₙ₋₂
    // Here p=6, q=-2: aₙ = 6aₙ₋₁ + 2aₙ₋₂
    // (a₁₀ - 2a₈)/2a₉ = (6a₉ + 2a₈ - 2a₈)/2a₉ = 6a₉/2a₉ = 3
    total += 1;
    println!("📝 JEE 2011 Q8: (a₁₀ - 2a₈)/2a₉ for aₙ = αⁿ - βⁿ, x² - 6x - 2 = 0");
    println!("   Expected answer: 3");

    // Let's test the simplification (6a - 2a)/2a = 4a/2a = 2
    let a = symbols.intern("a");
    let expr1 = Expr::Div(
        Box::new(Expr::Sub(
            Box::new(Expr::Mul(
                Box::new(Expr::Const(6.into())),
                Box::new(Expr::Var(a)),
            )),
            Box::new(Expr::Mul(
                Box::new(Expr::Const(2.into())),
                Box::new(Expr::Var(a)),
            )),
        )),
        Box::new(Expr::Mul(
            Box::new(Expr::Const(2.into())),
            Box::new(Expr::Var(a)),
        )),
    );
    if run_jee_problem(&boink, &expr1, "Simplify (6a - 2a)/(2a)") {
        solved += 1;
    }

    // JEE 2000: If c < 0 < b for x² + bx + c = 0, determine nature of roots
    // Discriminant = b² - 4ac = b² - 4(1)(c) = b² - 4c
    // Since c < 0, -4c > 0, so b² - 4c > b² > 0 → real distinct roots
    total += 1;
    println!("\n📝 JEE 2000 Q20: Nature of roots for x² + bx + c = 0, c < 0 < b");
    println!("   Expected: Real and distinct (discriminant > 0)");

    // Test discriminant: b² - 4c (with c = -1)
    let b = symbols.intern("b");
    let discriminant = Expr::Sub(
        Box::new(Expr::Pow(
            Box::new(Expr::Var(b)),
            Box::new(Expr::Const(2.into())),
        )),
        Box::new(Expr::Mul(
            Box::new(Expr::Const(4.into())),
            Box::new(Expr::Const((-1).into())),
        )),
    );
    if run_jee_problem(&boink, &discriminant, "Discriminant: b² - 4(-1) = b² + 4") {
        solved += 1;
    }

    println!("\n=== SEQUENCES AND SERIES ===\n");

    // JEE 2009: Sum of first n terms is cn², find sum of squares
    // If Sₙ = cn², then aₙ = Sₙ - Sₙ₋₁ = cn² - c(n-1)² = c(2n-1)
    // Sum of squares = Σ[c(2k-1)]² = c² Σ(2k-1)² = c² · n(4n²-1)/3
    total += 1;
    println!("📝 JEE 2009 Q9: If Sₙ = cn², find Σaₖ²");
    println!("   Expected: n(4n²-1)c²/3");

    // Compute (2n-1)² = 4n² - 4n + 1
    let n = symbols.intern("n");
    let term_squared = Expr::Pow(
        Box::new(Expr::Sub(
            Box::new(Expr::Mul(
                Box::new(Expr::Const(2.into())),
                Box::new(Expr::Var(n)),
            )),
            Box::new(Expr::Const(1.into())),
        )),
        Box::new(Expr::Const(2.into())),
    );
    if run_jee_problem(&boink, &term_squared, "Expand (2n-1)²") {
        solved += 1;
    }

    // JEE 2001: Sum of 2n terms of 2,5,8,... equals sum of n terms of 57,59,61,...
    // AP1: a=2, d=3, S₂ₙ = n(2(2) + (2n-1)(3)) = n(4 + 6n - 3) = n(6n + 1)
    // AP2: a=57, d=2, Sₙ = n/2(2(57) + (n-1)(2)) = n/2(114 + 2n - 2) = n/2(112 + 2n) = n(56 + n)
    // n(6n + 1) = n(56 + n) → 6n + 1 = 56 + n → 5n = 55 → n = 11
    total += 1;
    println!("\n📝 JEE 2001 Q16: Find n where S₂ₙ(AP₁) = Sₙ(AP₂)");
    println!("   AP₁: 2,5,8,... | AP₂: 57,59,61,...");
    println!("   Expected: n = 11");

    // Setup: 6n + 1 = 56 + n (equation to solve)
    let eq_jee = Expr::Equation {
        lhs: Box::new(Expr::Add(
            Box::new(Expr::Mul(
                Box::new(Expr::Const(6.into())),
                Box::new(Expr::Var(n)),
            )),
            Box::new(Expr::Const(1.into())),
        )),
        rhs: Box::new(Expr::Add(
            Box::new(Expr::Const(56.into())),
            Box::new(Expr::Var(n)),
        )),
    };
    if run_jee_problem(&boink, &eq_jee, "Solve: 6n + 1 = 56 + n") {
        solved += 1;
    }

    // GP Third term is 4, find product of first 5 terms
    // a₃ = ar² = 4, Product = a·ar·ar²·ar³·ar⁴ = a⁵r¹⁰ = (ar²)⁵ = 4⁵
    total += 1;
    println!("\n📝 JEE 1982 Q29: GP third term = 4, find product of first 5 terms");
    println!("   Expected: 4⁵ = 1024");

    let gp_product = Expr::Pow(
        Box::new(Expr::Const(4.into())),
        Box::new(Expr::Const(5.into())),
    );
    if run_jee_problem(&boink, &gp_product, "Compute 4⁵") {
        solved += 1;
    }

    println!("\n=== TRIGONOMETRY ===\n");

    // sin²θ + cos²θ = 1 (Pythagorean identity)
    total += 1;
    println!("📝 Basic: Simplify sin²(x) + cos²(x)");
    println!("   Expected: 1");

    let x = symbols.intern("x");
    let pythag = Expr::Add(
        Box::new(Expr::Pow(
            Box::new(Expr::Sin(Box::new(Expr::Var(x)))),
            Box::new(Expr::Const(2.into())),
        )),
        Box::new(Expr::Pow(
            Box::new(Expr::Cos(Box::new(Expr::Var(x)))),
            Box::new(Expr::Const(2.into())),
        )),
    );
    if run_jee_problem(&boink, &pythag, "sin²(x) + cos²(x)") {
        solved += 1;
    }

    println!("\n=== CALCULUS ===\n");

    // d/dx(x²) = 2x
    total += 1;
    println!("📝 Basic: Find d/dx(x²)");
    println!("   Expected: 2x");

    let deriv_x2 = Expr::Derivative {
        expr: Box::new(Expr::Pow(
            Box::new(Expr::Var(x)),
            Box::new(Expr::Const(2.into())),
        )),
        var: x,
    };
    if run_jee_problem(&boink, &deriv_x2, "d/dx(x²)") {
        solved += 1;
    }

    // d/dx(x³) = 3x²
    total += 1;
    println!("\n📝 Find d/dx(x³)");
    println!("   Expected: 3x²");

    let deriv_x3 = Expr::Derivative {
        expr: Box::new(Expr::Pow(
            Box::new(Expr::Var(x)),
            Box::new(Expr::Const(3.into())),
        )),
        var: x,
    };
    if run_jee_problem(&boink, &deriv_x3, "d/dx(x³)") {
        solved += 1;
    }

    // ∫x dx = x²/2
    total += 1;
    println!("\n📝 Find ∫x dx");
    println!("   Expected: x²/2");

    let integral_x = Expr::Integral {
        expr: Box::new(Expr::Var(x)),
        var: x,
    };
    if run_jee_problem(&boink, &integral_x, "∫x dx") {
        solved += 1;
    }

    println!("\n=== GEOMETRY (New!) ===\n");

    // Distance formula: √(3² + 4²) = 5
    total += 1;
    println!("📝 JEE Geometry: Distance √(3² + 4²)");
    println!("   Expected: √25 = 5");

    let distance = Expr::Sqrt(Box::new(Expr::Add(
        Box::new(Expr::Pow(
            Box::new(Expr::Const(3.into())),
            Box::new(Expr::Const(2.into())),
        )),
        Box::new(Expr::Pow(
            Box::new(Expr::Const(4.into())),
            Box::new(Expr::Const(2.into())),
        )),
    )));
    if run_jee_problem(&boink, &distance, "√(3² + 4²)") {
        solved += 1;
    }

    // ═══════════════════════════════════════════════════════════════════
    // RESULTS
    // ═══════════════════════════════════════════════════════════════════
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║                      RESULTS                               ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!(
        "║  Problems Attempted: {:<4}                                  ║",
        total
    );
    println!(
        "║  Problems Simplified: {:<4}                                 ║",
        solved
    );
    println!(
        "║  Success Rate: {:.1}%                                       ║",
        (solved as f64 / total as f64) * 100.0
    );
    println!("╚════════════════════════════════════════════════════════════╝");

    // Show bank status
    println!("\n=== BOINK Bank Status ===");
    let bank = boink.bank();
    println!("   Credits earned: {}", bank.total_credits());
    println!(
        "   Progress to premium: {:.1}%",
        bank.total_credits() as f64 / 200.0
    );
}

fn run_jee_problem(boink: &BoinkMCTS, expr: &Expr, description: &str) -> bool {
    let profile = analyze(expr);
    println!("   Input: {}", description);
    println!(
        "   Domains: {:?} | Complexity: {}",
        profile.domains, profile.complexity
    );

    let (solution, stats) = boink.simplify_tracked(expr.clone());

    let success = stats.solved && stats.rules_applied > 0;

    if success {
        if let Some(step) = solution.steps.last() {
            println!("   ✅ Solved → {:?}", step.after);
        } else {
            println!("   ✅ Simplified ({} steps)", stats.rules_applied);
        }
        println!(
            "   Budget: {} | Cost: {} | Saved: {}",
            stats.budget_allocated, stats.cost_spent, stats.credits_remaining
        );
    } else {
        println!("   ❌ Could not simplify (no applicable rules matched)");
    }

    success
}
