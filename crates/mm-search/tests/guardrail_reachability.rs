//! The guardrail must not hide rules that work.
//!
//! `NeuralMCTS::expand` filters the registry through `mm_boink::filter_rules` before trying
//! anything. A rule the filter never offers is unreachable by search however well it works,
//! and nothing in the type system says so.
//!
//! The filter's decision comes from `mm_boink::analyze`, which walks the expression setting
//! domain flags. A scanner that fails to recurse into some node kind reports "no
//! trigonometry" for `sin(x)^2 + cos(x)^2` and every trigonometric rule then disappears.
//! `mm_rules::guardrail` had exactly that defect: it recursed only through `Add`, `Sub`,
//! `Mul` and `Div`, so anything under a `Pow` was invisible.
//!
//! This test pins the property against the analyser the search actually uses.

use std::collections::HashSet;

use mm_boink::{analyze, filter_rules};
use mm_core::Expr;
use mm_rules::{corpus, standard_rules, RuleContext, WitnessSymbols};

/// Rules that work but are never offered, pinned.
///
/// Not zero, and not a regression introduced here: these rules carry a `domains` tag that the
/// analyser never derives for the expressions they match.
///
/// Two patterns account for all of them:
///
/// - Absolute-value manipulation (`inequalities::abs_product`, `abs_neg`, `abs_abs`, ...) is
///   tagged `Domain::Inequalities`, but `analyze` sets `has_inequalities` only for `Lt`,
///   `Lte`, `Gt` and `Gte`. `|a * b|` is not an inequality, so the tag never matches.
/// - Algebraic factoring and factorial identities (`number_theory::diff_squares_factor`,
///   `square_binomial_expand`, `factorial_zero`, ...) are tagged `Domain::NumberTheory`, but
///   `Factorial` sets `has_combinatorics` and a plain `a^2 - b^2` sets no number-theory flag.
///
/// The fix is to re-tag the rules rather than to loosen the filter, since loosening it
/// reintroduces exactly the cross-domain noise the guardrail exists to prevent. That is a
/// behavioural change to rule availability and belongs in its own change with its own
/// evaluation run, so this test measures the set instead of asserting it away.
const EXPECTED_HIDDEN: usize = 32;

fn hidden_rules() -> Vec<String> {
    let rules = standard_rules();
    let symbols = WitnessSymbols::new();
    let witnesses = corpus(&symbols);
    let ctx = RuleContext::default();

    // Rules the guardrail offers, per witness.
    let offered: Vec<HashSet<u32>> = witnesses
        .iter()
        .map(|w| {
            filter_rules(rules.all(), &analyze(w))
                .into_iter()
                .map(|r| r.id.0)
                .collect()
        })
        .collect();

    let mut hidden = Vec::new();

    for (rule, key) in rules.all().iter().zip(rules.keys()) {
        let mut transforms_anything = false;
        let mut ever_offered = false;

        for (index, witness) in witnesses.iter().enumerate() {
            if !rule.can_apply(witness, &ctx) {
                continue;
            }
            let changes = rule
                .apply(witness, &ctx)
                .iter()
                .any(|app| app.result != *witness);
            if !changes {
                continue;
            }
            transforms_anything = true;
            if offered[index].contains(&rule.id.0) {
                ever_offered = true;
                break;
            }
        }

        if transforms_anything && !ever_offered {
            hidden.push(key.to_string());
        }
    }

    hidden
}

#[test]
fn the_set_of_guardrail_hidden_rules_is_pinned() {
    let hidden = hidden_rules();

    println!(
        "\n{} rule(s) transform an expression but the guardrail never offers them for it:",
        hidden.len()
    );
    for name in &hidden {
        println!("  {name}");
    }

    assert_eq!(
        hidden.len(),
        EXPECTED_HIDDEN,
        "the number of working-but-hidden rules changed; if it went down, update the pin, \
         and if it went up, a rule was just made unreachable"
    );
}

#[test]
fn trigonometric_rules_survive_a_squared_argument() {
    // The specific shape that exposed the incomplete scanner. `sin(x)^2 + cos(x)^2` puts the
    // trig functions under a `Pow`; a scanner that stops at `Pow` reports no trigonometry.
    let symbols = WitnessSymbols::new();
    let x = Expr::Var(symbols.x);
    let squared = |f: Expr| Expr::Pow(Box::new(f), Box::new(Expr::int(2)));
    let expr = Expr::Add(
        Box::new(squared(Expr::Sin(Box::new(x.clone())))),
        Box::new(squared(Expr::Cos(Box::new(x)))),
    );

    let profile = analyze(&expr);
    assert!(
        profile.has_trig,
        "a squared trig function must still count as trigonometry"
    );

    let rules = standard_rules();
    let offered: HashSet<u32> = filter_rules(rules.all(), &profile)
        .into_iter()
        .map(|r| r.id.0)
        .collect();

    let pythagorean = rules
        .all()
        .iter()
        .zip(rules.keys())
        .find(|(_, key)| key.module == "trig" && key.name == "pythagorean_identity")
        .map(|(rule, _)| rule)
        .expect("trig::pythagorean_identity is registered");

    assert!(
        offered.contains(&pythagorean.id.0),
        "the guardrail must offer the rule that matches this expression"
    );
}

#[test]
fn calculus_rules_survive_nesting() {
    // The same failure mode for the other domain flag: a derivative inside a product.
    let symbols = WitnessSymbols::new();
    let x = Expr::Var(symbols.x);
    let expr = Expr::Mul(
        Box::new(Expr::int(2)),
        Box::new(Expr::Derivative {
            expr: Box::new(Expr::Pow(Box::new(x), Box::new(Expr::int(2)))),
            var: symbols.x,
        }),
    );

    let profile = analyze(&expr);
    assert!(
        profile.has_calculus_diff,
        "a nested derivative must still count as calculus"
    );
}
