//! Known limitation: equation rewrites are checked with expression semantics.
//!
//! `verify_step` establishes that a transformation preserves *value*: the two sides must
//! agree symbolically, or agree at sampled points. That is the right criterion for rewriting
//! a term. It is the wrong criterion for rewriting an equation, where the property to
//! preserve is the solution set.
//!
//! `2x = 10` and `x = 10/2` have the same solution, but as expressions they are not equal:
//! canonicalisation does not relate them and sampling `x` gives different values on each
//! side. So the verifier refuses the step, the search never gets the node, and dividing both
//! sides of an equation is unavailable in practice even though `equations::cancel_
//! multiplication` implements it correctly.
//!
//! This is why `2x = 10 -> x = 5` and `3x + 5 = 17 -> x = 4` are listed as unsolved in
//! `mm-solver`'s evaluation suite. The rules exist and work; the check rejects them.
//!
//! These tests pin the current behaviour so it cannot change silently. When equation-aware
//! verification is implemented they will fail, which is the intended signal: the evaluation
//! suite's `KNOWN_UNSOLVED` list should be revisited at the same time.

use mm_core::{Expr, SymbolTable};
use mm_rules::rule::RuleKey;
use mm_rules::{standard_rules, RuleContext, RuleSet};
use mm_verifier::{Verifier, VerifyResult};

fn equations_rule<'a>(rules: &'a RuleSet, name: &str) -> &'a mm_rules::Rule {
    rules
        .all()
        .iter()
        .zip(rules.keys())
        .find(|(_, key)| key.module == "equations" && key.name == name)
        .map(|(rule, _)| rule)
        .unwrap_or_else(|| panic!("equations::{name} is not registered"))
}

#[test]
fn dividing_both_sides_is_produced_correctly_but_refused() {
    let mut symbols = SymbolTable::new();
    let x = symbols.intern("x");
    let rules = standard_rules();
    let ctx = RuleContext::default();
    let rule = equations_rule(&rules, "cancel_multiplication");

    // 2x = 10
    let before = Expr::Equation {
        lhs: Box::new(Expr::Mul(Box::new(Expr::int(2)), Box::new(Expr::Var(x)))),
        rhs: Box::new(Expr::int(10)),
    };

    let applications = rule.apply(&before, &ctx);
    assert_eq!(
        applications.len(),
        1,
        "the rule should offer exactly one rewrite"
    );

    // The rule does the right thing: x = 10/2.
    let after = &applications[0].result;
    assert_eq!(
        *after,
        Expr::Equation {
            lhs: Box::new(Expr::Var(x)),
            rhs: Box::new(Expr::Div(Box::new(Expr::int(10)), Box::new(Expr::int(2)))),
        }
    );

    // And the verifier refuses it, because the two equations are not the same expression.
    let result = Verifier::new().verify_step(&before, after, rule, &ctx);
    match result {
        VerifyResult::Invalid { .. } => {}
        other => panic!(
            "equation-aware verification appears to have been implemented (got {other:?}). \
             Revisit KNOWN_UNSOLVED in crates/mm-solver/tests/evaluation.rs."
        ),
    }
}

#[test]
fn adding_to_both_sides_is_accepted_only_because_it_happens_to_balance() {
    // `x + 3 = 7 -> x = 4` is accepted, which is why that case does solve. It is not accepted
    // because the verifier understands equations: it is accepted because the rule's output
    // compares equal under the same expression semantics that reject the division case. The
    // asymmetry is the bug, not the acceptance.
    let mut symbols = SymbolTable::new();
    let x = symbols.intern("x");
    let rules = standard_rules();
    let ctx = RuleContext::default();
    let rule = equations_rule(&rules, "cancel_addition");

    let before = Expr::Equation {
        lhs: Box::new(Expr::Add(Box::new(Expr::Var(x)), Box::new(Expr::int(3)))),
        rhs: Box::new(Expr::int(7)),
    };

    let applications = rule.apply(&before, &ctx);
    assert!(!applications.is_empty());

    let accepted = applications.iter().any(|app| {
        Verifier::new()
            .verify_step(&before, &app.result, rule, &ctx)
            .is_valid()
    });
    assert!(
        accepted,
        "cancel_addition is expected to pass today; if it stopped, equation solving regressed"
    );
}

#[test]
fn expression_semantics_are_still_correct_for_term_rewrites() {
    // The criterion itself is not wrong, only misapplied to equations. A term rewrite that
    // preserves value must still be accepted.
    let mut symbols = SymbolTable::new();
    let x = symbols.intern("x");
    let rules = standard_rules();
    let ctx = RuleContext::default();

    let rule = rules
        .get_by_key(&RuleKey {
            module: "algebra",
            name: "identity_add_zero",
        })
        .expect("algebra::identity_add_zero is registered");

    let before = Expr::Add(Box::new(Expr::Var(x)), Box::new(Expr::int(0)));
    let after = Expr::Var(x);

    assert!(Verifier::new()
        .verify_step(&before, &after, rule, &ctx)
        .is_valid());
}
