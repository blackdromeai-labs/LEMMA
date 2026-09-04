//! Equation rewrites are checked against the solution set, not just the expression value.
//!
//! `verify_step` establishes that a transformation preserves *value*: the two sides must
//! agree symbolically, or agree at sampled points. That is the right criterion for rewriting
//! a term. It used to be applied unchanged to equations too, where the property to preserve
//! is the solution set, not the expression value -- `2x = 10` and `x = 10/2` have the same
//! solution, but as expressions they are not equal: canonicalisation does not relate them
//! and sampling `x` gives different values on each side. That made the verifier refuse a
//! correct step from `equations::cancel_multiplication`, and `2x = 10 -> x = 5` was listed as
//! unsolved in `mm-solver`'s evaluation suite as a result.
//!
//! `verify_step` and `verify_equivalence` now try an equation-specific check
//! (`mm_verifier::numerical::verify_equation_equivalent`) whenever both sides of the step are
//! equations: it samples the ratio between `lhs - rhs` of each equation and accepts the step
//! when that ratio is the same nonzero constant at every sample, which is what "add the same
//! term to both sides" or "scale both sides by a nonzero constant" actually preserves.
//!
//! These tests pin the current behaviour so a regression here is caught immediately.

use mm_core::{Expr, SymbolTable};
use mm_rules::rule::RuleKey;
use mm_rules::{standard_rules, RuleContext, RuleSet};
use mm_verifier::{VerificationMethod, Verifier, VerifyResult};

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
fn dividing_both_sides_is_produced_correctly_and_accepted() {
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

    // The verifier now recognises this as the same equation's solution set, not the same
    // expression value.
    let result = Verifier::new().verify_step(&before, after, rule, &ctx);
    match result {
        VerifyResult::Valid {
            method: VerificationMethod::NumericSampling,
            ..
        } => {}
        other => {
            panic!("expected the equation-aware numeric check to accept this step, got {other:?}")
        }
    }
}

#[test]
fn adding_to_both_sides_is_accepted() {
    // `x + 3 = 7 -> x = 4` was always accepted, but only by accident: adding the same term to
    // both sides happens to leave `lhs - rhs` literally unchanged, so it passed under plain
    // expression semantics even before the equation-aware check existed. It still passes now,
    // this time because the ratio check also holds (ratio 1).
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
