//! Adversarial tests for the verification contract.
//!
//! Each of these encodes a way the old `verified: bool` could be `true` without justifying
//! the answer it was attached to.

use mm_core::{Expr, SymbolTable};
use mm_rules::{Domain, Feature, Rule, RuleApplication, RuleCategory, RuleContext, RuleId};
use mm_verifier::{
    status_from_evidence, StepEvidence, VerificationLevel, VerificationMethod, VerificationStatus,
    Verifier, VerifyResult,
};

/// A rule that claims `x -> x + 1`. It reproduces its own output perfectly and is unsound.
fn broken_rule() -> Rule {
    Rule {
        id: RuleId(90_001),
        name: "off_by_one",
        category: RuleCategory::Simplification,
        description: "x -> x + 1 (deliberately unsound)",
        domains: &[] as &[Domain],
        requires: &[] as &[Feature],
        is_applicable: |e, _| matches!(e, Expr::Var(_)),
        apply: |e, _| {
            vec![RuleApplication {
                result: Expr::Add(Box::new(e.clone()), Box::new(Expr::int(1))),
                justification: "x -> x + 1".to_string(),
            }]
        },
        reversible: false,
        cost: 1,
    }
}

/// An unsound rule over a derivative, which the evaluator cannot sample.
fn broken_calculus_rule() -> Rule {
    Rule {
        id: RuleId(90_002),
        name: "wrong_derivative",
        category: RuleCategory::Derivative,
        description: "d/dx(f) -> 0 for any f (deliberately unsound)",
        domains: &[] as &[Domain],
        requires: &[] as &[Feature],
        is_applicable: |e, _| matches!(e, Expr::Derivative { .. }),
        apply: |_, _| {
            vec![RuleApplication {
                result: Expr::int(0),
                justification: "everything differentiates to zero".to_string(),
            }]
        },
        reversible: false,
        cost: 1,
    }
}

#[test]
fn a_broken_rule_is_not_certified_by_reproducing_its_own_output() {
    let mut symbols = SymbolTable::new();
    let x = symbols.intern("x");
    let rule = broken_rule();
    let ctx = RuleContext::default();

    let before = Expr::Var(x);
    let after = Expr::Add(Box::new(Expr::Var(x)), Box::new(Expr::int(1)));

    // Rerunning the rule reproduces `after` exactly. That must not be enough.
    assert_eq!(rule.apply(&before, &ctx)[0].result, after);

    let result = Verifier::new().verify_step(&before, &after, &rule, &ctx);
    match result {
        VerifyResult::Invalid { .. } => {}
        other => panic!("an unsound rule must be rejected, got {other:?}"),
    }
}

#[test]
fn a_calculus_step_is_labelled_replay_only_not_proved() {
    let mut symbols = SymbolTable::new();
    let x = symbols.intern("x");
    let rule = broken_calculus_rule();
    let ctx = RuleContext::default();

    let before = Expr::Derivative {
        expr: Box::new(Expr::Pow(Box::new(Expr::Var(x)), Box::new(Expr::int(2)))),
        var: x,
    };
    let after = Expr::int(0);

    let result = Verifier::new().verify_step(&before, &after, &rule, &ctx);

    // The verifier cannot sample a derivative, so it accepts the replay. What matters is that
    // it says so: this must not be indistinguishable from a checked equivalence.
    assert_eq!(result.method(), Some(VerificationMethod::RuleReplayOnly));
    assert!(!VerificationMethod::RuleReplayOnly.is_independent());

    // A whole trace of such steps is heuristic, never `Checked`.
    let status = status_from_evidence(&[result.evidence()]);
    assert!(
        !status.is_fully_checked(),
        "rule replay alone must not amount to full verification, got {status}"
    );
    assert!(matches!(status, VerificationStatus::Heuristic { .. }));
}

#[test]
fn formal_mode_reports_that_it_is_unsupported() {
    let mut symbols = SymbolTable::new();
    let x = symbols.intern("x");
    let ctx = RuleContext::default();
    let verifier = Verifier::new().with_level(VerificationLevel::Formal);

    // A rule application that the symbolic checker would happily accept.
    let rule = Rule {
        id: RuleId(90_003),
        name: "identity",
        category: RuleCategory::Simplification,
        description: "x + 0 -> x",
        domains: &[] as &[Domain],
        requires: &[] as &[Feature],
        is_applicable: |e, _| matches!(e, Expr::Add(_, _)),
        apply: |e, _| match e {
            Expr::Add(a, _) => vec![RuleApplication {
                result: (**a).clone(),
                justification: "x + 0 -> x".to_string(),
            }],
            _ => vec![],
        },
        reversible: false,
        cost: 1,
    };

    let before = Expr::Add(Box::new(Expr::Var(x)), Box::new(Expr::int(0)));
    let after = Expr::Var(x);

    match verifier.verify_step(&before, &after, &rule, &ctx) {
        VerifyResult::Unsupported { reason } => {
            assert!(reason.contains("not implemented"));
        }
        other => panic!("formal mode must not claim a verdict it cannot produce, got {other:?}"),
    }
}

#[test]
fn formal_mode_does_not_dress_up_a_calculus_step() {
    let mut symbols = SymbolTable::new();
    let x = symbols.intern("x");
    let ctx = RuleContext::default();
    let verifier = Verifier::new().with_level(VerificationLevel::Formal);

    let before = Expr::Derivative {
        expr: Box::new(Expr::Pow(Box::new(Expr::Var(x)), Box::new(Expr::int(2)))),
        var: x,
    };

    // The calculus escape hatch previously returned `Valid { confidence: 0.95 }` before the
    // level was even consulted, so formal mode silently produced a confident answer here.
    let result = verifier.verify_step(&before, &Expr::int(0), &broken_calculus_rule(), &ctx);
    assert!(
        matches!(result, VerifyResult::Unsupported { .. }),
        "formal mode must stay unsupported for calculus too, got {result:?}"
    );
}

#[test]
fn an_unchecked_step_sinks_the_whole_result() {
    let evidence = [
        StepEvidence::Checked(VerificationMethod::SymbolicEquivalence),
        StepEvidence::Checked(VerificationMethod::SymbolicEquivalence),
        StepEvidence::Unchecked,
    ];
    let status = status_from_evidence(&evidence);

    assert!(!status.is_fully_checked());
    assert!(matches!(status, VerificationStatus::Unverified { .. }));
}

#[test]
fn independent_equivalence_can_justify_a_non_rule_step() {
    // Post-processing such as constant folding is not a registry rule, so it has to be
    // checked on its own terms rather than assumed sound.
    let verifier = Verifier::new();
    let before = Expr::Add(Box::new(Expr::int(2)), Box::new(Expr::int(3)));

    let good = verifier.verify_equivalence(&before, &Expr::int(5));
    assert_eq!(good.method(), Some(VerificationMethod::SymbolicEquivalence));

    let bad = verifier.verify_equivalence(&before, &Expr::int(6));
    assert!(
        matches!(bad, VerifyResult::Invalid { .. }),
        "a fold that changes the value must be rejected, got {bad:?}"
    );
    assert_eq!(bad.evidence(), StepEvidence::Unchecked);
}
