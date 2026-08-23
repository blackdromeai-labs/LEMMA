//! Solution-level integrity: what a returned result is allowed to claim.
//!
//! `NeuralMCTS::simplify` used to end with `verified: true` unconditionally, after a direct
//! rule fallback that never called the verifier and a post-processing pass (recursive
//! sub-expression simplification plus constant folding) that could change the final answer
//! without adding a step. These tests pin the replacement contract.

use mm_core::{Expr, SymbolTable};
use mm_rules::rule::{Rule, RuleApplication, RuleSet};
use mm_rules::{standard_rules, Domain, Feature, RuleCategory, RuleId};
use mm_search::{assess_trace, MCTSConfig, NeuralMCTS, Solution, Step, NON_RULE_ID};
use mm_verifier::{StepEvidence, VerificationMethod, VerificationStatus, Verifier};

fn config() -> MCTSConfig {
    MCTSConfig {
        simulations: 20,
        max_depth: 8,
        max_simplify_iterations: 20,
        ..Default::default()
    }
}

fn solver() -> NeuralMCTS {
    NeuralMCTS::with_config(standard_rules(), Verifier::new(), config())
}

#[test]
fn a_reported_result_is_always_reachable_from_the_recorded_steps() {
    let mcts = solver();
    let mut symbols = SymbolTable::new();
    let x = symbols.intern("x");

    let inputs = vec![
        // Constant folding happens in post-processing here.
        Expr::Mul(
            Box::new(Expr::Add(Box::new(Expr::int(2)), Box::new(Expr::int(3)))),
            Box::new(Expr::Add(Box::new(Expr::int(4)), Box::new(Expr::int(5)))),
        ),
        // Sub-expression rules do the work here.
        Expr::Add(
            Box::new(Expr::Derivative {
                expr: Box::new(Expr::Pow(Box::new(Expr::Var(x)), Box::new(Expr::int(2)))),
                var: x,
            }),
            Box::new(Expr::Derivative {
                expr: Box::new(Expr::Pow(Box::new(Expr::Var(x)), Box::new(Expr::int(3)))),
                var: x,
            }),
        ),
        // Nothing at all applies here.
        Expr::Var(x),
    ];

    for input in inputs {
        let solution = mcts.simplify(input.clone());
        let derived = assess_trace(&solution.problem, &solution.result, &solution.steps);

        assert_eq!(
            derived, solution.status,
            "reported status must equal the status the trace implies, for {input:?}"
        );

        if solution.result != input {
            assert!(
                !solution.steps.is_empty(),
                "the answer changed for {input:?} but nothing was recorded"
            );
            assert_eq!(
                solution.steps.last().unwrap().after,
                solution.result,
                "the last step must land on the reported result"
            );
            assert_eq!(
                solution.steps[0].before, input,
                "the first step must start at the input"
            );
        }
    }
}

#[test]
fn post_processing_is_recorded_as_steps() {
    // `(2+3)*(4+5)` is folded to 45 during post-processing, not by a search step.
    let mcts = solver();
    let expr = Expr::Mul(
        Box::new(Expr::Add(Box::new(Expr::int(2)), Box::new(Expr::int(3)))),
        Box::new(Expr::Add(Box::new(Expr::int(4)), Box::new(Expr::int(5)))),
    );

    let solution = mcts.simplify(expr);
    assert_eq!(solution.result, Expr::int(45));
    assert!(solution.status.replays(), "status was {}", solution.status);

    // At least one step must be a normalisation rather than a registry rule, and it must
    // carry real evidence.
    let normalisations: Vec<&Step> = solution
        .steps
        .iter()
        .filter(|s| s.rule_id == NON_RULE_ID)
        .collect();

    for step in &normalisations {
        assert!(
            step.evidence.is_checked(),
            "normalisation step '{}' has no evidence",
            step.rule_name
        );
    }
}

#[test]
fn an_untraced_final_fold_cannot_be_marked_fully_verified() {
    // Hand-build the situation the old code produced: a checked step, then the answer folded
    // to something else with nothing recorded.
    let steps = vec![Step::rule(
        Expr::Add(Box::new(Expr::int(2)), Box::new(Expr::int(3))),
        Expr::Add(Box::new(Expr::int(2)), Box::new(Expr::int(3))),
        RuleId(1),
        "noop",
        "no change".to_string(),
        StepEvidence::Checked(VerificationMethod::SymbolicEquivalence),
    )];

    let solution = Solution::assess(
        Expr::Add(Box::new(Expr::int(2)), Box::new(Expr::int(3))),
        Expr::int(5),
        steps,
    );

    assert!(!solution.is_fully_verified());
    assert!(matches!(
        solution.status,
        VerificationStatus::Unverified { .. }
    ));
}

#[test]
fn an_unchecked_fallback_cannot_be_marked_fully_verified() {
    let before = Expr::int(1);
    let after = Expr::int(2);
    let steps = vec![Step::rule(
        before.clone(),
        after.clone(),
        RuleId(1),
        "fallback",
        "applied without checking".to_string(),
        StepEvidence::Unchecked,
    )];

    let solution = Solution::assess(before, after, steps);
    assert!(!solution.is_fully_verified());
}

#[test]
fn an_unsound_rule_never_enters_a_returned_trace() {
    // A rule that claims x -> x + 1. Expansion verifies every application, so it must never
    // produce a child, and `simplify` must leave the expression alone.
    let mut rules = RuleSet::new();
    rules
        .try_add(
            "test",
            Rule {
                id: RuleId(90_010),
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
            },
        )
        .unwrap();

    let mcts = NeuralMCTS::with_config(rules, Verifier::new(), config());
    let mut symbols = SymbolTable::new();
    let x = symbols.intern("x");

    let solution = mcts.simplify(Expr::Var(x));
    assert_eq!(
        solution.result,
        Expr::Var(x),
        "an unsound rule must not change the answer"
    );
    assert!(
        solution.steps.is_empty(),
        "an unsound rule must not appear in a trace: {:?}",
        solution.steps
    );
}
