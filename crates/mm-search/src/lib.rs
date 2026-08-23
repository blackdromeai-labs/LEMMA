// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Author: Pushp Kharat

//! # mm-search
//!
//! Search algorithms for finding solution paths through the mathematical
//! transformation space.
//!
//! This crate provides:
//! - [`BeamSearch`] - A simple beam search algorithm
//! - [`NeuralMCTS`] - Monte Carlo Tree Search over rule applications
//! - [`DeepMCTS`] - Parallel tree search for large node budgets
//!
//! ## Solution integrity
//!
//! A [`Solution`] carries a [`VerificationStatus`], not a boolean. The status is derived from
//! the recorded trace by [`Solution::assess`]: the steps must chain from the exact problem to
//! the exact reported result, and each one must carry the evidence the verifier produced for
//! it. Any transformation applied without a check — including post-processing such as
//! constant folding — either records a checked step or downgrades the status.

pub mod beam;
pub mod boink_mcts;
pub mod bridge;
pub mod deep_mcts;
pub mod mcts;

use mm_core::Expr;
use mm_rules::RuleId;
use mm_verifier::{status_from_evidence, StepEvidence, VerificationStatus};

/// Identifier recorded for a step that is not a registry rule application.
///
/// No rule uses identifier 0; normalisation steps (canonicalisation, constant folding) use it
/// so that a reader can tell them apart from rule applications at a glance.
pub const NON_RULE_ID: RuleId = RuleId(0);

/// A step in a solution path.
#[derive(Debug, Clone)]
pub struct Step {
    /// The expression before this step.
    pub before: Expr,
    /// The expression after this step.
    pub after: Expr,
    /// The rule that was applied, or [`NON_RULE_ID`] for a normalisation step.
    pub rule_id: RuleId,
    /// The rule name, or the name of the normalisation.
    pub rule_name: &'static str,
    /// Justification for this step.
    pub justification: String,
    /// What the verifier established about this transition.
    pub evidence: StepEvidence,
}

impl Step {
    /// Record a checked or unchecked rule application.
    pub fn rule(
        before: Expr,
        after: Expr,
        rule_id: RuleId,
        rule_name: &'static str,
        justification: String,
        evidence: StepEvidence,
    ) -> Self {
        Self {
            before,
            after,
            rule_id,
            rule_name,
            justification,
            evidence,
        }
    }

    /// Record a normalisation step that is not a registry rule.
    pub fn normalization(
        before: Expr,
        after: Expr,
        name: &'static str,
        justification: String,
        evidence: StepEvidence,
    ) -> Self {
        Self {
            before,
            after,
            rule_id: NON_RULE_ID,
            rule_name: name,
            justification,
            evidence,
        }
    }

    /// Whether this step came from a registry rule.
    pub fn is_rule_application(&self) -> bool {
        self.rule_id != NON_RULE_ID
    }
}

/// A complete solution.
#[derive(Debug, Clone)]
pub struct Solution {
    /// The original problem.
    pub problem: Expr,
    /// The final result.
    pub result: Expr,
    /// The steps taken.
    pub steps: Vec<Step>,
    /// What is known about the result.
    pub status: VerificationStatus,
}

impl Solution {
    /// Build a solution and derive its status from the recorded trace.
    ///
    /// This is the only constructor that should be used by search code: it makes the status a
    /// function of the evidence rather than something a caller can assert.
    pub fn assess(problem: Expr, result: Expr, steps: Vec<Step>) -> Self {
        let status = assess_trace(&problem, &result, &steps);
        Self {
            problem,
            result,
            steps,
            status,
        }
    }

    /// Build a solution with an explicitly weaker status, keeping the stronger of the two.
    ///
    /// Used when the caller knows something the trace does not show, for example that the
    /// search stopped short of the goal.
    pub fn assess_at_most(
        problem: Expr,
        result: Expr,
        steps: Vec<Step>,
        ceiling: VerificationStatus,
    ) -> Self {
        let derived = assess_trace(&problem, &result, &steps);
        let status = if derived.is_fully_checked() {
            ceiling
        } else {
            derived
        };
        Self {
            problem,
            result,
            steps,
            status,
        }
    }

    /// Get the number of steps in this solution.
    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }

    /// Check if this solution is empty (no steps needed).
    pub fn is_trivial(&self) -> bool {
        self.steps.is_empty()
    }

    /// Whether the trace replays and every step was independently checked.
    pub fn is_fully_verified(&self) -> bool {
        self.status.is_fully_checked()
    }
}

/// Replay a trace and combine its evidence into a status.
///
/// The trace must start at `problem`, chain step by step, and end at `result`. If it does
/// not, the result is [`VerificationStatus::Unverified`] whatever the individual steps claim:
/// a checked step that leads somewhere other than the reported answer proves nothing about
/// that answer.
pub fn assess_trace(problem: &Expr, result: &Expr, steps: &[Step]) -> VerificationStatus {
    if steps.is_empty() {
        return if expressions_identical(problem, result) {
            VerificationStatus::Checked
        } else {
            VerificationStatus::Unverified {
                reason: "result differs from the input but no steps were recorded".to_string(),
            }
        };
    }

    if !expressions_identical(&steps[0].before, problem) {
        return VerificationStatus::Unverified {
            reason: "trace does not start at the input expression".to_string(),
        };
    }

    for (i, pair) in steps.windows(2).enumerate() {
        if !expressions_identical(&pair[0].after, &pair[1].before) {
            return VerificationStatus::Unverified {
                reason: format!("trace breaks between step {} and step {}", i + 1, i + 2),
            };
        }
    }

    if !expressions_identical(&steps[steps.len() - 1].after, result) {
        return VerificationStatus::Unverified {
            reason: "trace does not end at the reported result".to_string(),
        };
    }

    let evidence: Vec<StepEvidence> = steps.iter().map(|s| s.evidence).collect();
    status_from_evidence(&evidence)
}

/// Exact structural equality.
///
/// Deliberately not canonical or numeric equality: the replay invariant is that the recorded
/// steps produce the reported expression, so anything weaker would let post-processing change
/// the answer unnoticed.
fn expressions_identical(a: &Expr, b: &Expr) -> bool {
    a == b
}

/// A predicate for checking if a goal has been reached.
pub trait GoalPredicate: Fn(&Expr) -> bool {}
impl<F: Fn(&Expr) -> bool> GoalPredicate for F {}

/// Search configuration.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Maximum search depth.
    pub max_depth: usize,
    /// Beam width for beam search.
    pub beam_width: usize,
    /// Number of MCTS iterations.
    pub mcts_iterations: usize,
    /// Exploration weight for UCB.
    pub exploration_weight: f64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_depth: 20,
            beam_width: 10,
            mcts_iterations: 1000,
            exploration_weight: 1.41,
        }
    }
}

pub use beam::BeamSearch;
pub use boink_mcts::{BoinkMCTS, BoinkStats};
pub use deep_mcts::{DeepMCTS, DeepMCTSConfig, DeepNode, SearchStats};
pub use mcts::{MCTSConfig, MCTSNode, NeuralMCTS};

#[cfg(test)]
mod tests {
    use super::*;
    use mm_verifier::VerificationMethod;

    fn checked_step(before: Expr, after: Expr) -> Step {
        Step::rule(
            before,
            after,
            RuleId(1),
            "test_rule",
            "test".to_string(),
            StepEvidence::Checked(VerificationMethod::SymbolicEquivalence),
        )
    }

    #[test]
    fn empty_trace_with_unchanged_result_is_checked() {
        let e = Expr::int(5);
        assert_eq!(assess_trace(&e, &e, &[]), VerificationStatus::Checked);
    }

    #[test]
    fn untraced_change_cannot_be_checked() {
        // This is the untraced-final-fold case: the answer changed with nothing to show why.
        let status = assess_trace(&Expr::int(5), &Expr::int(6), &[]);
        assert!(matches!(status, VerificationStatus::Unverified { .. }));
    }

    #[test]
    fn a_trace_that_ends_elsewhere_is_unverified() {
        let steps = vec![checked_step(Expr::int(1), Expr::int(2))];
        let status = assess_trace(&Expr::int(1), &Expr::int(3), &steps);
        assert!(matches!(status, VerificationStatus::Unverified { .. }));
    }

    #[test]
    fn a_broken_chain_is_unverified() {
        let steps = vec![
            checked_step(Expr::int(1), Expr::int(2)),
            checked_step(Expr::int(9), Expr::int(3)),
        ];
        let status = assess_trace(&Expr::int(1), &Expr::int(3), &steps);
        assert!(matches!(status, VerificationStatus::Unverified { .. }));
    }

    #[test]
    fn a_trace_that_starts_elsewhere_is_unverified() {
        let steps = vec![checked_step(Expr::int(7), Expr::int(3))];
        let status = assess_trace(&Expr::int(1), &Expr::int(3), &steps);
        assert!(matches!(status, VerificationStatus::Unverified { .. }));
    }

    #[test]
    fn a_complete_checked_chain_is_checked() {
        let steps = vec![
            checked_step(Expr::int(1), Expr::int(2)),
            checked_step(Expr::int(2), Expr::int(3)),
        ];
        let solution = Solution::assess(Expr::int(1), Expr::int(3), steps);
        assert!(solution.is_fully_verified());
    }

    #[test]
    fn one_unchecked_step_prevents_full_verification() {
        let steps = vec![
            checked_step(Expr::int(1), Expr::int(2)),
            Step::rule(
                Expr::int(2),
                Expr::int(3),
                RuleId(1),
                "test_rule",
                "test".to_string(),
                StepEvidence::Unchecked,
            ),
        ];
        let solution = Solution::assess(Expr::int(1), Expr::int(3), steps);
        assert!(!solution.is_fully_verified());
    }
}
