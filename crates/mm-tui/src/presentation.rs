// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Author: Pushp Kharat

//! Owned presentation values, and the honest mapping from domain state onto them.
//!
//! Everything here is `String`-based and self-contained. `Symbol` values are indices into the
//! `SymbolTable` that produced them, so an `Expr` cannot be rendered after that table is out
//! of scope. The worker builds these values while its solver is alive and sends them to the
//! render thread, which never touches a domain type.
//!
//! The status mapping is the part that has to stay honest. `mm-verifier` distinguishes a
//! replayed trace with independent checks from one resting on numeric sampling or on rule
//! replay alone, and the UI must not flatten those back into a checkmark.

use std::time::Duration;

use mm_verifier::{StepEvidence, VerificationMethod, VerificationStatus, VerifyResult};

/// The three supported operations.
///
/// `solve_for` and natural-language solving are deliberately absent: the underlying APIs
/// report them as unimplemented and unsupported, and a menu entry would imply otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mode {
    /// Simplify an expression.
    #[default]
    Simplify,
    /// Differentiate an expression with respect to a variable.
    Differentiate,
    /// Check whether a candidate value satisfies an equation.
    VerifyCandidate,
}

impl Mode {
    /// All modes, in selector order.
    pub const ALL: [Mode; 3] = [Mode::Simplify, Mode::Differentiate, Mode::VerifyCandidate];

    /// Short label for the operation selector.
    pub fn label(self) -> &'static str {
        match self {
            Mode::Simplify => "Simplify",
            Mode::Differentiate => "Differentiate",
            Mode::VerifyCandidate => "Verify candidate",
        }
    }

    /// Whether this mode reads the variable field.
    pub fn uses_variable(self) -> bool {
        matches!(self, Mode::Differentiate | Mode::VerifyCandidate)
    }

    /// Whether this mode reads the candidate field.
    pub fn uses_candidate(self) -> bool {
        matches!(self, Mode::VerifyCandidate)
    }

    /// One-line description of what the mode expects, shown under the form.
    pub fn hint(self) -> &'static str {
        match self {
            Mode::Simplify => "Example: (x + 0) * 1, sin(x)^2 + cos(x)^2, 2 * (3 + 4)",
            Mode::Differentiate => "Example: x^3 with variable x, or sin(x) with variable x",
            Mode::VerifyCandidate => "Example: x + 3 = 7 with variable x and candidate 4",
        }
    }
}

/// Verification badge shown beside a result.
///
/// One variant per domain state plus the two UI-only states the domain has no word for: a
/// bounded search that returned nothing, and input the parser rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusBadge {
    /// Trace replays end to end and every step was independently checked.
    Checked,
    /// Trace replays, evidence is mixed strength.
    Partial,
    /// Trace replays on sampling or rule replay only.
    Heuristic,
    /// No accepted checked path from the input to the output.
    Unverified,
    /// The requested verification is not implemented.
    Unsupported,
    /// Bounded search found nothing. Not a proof that nothing exists.
    NotFound,
    /// The input could not be parsed.
    InputError,
    /// A candidate satisfies its equation.
    CandidateValid,
    /// A candidate does not satisfy its equation.
    CandidateInvalid,
}

impl StatusBadge {
    /// Uppercase text label. Always rendered, so colour is never the only cue.
    pub fn label(self) -> &'static str {
        match self {
            StatusBadge::Checked => "CHECKED",
            StatusBadge::Partial => "PARTIAL",
            StatusBadge::Heuristic => "HEURISTIC",
            StatusBadge::Unverified => "UNVERIFIED",
            StatusBadge::Unsupported => "UNSUPPORTED",
            StatusBadge::NotFound => "NOT FOUND",
            StatusBadge::InputError => "INPUT ERROR",
            StatusBadge::CandidateValid => "SATISFIED",
            StatusBadge::CandidateInvalid => "NOT SATISFIED",
        }
    }

    /// A leading glyph, so the badge survives a monochrome terminal or a screen reader.
    pub fn marker(self) -> &'static str {
        match self {
            StatusBadge::Checked | StatusBadge::CandidateValid => "++",
            StatusBadge::Partial => "+-",
            StatusBadge::Heuristic => "~ ",
            StatusBadge::Unverified | StatusBadge::CandidateInvalid => "!!",
            StatusBadge::Unsupported => "--",
            StatusBadge::NotFound => "..",
            StatusBadge::InputError => "XX",
        }
    }

    /// Whether this state may be described as verified.
    ///
    /// Only [`StatusBadge::Checked`] qualifies. Partial, heuristic and sampling-backed results
    /// are evidence, not proof.
    pub fn is_verified(self) -> bool {
        matches!(self, StatusBadge::Checked)
    }
}

/// How a single recorded step was established.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UiEvidence {
    /// Short label, for example `checked (symbolic equivalence)`.
    pub label: String,
    /// Whether the step was checked at all.
    pub checked: bool,
    /// Whether the check was independent of re-running the rule.
    pub independent: bool,
}

impl UiEvidence {
    /// Describe a step's evidence.
    pub fn from_step(evidence: StepEvidence) -> Self {
        match evidence {
            StepEvidence::Checked(method) => Self {
                label: format!("checked ({})", describe_method(method)),
                checked: true,
                independent: method.is_independent(),
            },
            StepEvidence::Unchecked => Self {
                label: "unchecked".to_string(),
                checked: false,
                independent: false,
            },
        }
    }
}

fn describe_method(method: VerificationMethod) -> &'static str {
    match method {
        VerificationMethod::SymbolicEquivalence => "symbolic equivalence",
        VerificationMethod::NumericSampling => "numeric sampling",
        VerificationMethod::RuleReplayOnly => "rule replay only",
    }
}

/// One recorded transformation, fully rendered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UiStep {
    /// 1-based position in the trace.
    pub index: usize,
    /// Expression before the step.
    pub before: String,
    /// Expression after the step.
    pub after: String,
    /// Stable rule identity, for example `algebra::identity_add_zero`, or the name of a
    /// normalisation for steps that are not registry rules.
    pub rule: String,
    /// Numeric rule identifier, absent for normalisation steps.
    pub rule_id: Option<u32>,
    /// The rule's own justification text.
    pub justification: String,
    /// What the verifier established for this step.
    pub evidence: UiEvidence,
}

impl UiStep {
    /// The compact form shown in the trace list.
    pub fn summary(&self) -> String {
        let method = if self.evidence.checked {
            if self.evidence.independent {
                "SYMBOLIC"
            } else {
                "REPLAY"
            }
        } else {
            "UNCHECKED"
        };
        format!("{:02} {} {}", self.index, self.rule, method)
    }
}

/// A completed request, ready to render.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UiResult {
    /// Operation that produced this.
    pub mode: Mode,
    /// The input expression, re-rendered by the formatter so it matches the trace.
    pub input: String,
    /// The final expression, or the verdict text for a candidate check.
    pub output: String,
    /// Verification badge.
    pub badge: StatusBadge,
    /// Full reason from the domain status, or the UI's own explanation.
    pub reason: String,
    /// Recorded transformations, in order.
    pub steps: Vec<UiStep>,
    /// Wall-clock time the worker spent on the request.
    pub elapsed: Duration,
}

impl UiResult {
    /// One-line summary used by the history drawer.
    pub fn history_summary(&self) -> String {
        format!(
            "{} · {} · {}",
            self.mode.label(),
            self.badge.label(),
            self.output
        )
    }
}

/// A request that could not run.
///
/// Kept separate from a completed-but-unverified result: a parse failure is an input problem,
/// not a statement about mathematics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UiError {
    /// Which field the problem belongs to, when it is attributable.
    pub field: ErrorField,
    /// Message shown to the user.
    pub message: String,
}

/// The field an input error belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorField {
    /// The expression editor.
    Expression,
    /// The variable editor.
    Variable,
    /// The candidate editor.
    Candidate,
    /// Not attributable to one field.
    Request,
}

impl ErrorField {
    /// Label used when reporting the error.
    pub fn label(self) -> &'static str {
        match self {
            ErrorField::Expression => "expression",
            ErrorField::Variable => "variable",
            ErrorField::Candidate => "candidate",
            ErrorField::Request => "request",
        }
    }
}

/// Map a domain [`VerificationStatus`] onto a badge and its reason.
///
/// The reason is taken from the domain wherever the domain supplies one, so the UI cannot
/// drift from what the verifier actually said.
pub fn badge_for_status(status: &VerificationStatus) -> (StatusBadge, String) {
    match status {
        VerificationStatus::Checked => (
            StatusBadge::Checked,
            "Trace replays from the input to the result and every step was independently \
             checked."
                .to_string(),
        ),
        VerificationStatus::Partial { reason } => (StatusBadge::Partial, reason.clone()),
        VerificationStatus::Heuristic { reason } => (StatusBadge::Heuristic, reason.clone()),
        VerificationStatus::Unverified { reason } => (StatusBadge::Unverified, reason.clone()),
        VerificationStatus::Unsupported { reason } => (StatusBadge::Unsupported, reason.clone()),
    }
}

/// Map a domain [`VerifyResult`] onto a badge and its reason.
pub fn badge_for_verify_result(result: &VerifyResult) -> (StatusBadge, String) {
    match result {
        VerifyResult::Valid { confidence, method } => (
            StatusBadge::CandidateValid,
            format!(
                "Substituting the candidate satisfies the equation, established by {} \
                 (confidence {:.3}).",
                describe_method(*method),
                confidence
            ),
        ),
        VerifyResult::Invalid { reason } => (StatusBadge::CandidateInvalid, reason.clone()),
        VerifyResult::Unknown { reason } => {
            (StatusBadge::NotFound, format!("No verdict: {reason}"))
        }
        VerifyResult::Unsupported { reason } => (StatusBadge::Unsupported, reason.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_checked_counts_as_verified() {
        assert!(StatusBadge::Checked.is_verified());
        for badge in [
            StatusBadge::Partial,
            StatusBadge::Heuristic,
            StatusBadge::Unverified,
            StatusBadge::Unsupported,
            StatusBadge::NotFound,
            StatusBadge::InputError,
        ] {
            assert!(
                !badge.is_verified(),
                "{} must not be reported as verified",
                badge.label()
            );
        }
    }

    #[test]
    fn every_badge_has_a_distinct_label_and_marker() {
        let badges = [
            StatusBadge::Checked,
            StatusBadge::Partial,
            StatusBadge::Heuristic,
            StatusBadge::Unverified,
            StatusBadge::Unsupported,
            StatusBadge::NotFound,
            StatusBadge::InputError,
            StatusBadge::CandidateValid,
            StatusBadge::CandidateInvalid,
        ];
        let labels: std::collections::HashSet<&str> = badges.iter().map(|b| b.label()).collect();
        assert_eq!(labels.len(), badges.len(), "badge labels must be distinct");

        for badge in badges {
            assert!(!badge.marker().is_empty());
            assert!(!badge.label().is_empty());
        }
    }

    #[test]
    fn status_reasons_come_from_the_domain() {
        let (badge, reason) = badge_for_status(&VerificationStatus::Partial {
            reason: "2 of 5 steps rely on numeric sampling".to_string(),
        });
        assert_eq!(badge, StatusBadge::Partial);
        assert_eq!(reason, "2 of 5 steps rely on numeric sampling");

        let (badge, reason) = badge_for_status(&VerificationStatus::Unverified {
            reason: "trace does not end at the reported result".to_string(),
        });
        assert_eq!(badge, StatusBadge::Unverified);
        assert!(reason.contains("does not end"));
    }

    #[test]
    fn heuristic_status_never_becomes_checked() {
        let (badge, _) = badge_for_status(&VerificationStatus::Heuristic {
            reason: "every step was accepted by rule replay only".to_string(),
        });
        assert_eq!(badge, StatusBadge::Heuristic);
        assert!(!badge.is_verified());
    }

    #[test]
    fn verify_result_maps_to_candidate_badges() {
        let (badge, reason) = badge_for_verify_result(&VerifyResult::Valid {
            confidence: 1.0,
            method: VerificationMethod::SymbolicEquivalence,
        });
        assert_eq!(badge, StatusBadge::CandidateValid);
        assert!(reason.contains("symbolic equivalence"));

        let (badge, _) = badge_for_verify_result(&VerifyResult::Invalid {
            reason: "Solution does not satisfy the equation".to_string(),
        });
        assert_eq!(badge, StatusBadge::CandidateInvalid);

        let (badge, _) = badge_for_verify_result(&VerifyResult::Unsupported {
            reason: "formal verification is not implemented".to_string(),
        });
        assert_eq!(badge, StatusBadge::Unsupported);

        // An "Unknown" verdict is a bounded failure to decide, not a refutation.
        let (badge, reason) = badge_for_verify_result(&VerifyResult::Unknown {
            reason: "expressions contain calculus operators".to_string(),
        });
        assert_eq!(badge, StatusBadge::NotFound);
        assert!(reason.starts_with("No verdict"));
    }

    #[test]
    fn evidence_distinguishes_replay_from_independent_checks() {
        let symbolic = UiEvidence::from_step(StepEvidence::Checked(
            VerificationMethod::SymbolicEquivalence,
        ));
        assert!(symbolic.checked && symbolic.independent);
        assert!(symbolic.label.contains("symbolic"));

        let replay =
            UiEvidence::from_step(StepEvidence::Checked(VerificationMethod::RuleReplayOnly));
        assert!(replay.checked);
        assert!(
            !replay.independent,
            "rule replay is not an independent check"
        );

        let sampled =
            UiEvidence::from_step(StepEvidence::Checked(VerificationMethod::NumericSampling));
        assert!(sampled.checked && sampled.independent);
        assert!(sampled.label.contains("sampling"));

        let unchecked = UiEvidence::from_step(StepEvidence::Unchecked);
        assert!(!unchecked.checked && !unchecked.independent);
    }

    #[test]
    fn step_summary_reports_the_evidence_class() {
        let step = UiStep {
            index: 1,
            before: "x + 0".to_string(),
            after: "x".to_string(),
            rule: "algebra::identity_add_zero".to_string(),
            rule_id: Some(2),
            justification: "Removed additive identity".to_string(),
            evidence: UiEvidence::from_step(StepEvidence::Checked(
                VerificationMethod::RuleReplayOnly,
            )),
        };
        assert!(step.summary().contains("REPLAY"));
        assert!(step.summary().starts_with("01 "));
    }

    #[test]
    fn modes_declare_the_fields_they_read() {
        assert!(!Mode::Simplify.uses_variable());
        assert!(!Mode::Simplify.uses_candidate());

        assert!(Mode::Differentiate.uses_variable());
        assert!(!Mode::Differentiate.uses_candidate());

        assert!(Mode::VerifyCandidate.uses_variable());
        assert!(Mode::VerifyCandidate.uses_candidate());

        assert_eq!(
            Mode::ALL.len(),
            3,
            "there are exactly three supported operations"
        );
    }
}
