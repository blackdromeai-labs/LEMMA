// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Author: Pushp Kharat

//! Evidence-bearing verification status.
//!
//! Search results used to carry a bare `verified: bool`, which was set to `true`
//! unconditionally by several code paths — including ones that applied rules without calling
//! the verifier at all, and post-processing that changed the final expression without
//! recording a step. The types here replace that boolean: a result says what was checked and
//! how, and "fully checked" is reserved for a trace that replays from the exact input to the
//! exact reported output through individually checked transitions.

use std::fmt;

/// How a single transition was established.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum VerificationMethod {
    /// Both sides reduce to the same canonical form.
    ///
    /// This is the strongest check available here. It is still equivalence checking, not a
    /// machine-checked proof of the rule's soundness.
    SymbolicEquivalence,
    /// Both sides agreed at every sampled point.
    ///
    /// Sampling can miss a disagreement, so this is evidence, not proof.
    NumericSampling,
    /// Re-running the rule reproduced the claimed output, and nothing further was checked.
    ///
    /// Used where the expression contains a derivative or integral, which the evaluator
    /// cannot sample. It shows the step is the rule's own output; it does not show the rule
    /// is mathematically sound.
    RuleReplayOnly,
}

impl VerificationMethod {
    /// Whether this method independently checks the two expressions agree.
    pub fn is_independent(&self) -> bool {
        matches!(
            self,
            VerificationMethod::SymbolicEquivalence | VerificationMethod::NumericSampling
        )
    }
}

impl fmt::Display for VerificationMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VerificationMethod::SymbolicEquivalence => write!(f, "symbolic equivalence"),
            VerificationMethod::NumericSampling => write!(f, "numeric sampling"),
            VerificationMethod::RuleReplayOnly => write!(f, "rule replay only"),
        }
    }
}

/// Evidence attached to one recorded step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepEvidence {
    /// The verifier accepted this transition, by the recorded method.
    Checked(VerificationMethod),
    /// The transition was applied without being checked.
    Unchecked,
}

impl StepEvidence {
    /// Whether the transition was checked at all.
    pub fn is_checked(&self) -> bool {
        matches!(self, StepEvidence::Checked(_))
    }

    /// The method used, if any.
    pub fn method(&self) -> Option<VerificationMethod> {
        match self {
            StepEvidence::Checked(m) => Some(*m),
            StepEvidence::Unchecked => None,
        }
    }
}

impl fmt::Display for StepEvidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StepEvidence::Checked(m) => write!(f, "checked ({m})"),
            StepEvidence::Unchecked => write!(f, "unchecked"),
        }
    }
}

/// What is actually known about a produced result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationStatus {
    /// The trace replays from the exact input to the exact output, and every step was checked
    /// by an independent equivalence check.
    Checked,
    /// The trace replays, but at least one step was accepted on weaker grounds than the rest.
    Partial {
        /// What was weaker, and where.
        reason: String,
    },
    /// The trace replays, but its evidence is sampling or rule replay throughout.
    Heuristic {
        /// Which weaker method was used.
        reason: String,
    },
    /// There is no checked path from the exact input to the exact output.
    Unverified {
        /// Why the trace could not be accepted.
        reason: String,
    },
    /// The requested verification mode is not implemented.
    Unsupported {
        /// What was requested and why it cannot be answered.
        reason: String,
    },
}

impl VerificationStatus {
    /// Whether the result is fully replayed and independently checked.
    ///
    /// This is the only status that may be reported as "verified".
    pub fn is_fully_checked(&self) -> bool {
        matches!(self, VerificationStatus::Checked)
    }

    /// Whether the trace at least replays from input to output.
    pub fn replays(&self) -> bool {
        matches!(
            self,
            VerificationStatus::Checked
                | VerificationStatus::Partial { .. }
                | VerificationStatus::Heuristic { .. }
        )
    }

    /// Short machine-readable label, for tables and reports.
    pub fn label(&self) -> &'static str {
        match self {
            VerificationStatus::Checked => "checked",
            VerificationStatus::Partial { .. } => "partial",
            VerificationStatus::Heuristic { .. } => "heuristic",
            VerificationStatus::Unverified { .. } => "unverified",
            VerificationStatus::Unsupported { .. } => "unsupported",
        }
    }
}

impl fmt::Display for VerificationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VerificationStatus::Checked => write!(f, "checked"),
            VerificationStatus::Partial { reason } => write!(f, "partial: {reason}"),
            VerificationStatus::Heuristic { reason } => write!(f, "heuristic: {reason}"),
            VerificationStatus::Unverified { reason } => write!(f, "unverified: {reason}"),
            VerificationStatus::Unsupported { reason } => write!(f, "unsupported: {reason}"),
        }
    }
}

/// Combine per-step evidence into a status, given that the trace already replays.
///
/// `evidence` must be in step order. An empty trace means the input was already the answer,
/// which is checked by definition.
pub fn status_from_evidence(evidence: &[StepEvidence]) -> VerificationStatus {
    let unchecked = evidence.iter().filter(|e| !e.is_checked()).count();
    if unchecked > 0 {
        return VerificationStatus::Unverified {
            reason: format!("{unchecked} of {} steps were not checked", evidence.len()),
        };
    }

    let weak: Vec<VerificationMethod> = evidence
        .iter()
        .filter_map(|e| e.method())
        .filter(|m| !m.is_independent())
        .collect();

    if weak.is_empty() {
        let sampled = evidence
            .iter()
            .filter_map(|e| e.method())
            .filter(|m| *m == VerificationMethod::NumericSampling)
            .count();
        if sampled == 0 {
            VerificationStatus::Checked
        } else if sampled == evidence.len() {
            VerificationStatus::Heuristic {
                reason: "every step was checked by numeric sampling only".to_string(),
            }
        } else {
            VerificationStatus::Partial {
                reason: format!(
                    "{sampled} of {} steps rely on numeric sampling",
                    evidence.len()
                ),
            }
        }
    } else if weak.len() == evidence.len() {
        VerificationStatus::Heuristic {
            reason: "every step was accepted by rule replay only".to_string(),
        }
    } else {
        VerificationStatus::Partial {
            reason: format!(
                "{} of {} steps were accepted by rule replay only",
                weak.len(),
                evidence.len()
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_trace_is_checked() {
        assert_eq!(status_from_evidence(&[]), VerificationStatus::Checked);
    }

    #[test]
    fn all_symbolic_is_checked() {
        let e = vec![StepEvidence::Checked(VerificationMethod::SymbolicEquivalence); 3];
        assert_eq!(status_from_evidence(&e), VerificationStatus::Checked);
    }

    #[test]
    fn any_unchecked_step_makes_the_whole_result_unverified() {
        let e = vec![
            StepEvidence::Checked(VerificationMethod::SymbolicEquivalence),
            StepEvidence::Unchecked,
        ];
        let status = status_from_evidence(&e);
        assert!(!status.is_fully_checked());
        assert!(matches!(status, VerificationStatus::Unverified { .. }));
    }

    #[test]
    fn replay_only_evidence_is_heuristic_not_checked() {
        let e = vec![StepEvidence::Checked(VerificationMethod::RuleReplayOnly); 2];
        let status = status_from_evidence(&e);
        assert!(!status.is_fully_checked());
        assert!(matches!(status, VerificationStatus::Heuristic { .. }));
    }

    #[test]
    fn mixed_strength_evidence_is_partial() {
        let e = vec![
            StepEvidence::Checked(VerificationMethod::SymbolicEquivalence),
            StepEvidence::Checked(VerificationMethod::RuleReplayOnly),
        ];
        assert!(matches!(
            status_from_evidence(&e),
            VerificationStatus::Partial { .. }
        ));
    }

    #[test]
    fn sampling_only_is_heuristic() {
        let e = vec![StepEvidence::Checked(VerificationMethod::NumericSampling); 2];
        assert!(matches!(
            status_from_evidence(&e),
            VerificationStatus::Heuristic { .. }
        ));
    }
}
