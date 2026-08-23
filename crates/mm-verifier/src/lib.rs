// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Author: Pushp Kharat

//! # mm-verifier
//!
//! Verification system for LEMMA.
//!
//! What each level actually does:
//! - **Numerical**: samples both expressions at random points and compares.
//! - **Symbolic**: compares canonical forms, falling back to sampling.
//! - **Formal**: not implemented. It reports [`VerifyResult::Unsupported`] rather than
//!   pretending an SMT solver ran.
//!
//! None of these prove a rule is mathematically sound. They check that a claimed transition
//! is the rule's own output and that the two expressions agree; every accepted step records
//! the [`VerificationMethod`] that established it, so a caller can tell replay from
//! equivalence. See [`status`] for how per-step evidence becomes a result status.

pub mod numerical;
pub mod status;
pub mod symbolic;

use mm_core::Expr;
use mm_rules::{Rule, RuleContext};

pub use status::{status_from_evidence, StepEvidence, VerificationMethod, VerificationStatus};

/// Verification confidence level.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VerificationLevel {
    /// Fast numerical spot-checking.
    Numerical,
    /// Symbolic canonical form comparison.
    Symbolic,
    /// Machine-checked proof. Not implemented.
    Formal,
}

/// Result of verification.
#[derive(Debug, Clone)]
pub enum VerifyResult {
    /// Step is valid, with the method that established it.
    Valid {
        /// Confidence in the check itself, not in the rule's soundness.
        confidence: f64,
        /// How the step was established.
        method: VerificationMethod,
    },
    /// Step is invalid with reason.
    Invalid {
        /// Why the step was rejected.
        reason: String,
    },
    /// Could not determine (timeout, complexity).
    Unknown {
        /// Why no verdict was reached.
        reason: String,
    },
    /// The requested verification mode is not implemented.
    Unsupported {
        /// What was requested and why it cannot be answered.
        reason: String,
    },
}

impl VerifyResult {
    /// Check if the result is valid.
    pub fn is_valid(&self) -> bool {
        matches!(self, VerifyResult::Valid { .. })
    }

    /// Get confidence if valid.
    pub fn confidence(&self) -> Option<f64> {
        match self {
            VerifyResult::Valid { confidence, .. } => Some(*confidence),
            _ => None,
        }
    }

    /// Get the method that established validity, if any.
    pub fn method(&self) -> Option<VerificationMethod> {
        match self {
            VerifyResult::Valid { method, .. } => Some(*method),
            _ => None,
        }
    }

    /// Evidence to record on a step for this result.
    pub fn evidence(&self) -> StepEvidence {
        match self {
            VerifyResult::Valid { method, .. } => StepEvidence::Checked(*method),
            _ => StepEvidence::Unchecked,
        }
    }
}

/// Whether an expression contains a derivative or integral anywhere inside it.
///
/// Such expressions cannot be sampled by the numeric evaluator, so steps involving them can
/// only be established by rule replay.
fn is_calculus_expr(expr: &Expr) -> bool {
    match expr {
        Expr::Derivative { .. } | Expr::Integral { .. } => true,
        Expr::Neg(e)
        | Expr::Sqrt(e)
        | Expr::Sin(e)
        | Expr::Cos(e)
        | Expr::Tan(e)
        | Expr::Arcsin(e)
        | Expr::Arccos(e)
        | Expr::Arctan(e)
        | Expr::Ln(e)
        | Expr::Exp(e)
        | Expr::Abs(e) => is_calculus_expr(e),
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Pow(a, b) => {
            is_calculus_expr(a) || is_calculus_expr(b)
        }
        Expr::Sum(terms) => terms.iter().any(|t| is_calculus_expr(&t.expr)),
        Expr::Product(factors) => factors
            .iter()
            .any(|f| is_calculus_expr(&f.base) || is_calculus_expr(&f.power)),
        Expr::Equation { lhs, rhs }
        | Expr::GCD(lhs, rhs)
        | Expr::LCM(lhs, rhs)
        | Expr::Mod(lhs, rhs)
        | Expr::Binomial(lhs, rhs)
        | Expr::Gte(lhs, rhs)
        | Expr::Gt(lhs, rhs)
        | Expr::Lte(lhs, rhs)
        | Expr::Lt(lhs, rhs) => is_calculus_expr(lhs) || is_calculus_expr(rhs),
        Expr::Floor(e) | Expr::Ceiling(e) | Expr::Factorial(e) => is_calculus_expr(e),
        Expr::Summation { from, to, body, .. } | Expr::BigProduct { from, to, body, .. } => {
            is_calculus_expr(from) || is_calculus_expr(to) || is_calculus_expr(body)
        }
        Expr::ForAll { domain, body, .. } | Expr::Exists { domain, body, .. } => {
            domain
                .as_ref()
                .map(|d| is_calculus_expr(d))
                .unwrap_or(false)
                || is_calculus_expr(body)
        }
        Expr::And(a, b) | Expr::Or(a, b) | Expr::Implies(a, b) => {
            is_calculus_expr(a) || is_calculus_expr(b)
        }
        Expr::Not(e) => is_calculus_expr(e),
        Expr::Const(_) | Expr::Var(_) | Expr::Pi | Expr::E => false,
    }
}

/// Verifier for mathematical steps.
pub struct Verifier {
    level: VerificationLevel,
    num_samples: usize,
    tolerance: f64,
}

impl Default for Verifier {
    fn default() -> Self {
        Self::new()
    }
}

impl Verifier {
    /// Create a new verifier with default settings.
    pub fn new() -> Self {
        Self {
            level: VerificationLevel::Symbolic,
            num_samples: 10,
            tolerance: 1e-10,
        }
    }

    /// Set the verification level.
    pub fn with_level(mut self, level: VerificationLevel) -> Self {
        self.level = level;
        self
    }

    /// Set the number of numerical samples.
    pub fn with_samples(mut self, n: usize) -> Self {
        self.num_samples = n;
        self
    }

    /// Verify a transformation step.
    ///
    /// Checks that applying the rule to `before` produces `after`.
    pub fn verify_step(
        &self,
        before: &Expr,
        after: &Expr,
        rule: &Rule,
        ctx: &RuleContext,
    ) -> VerifyResult {
        // 1. Check rule is applicable
        if !rule.can_apply(before, ctx) {
            return VerifyResult::Invalid {
                reason: format!("Rule '{}' is not applicable to this expression", rule.name),
            };
        }

        // 2. Check if result is in possible outputs
        let possible_results = rule.apply(before, ctx);
        let result_matches = possible_results
            .iter()
            .any(|r| self.expressions_equal(&r.result, after));

        if !result_matches {
            return VerifyResult::Invalid {
                reason: format!("Rule '{}' does not produce the claimed result", rule.name),
            };
        }

        // 3. Formal mode is not implemented for any expression, calculus or not.
        if self.level == VerificationLevel::Formal {
            return VerifyResult::Unsupported {
                reason: "formal verification is not implemented; no SMT or proof-checking \
                         backend is present"
                    .to_string(),
            };
        }

        // 4. Expressions containing a derivative or integral cannot be evaluated numerically,
        //    so nothing beyond the replay in step 2 can be established about them. Say so.
        if is_calculus_expr(before) || is_calculus_expr(after) {
            return VerifyResult::Valid {
                confidence: 0.95,
                method: VerificationMethod::RuleReplayOnly,
            };
        }

        match self.level {
            VerificationLevel::Numerical => {
                if numerical::verify_equivalent(before, after, self.num_samples, self.tolerance) {
                    VerifyResult::Valid {
                        confidence: 0.999,
                        method: VerificationMethod::NumericSampling,
                    }
                } else {
                    VerifyResult::Invalid {
                        reason: "Numerical verification failed".to_string(),
                    }
                }
            }
            VerificationLevel::Symbolic => {
                if symbolic::verify_equivalent(before, after) {
                    VerifyResult::Valid {
                        confidence: 1.0,
                        method: VerificationMethod::SymbolicEquivalence,
                    }
                } else if numerical::verify_equivalent(
                    before,
                    after,
                    self.num_samples,
                    self.tolerance,
                ) {
                    VerifyResult::Valid {
                        confidence: 0.999,
                        method: VerificationMethod::NumericSampling,
                    }
                } else {
                    VerifyResult::Invalid {
                        reason: "Symbolic verification failed".to_string(),
                    }
                }
            }
            VerificationLevel::Formal => unreachable!("handled above"),
        }
    }

    /// Check that two expressions are equivalent, independently of any rule.
    ///
    /// Used for transitions that are not a registry rule application, such as
    /// canonicalisation or constant folding done during post-processing. Those must still
    /// produce evidence, or the result they contribute to cannot be called checked.
    pub fn verify_equivalence(&self, before: &Expr, after: &Expr) -> VerifyResult {
        if self.level == VerificationLevel::Formal {
            return VerifyResult::Unsupported {
                reason: "formal verification is not implemented".to_string(),
            };
        }

        if symbolic::verify_equivalent(before, after) {
            return VerifyResult::Valid {
                confidence: 1.0,
                method: VerificationMethod::SymbolicEquivalence,
            };
        }

        if is_calculus_expr(before) || is_calculus_expr(after) {
            return VerifyResult::Unknown {
                reason: "expressions contain calculus operators and cannot be sampled".to_string(),
            };
        }

        if numerical::verify_equivalent(before, after, self.num_samples, self.tolerance) {
            VerifyResult::Valid {
                confidence: 0.999,
                method: VerificationMethod::NumericSampling,
            }
        } else {
            VerifyResult::Invalid {
                reason: "expressions are not equivalent".to_string(),
            }
        }
    }

    /// Verify that a solution satisfies an equation.
    pub fn verify_solution(
        &self,
        equation: &Expr,
        variable: mm_core::Symbol,
        solution: &Expr,
    ) -> VerifyResult {
        // Substitute solution into equation and check if lhs == rhs
        if let Expr::Equation { lhs, rhs } = equation {
            let lhs_subst = substitute(lhs, variable, solution);
            let rhs_subst = substitute(rhs, variable, solution);

            // After substitution, lhs should equal rhs
            if self.expressions_equal(&lhs_subst, &rhs_subst) {
                return VerifyResult::Valid {
                    confidence: 1.0,
                    method: VerificationMethod::SymbolicEquivalence,
                };
            }

            // Try numerical verification
            let diff = Expr::Sub(Box::new(lhs_subst.clone()), Box::new(rhs_subst.clone()));
            if numerical::is_zero(&diff, self.num_samples, self.tolerance) {
                return VerifyResult::Valid {
                    confidence: 0.999,
                    method: VerificationMethod::NumericSampling,
                };
            }

            return VerifyResult::Invalid {
                reason: "Solution does not satisfy the equation".to_string(),
            };
        }

        VerifyResult::Invalid {
            reason: "Expected an equation".to_string(),
        }
    }

    /// Check if two expressions are equal.
    fn expressions_equal(&self, a: &Expr, b: &Expr) -> bool {
        // First try structural equality
        if a == b {
            return true;
        }

        // Then try canonical form
        let canon_a = a.canonicalize();
        let canon_b = b.canonicalize();
        if canon_a == canon_b {
            return true;
        }

        // Finally try numerical
        a.approx_equals(b, self.num_samples, self.tolerance)
    }
}

/// Replace all free occurrences of a variable in an expression with another expression.
///
/// Occurrences bound by a local quantifier or a summation/product index are left alone when
/// the bound variable shadows `var`.
fn substitute(expr: &Expr, var: mm_core::Symbol, value: &Expr) -> Expr {
    match expr {
        Expr::Var(v) if *v == var => value.clone(),
        Expr::Var(_) | Expr::Const(_) | Expr::Pi | Expr::E => expr.clone(),
        Expr::Neg(e) => Expr::Neg(Box::new(substitute(e, var, value))),
        Expr::Sqrt(e) => Expr::Sqrt(Box::new(substitute(e, var, value))),
        Expr::Sin(e) => Expr::Sin(Box::new(substitute(e, var, value))),
        Expr::Cos(e) => Expr::Cos(Box::new(substitute(e, var, value))),
        Expr::Tan(e) => Expr::Tan(Box::new(substitute(e, var, value))),
        Expr::Arcsin(e) => Expr::Arcsin(Box::new(substitute(e, var, value))),
        Expr::Arccos(e) => Expr::Arccos(Box::new(substitute(e, var, value))),
        Expr::Arctan(e) => Expr::Arctan(Box::new(substitute(e, var, value))),
        Expr::Ln(e) => Expr::Ln(Box::new(substitute(e, var, value))),
        Expr::Exp(e) => Expr::Exp(Box::new(substitute(e, var, value))),
        Expr::Abs(e) => Expr::Abs(Box::new(substitute(e, var, value))),
        Expr::Add(a, b) => Expr::Add(
            Box::new(substitute(a, var, value)),
            Box::new(substitute(b, var, value)),
        ),
        Expr::Sub(a, b) => Expr::Sub(
            Box::new(substitute(a, var, value)),
            Box::new(substitute(b, var, value)),
        ),
        Expr::Mul(a, b) => Expr::Mul(
            Box::new(substitute(a, var, value)),
            Box::new(substitute(b, var, value)),
        ),
        Expr::Div(a, b) => Expr::Div(
            Box::new(substitute(a, var, value)),
            Box::new(substitute(b, var, value)),
        ),
        Expr::Pow(a, b) => Expr::Pow(
            Box::new(substitute(a, var, value)),
            Box::new(substitute(b, var, value)),
        ),
        Expr::Sum(terms) => Expr::Sum(
            terms
                .iter()
                .map(|t| mm_core::Term {
                    coeff: t.coeff,
                    expr: substitute(&t.expr, var, value),
                })
                .collect(),
        ),
        Expr::Product(factors) => Expr::Product(
            factors
                .iter()
                .map(|f| mm_core::Factor {
                    base: substitute(&f.base, var, value),
                    power: substitute(&f.power, var, value),
                })
                .collect(),
        ),
        Expr::Derivative { expr: e, var: v } => Expr::Derivative {
            expr: Box::new(substitute(e, var, value)),
            var: *v,
        },
        Expr::Integral { expr: e, var: v } => Expr::Integral {
            expr: Box::new(substitute(e, var, value)),
            var: *v,
        },
        Expr::Equation { lhs, rhs } => Expr::Equation {
            lhs: Box::new(substitute(lhs, var, value)),
            rhs: Box::new(substitute(rhs, var, value)),
        },
        // Number theory
        Expr::GCD(a, b) => Expr::GCD(
            Box::new(substitute(a, var, value)),
            Box::new(substitute(b, var, value)),
        ),
        Expr::LCM(a, b) => Expr::LCM(
            Box::new(substitute(a, var, value)),
            Box::new(substitute(b, var, value)),
        ),
        Expr::Mod(a, b) => Expr::Mod(
            Box::new(substitute(a, var, value)),
            Box::new(substitute(b, var, value)),
        ),
        Expr::Binomial(n, k) => Expr::Binomial(
            Box::new(substitute(n, var, value)),
            Box::new(substitute(k, var, value)),
        ),
        Expr::Gte(a, b) => Expr::Gte(
            Box::new(substitute(a, var, value)),
            Box::new(substitute(b, var, value)),
        ),
        Expr::Gt(a, b) => Expr::Gt(
            Box::new(substitute(a, var, value)),
            Box::new(substitute(b, var, value)),
        ),
        Expr::Lte(a, b) => Expr::Lte(
            Box::new(substitute(a, var, value)),
            Box::new(substitute(b, var, value)),
        ),
        Expr::Lt(a, b) => Expr::Lt(
            Box::new(substitute(a, var, value)),
            Box::new(substitute(b, var, value)),
        ),
        Expr::Floor(e) => Expr::Floor(Box::new(substitute(e, var, value))),
        Expr::Ceiling(e) => Expr::Ceiling(Box::new(substitute(e, var, value))),
        Expr::Factorial(e) => Expr::Factorial(Box::new(substitute(e, var, value))),
        Expr::Summation {
            var: v,
            from,
            to,
            body,
        } => {
            // Don't substitute bound variable in body if it shadows
            if *v == var {
                Expr::Summation {
                    var: *v,
                    from: Box::new(substitute(from, var, value)),
                    to: Box::new(substitute(to, var, value)),
                    body: body.clone(),
                }
            } else {
                Expr::Summation {
                    var: *v,
                    from: Box::new(substitute(from, var, value)),
                    to: Box::new(substitute(to, var, value)),
                    body: Box::new(substitute(body, var, value)),
                }
            }
        }
        Expr::BigProduct {
            var: v,
            from,
            to,
            body,
        } => {
            if *v == var {
                Expr::BigProduct {
                    var: *v,
                    from: Box::new(substitute(from, var, value)),
                    to: Box::new(substitute(to, var, value)),
                    body: body.clone(),
                }
            } else {
                Expr::BigProduct {
                    var: *v,
                    from: Box::new(substitute(from, var, value)),
                    to: Box::new(substitute(to, var, value)),
                    body: Box::new(substitute(body, var, value)),
                }
            }
        }
        Expr::ForAll {
            var: v,
            domain,
            body,
        } => {
            if *v == var {
                Expr::ForAll {
                    var: *v,
                    domain: domain.as_ref().map(|d| Box::new(substitute(d, var, value))),
                    body: body.clone(),
                }
            } else {
                Expr::ForAll {
                    var: *v,
                    domain: domain.as_ref().map(|d| Box::new(substitute(d, var, value))),
                    body: Box::new(substitute(body, var, value)),
                }
            }
        }
        Expr::Exists {
            var: v,
            domain,
            body,
        } => {
            if *v == var {
                Expr::Exists {
                    var: *v,
                    domain: domain.as_ref().map(|d| Box::new(substitute(d, var, value))),
                    body: body.clone(),
                }
            } else {
                Expr::Exists {
                    var: *v,
                    domain: domain.as_ref().map(|d| Box::new(substitute(d, var, value))),
                    body: Box::new(substitute(body, var, value)),
                }
            }
        }
        Expr::And(a, b) => Expr::And(
            Box::new(substitute(a, var, value)),
            Box::new(substitute(b, var, value)),
        ),
        Expr::Or(a, b) => Expr::Or(
            Box::new(substitute(a, var, value)),
            Box::new(substitute(b, var, value)),
        ),
        Expr::Not(e) => Expr::Not(Box::new(substitute(e, var, value))),
        Expr::Implies(a, b) => Expr::Implies(
            Box::new(substitute(a, var, value)),
            Box::new(substitute(b, var, value)),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::SymbolTable;

    #[test]
    fn test_verifier_creation() {
        let verifier = Verifier::new();
        assert_eq!(verifier.level, VerificationLevel::Symbolic);
    }

    #[test]
    fn test_substitution() {
        let mut symbols = SymbolTable::new();
        let x = symbols.intern("x");

        // Substitute x = 3 into x + 1
        let expr = Expr::Add(Box::new(Expr::Var(x)), Box::new(Expr::int(1)));
        let result = substitute(&expr, x, &Expr::int(3));

        // Should get 3 + 1
        assert_eq!(
            result,
            Expr::Add(Box::new(Expr::int(3)), Box::new(Expr::int(1)))
        );
    }
}
