// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Author: Pushp Kharat

//! Guardrails for rule application.
//!
//! Analyses an expression into a coarse [`ProblemProfile`] and filters rules whose declared
//! domain cannot apply to it.
//!
//! `mm-boink` carries a richer profile (more flags, nesting depth, a domain list) and it is
//! the one the search actually consults. This one exists for callers that only need the
//! complexity estimate. Both must be complete traversals: a scanner that skips a node kind
//! silently reports "no trigonometry" for `sin(x)^2 + cos(x)^2` and the guardrail then hides
//! every trigonometric rule.

use crate::{Domain, Rule};
use mm_core::Expr;

/// Characteristics of a problem or sub-problem.
#[derive(Debug, Clone, Default)]
pub struct ProblemProfile {
    pub has_trig: bool,
    pub has_calculus: bool,
    pub has_combinatorics: bool,
    pub complexity: u32,
}

/// Analyze an expression to determine its problem profile.
pub fn analyze(expr: &Expr) -> ProblemProfile {
    let mut profile = ProblemProfile::default();
    scan_expr(expr, &mut profile);
    profile
}

fn scan_expr(expr: &Expr, profile: &mut ProblemProfile) {
    match expr {
        // Binary operations.
        Expr::Add(a, b)
        | Expr::Sub(a, b)
        | Expr::Mul(a, b)
        | Expr::Div(a, b)
        | Expr::Pow(a, b)
        | Expr::Equation { lhs: a, rhs: b }
        | Expr::Gte(a, b)
        | Expr::Gt(a, b)
        | Expr::Lte(a, b)
        | Expr::Lt(a, b)
        | Expr::And(a, b)
        | Expr::Or(a, b)
        | Expr::Implies(a, b) => {
            scan_expr(a, profile);
            scan_expr(b, profile);
            profile.complexity += 1;
        }

        // Trigonometry, direct and inverse.
        Expr::Sin(inner)
        | Expr::Cos(inner)
        | Expr::Tan(inner)
        | Expr::Arcsin(inner)
        | Expr::Arccos(inner)
        | Expr::Arctan(inner) => {
            profile.has_trig = true;
            scan_expr(inner, profile);
            profile.complexity += 2;
        }

        // Calculus.
        Expr::Derivative { expr: inner, .. } | Expr::Integral { expr: inner, .. } => {
            profile.has_calculus = true;
            scan_expr(inner, profile);
            profile.complexity += 5;
        }

        // Combinatorics.
        Expr::Factorial(inner) => {
            profile.has_combinatorics = true;
            scan_expr(inner, profile);
            profile.complexity += 4;
        }
        Expr::Binomial(a, b) => {
            profile.has_combinatorics = true;
            scan_expr(a, profile);
            scan_expr(b, profile);
            profile.complexity += 5;
        }
        Expr::Summation { from, to, body, .. } | Expr::BigProduct { from, to, body, .. } => {
            profile.has_combinatorics = true;
            scan_expr(from, profile);
            scan_expr(to, profile);
            scan_expr(body, profile);
            profile.complexity += 6;
        }

        // Remaining binary and unary forms carry no domain flag of their own.
        Expr::GCD(a, b) | Expr::LCM(a, b) | Expr::Mod(a, b) => {
            scan_expr(a, profile);
            scan_expr(b, profile);
            profile.complexity += 3;
        }
        Expr::Sqrt(inner)
        | Expr::Ln(inner)
        | Expr::Exp(inner)
        | Expr::Abs(inner)
        | Expr::Neg(inner)
        | Expr::Floor(inner)
        | Expr::Ceiling(inner)
        | Expr::Not(inner) => {
            scan_expr(inner, profile);
            profile.complexity += 1;
        }
        Expr::ForAll { domain, body, .. } | Expr::Exists { domain, body, .. } => {
            if let Some(d) = domain {
                scan_expr(d, profile);
            }
            scan_expr(body, profile);
            profile.complexity += 3;
        }
        Expr::Sum(terms) => {
            for term in terms {
                scan_expr(&term.expr, profile);
            }
            profile.complexity += terms.len() as u32;
        }
        Expr::Product(factors) => {
            for factor in factors {
                scan_expr(&factor.base, profile);
                scan_expr(&factor.power, profile);
            }
            profile.complexity += factors.len() as u32;
        }

        // Atoms.
        Expr::Const(_) | Expr::Var(_) | Expr::Pi | Expr::E => {
            profile.complexity += 1;
        }
    }
}

/// Determines if a rule is applicable based on the problem profile.
///
/// This acts as a high-level filter. For example, if a problem has no
/// trigonometric functions, we shouldn't waste time checking trig rules.
pub fn is_rule_applicable(rule: &Rule, profile: &ProblemProfile) -> bool {
    // Check domain constraints
    for domain in rule.domains {
        match domain {
            Domain::Trigonometry if !profile.has_trig => return false,
            Domain::CalculusDiff | Domain::CalculusInt if !profile.has_calculus => return false,
            _ => {}
        }
    }
    true
}

/// Filters a list of rules to those relevant for the current problem profile.
pub fn filter_rules<'a>(rules: &'a [Rule], profile: &ProblemProfile) -> Vec<&'a Rule> {
    rules
        .iter()
        .filter(|r| is_rule_applicable(r, profile))
        .collect()
}

/// Helper to decompose an additive expression into its terms.
///
/// Useful for analyzing or processing terms independently.
/// e.g., a + b + c -> [a, b, c]
pub fn decompose_additive(expr: &Expr) -> Vec<Expr> {
    let mut terms = Vec::new();
    collect_additive_terms(expr, &mut terms);
    terms
}

fn collect_additive_terms(expr: &Expr, terms: &mut Vec<Expr>) {
    if let Expr::Add(a, b) = expr {
        collect_additive_terms(a, terms);
        collect_additive_terms(b, terms);
    } else {
        terms.push(expr.clone());
    }
}

/// Calculate a heuristic solvability score (0.0 to 1.0).
///
/// Higher scores indicate the expression is "simpler" or closer to a solution.
pub fn solvability_score(expr: &Expr) -> f64 {
    let profile = analyze(expr);
    // Very basic heuristic: lower complexity is better.
    if profile.complexity == 0 {
        return 1.0;
    }
    1.0 / (profile.complexity as f64).sqrt()
}
