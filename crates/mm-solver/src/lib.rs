// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Author: Pushp Kharat

//! # mm-solver
//!
//! The unified API for the LEMMA system.
//!
//! This crate combines all components into a single, easy-to-use interface
//! for solving mathematical problems with step-by-step reasoning.
//!
//! ## Solvers
//!
//! - [`LemmaSolver`] - Algebraic simplification and differentiation over parsed expressions
//! - [`IMOSolver`] - Search over an expression that is already in formal form
//!
//! Neither accepts a natural-language problem statement. [`IMOSolver::solve_text`] reports
//! that the input is unsupported rather than searching something else.
//!
//! ## Example
//!
//! ```rust
//! use mm_solver::LemmaSolver;
//!
//! let mut solver = LemmaSolver::new();
//! let result = solver.simplify("2 + 3").unwrap();
//! ```

pub mod imo_solver;
pub mod orchestrator;

use mm_core::{Expr, MathError, SymbolTable};
use mm_rules::{rule::standard_rules, RuleSet};
use mm_search::{BeamSearch, SearchConfig, Step};
use mm_verifier::{VerificationStatus, Verifier, VerifyResult};

pub use imo_solver::{
    IMOOutcome, IMOSolveResult, IMOSolver, IMOSolverConfig, SolutionStep, UnsupportedInput,
};

/// The LEMMA solver.
///
/// This is the main entry point for mathematical reasoning.
pub struct LemmaSolver {
    rules: RuleSet,
    verifier: Verifier,
    search: BeamSearch,
    symbols: SymbolTable,
}

impl Default for LemmaSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl LemmaSolver {
    /// Create a new LEMMA solver with default settings.
    pub fn new() -> Self {
        let rules = standard_rules();
        let verifier = Verifier::new();
        let search = BeamSearch::new(standard_rules(), Verifier::new());
        let symbols = SymbolTable::new();

        Self {
            rules,
            verifier,
            search,
            symbols,
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: SearchConfig) -> Self {
        let rules = standard_rules();
        let verifier = Verifier::new();
        let search = BeamSearch::with_config(standard_rules(), Verifier::new(), config);
        let symbols = SymbolTable::new();

        Self {
            rules,
            verifier,
            search,
            symbols,
        }
    }

    /// Parse an expression from a string.
    pub fn parse(&mut self, input: &str) -> Result<Expr, MathError> {
        use mm_core::parse::Parser;
        let mut parser = Parser::new(&mut self.symbols);
        parser.parse(input)
    }

    /// Simplify an expression.
    pub fn simplify(&mut self, input: &str) -> Result<SolveResult, MathError> {
        let expr = self.parse(input)?;
        let solution = self.search.simplify(expr);

        Ok(SolveResult::from_solution(solution))
    }

    /// Simplify an already-parsed expression.
    pub fn simplify_expr(&self, expr: Expr) -> SolveResult {
        SolveResult::from_solution(self.search.simplify(expr))
    }

    /// Compute the derivative of an expression.
    pub fn differentiate(&mut self, input: &str, var: &str) -> Result<SolveResult, MathError> {
        let expr = self.parse(input)?;
        let var_symbol = self.symbols.intern(var);

        // Create derivative expression
        let deriv = Expr::Derivative {
            expr: Box::new(expr),
            var: var_symbol,
        };

        // Simplify to evaluate the derivative
        Ok(SolveResult::from_solution(self.search.simplify(deriv)))
    }

    /// Solve an equation for a variable.
    ///
    /// Not implemented. It parses and validates the equation, then reports
    /// [`MathError::NotImplemented`]. It previously returned an empty vector, which reads as
    /// "this equation has no solutions" rather than "nothing looked for any".
    ///
    /// [`Self::simplify_expr`] can make progress on an [`Expr::Equation`] through the
    /// equation rules; [`Self::verify_solution`] can check a candidate value.
    pub fn solve_for(&mut self, equation: &str, var: &str) -> Result<Vec<SolveResult>, MathError> {
        let parts: Vec<&str> = equation.split('=').collect();
        if parts.len() != 2 {
            return Err(MathError::ParseError(
                "Expected equation in 'lhs = rhs' format".to_string(),
            ));
        }

        // Parse both sides so a malformed equation is still reported as a parse error.
        self.parse(parts[0].trim())?;
        self.parse(parts[1].trim())?;
        self.symbols.intern(var);

        Err(MathError::NotImplemented(
            "solve_for: equation solving is not implemented; use simplify_expr on an \
             Expr::Equation, or verify_solution to check a candidate"
                .to_string(),
        ))
    }

    /// Verify that a value is a solution to an equation.
    pub fn verify_solution(
        &mut self,
        equation: &str,
        var: &str,
        value: &str,
    ) -> Result<VerifyResult, MathError> {
        let parts: Vec<&str> = equation.split('=').collect();
        if parts.len() != 2 {
            return Err(MathError::ParseError(
                "Expected equation in 'lhs = rhs' format".to_string(),
            ));
        }

        let lhs = self.parse(parts[0].trim())?;
        let rhs = self.parse(parts[1].trim())?;
        let var_symbol = self.symbols.intern(var);
        let value_expr = self.parse(value)?;

        let eq = Expr::Equation {
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        };

        Ok(self.verifier.verify_solution(&eq, var_symbol, &value_expr))
    }

    /// Get a reference to the symbol table.
    pub fn symbols(&self) -> &SymbolTable {
        &self.symbols
    }

    /// Get a mutable reference to the symbol table.
    pub fn symbols_mut(&mut self) -> &mut SymbolTable {
        &mut self.symbols
    }

    /// Get the number of rules loaded.
    pub fn num_rules(&self) -> usize {
        self.rules.len()
    }

    /// Read-only access to the rule registry.
    ///
    /// A [`Step`] records a [`mm_rules::RuleId`], and resolving that to the stable
    /// `module::name` key needs the registry the step came from. Presentation layers that
    /// show rule identity would otherwise have to build a second registry and hope the
    /// identifiers line up.
    pub fn rules(&self) -> &RuleSet {
        &self.rules
    }

    /// Read-only access to the verifier.
    ///
    /// Lets a caller check a candidate against an already-parsed [`Expr`], rather than going
    /// through [`Self::verify_solution`], which re-parses from strings and so cannot report
    /// which input field a parse error belongs to.
    pub fn verifier(&self) -> &Verifier {
        &self.verifier
    }
}

/// Result of solving a problem.
#[derive(Debug, Clone)]
pub struct SolveResult {
    /// The final result expression.
    pub result: Expr,
    /// The steps taken to reach the result.
    pub steps: Vec<Step>,
    /// What is actually known about the result.
    ///
    /// Replaces a `verified: bool` that several code paths set to `true` unconditionally.
    pub status: VerificationStatus,
}

impl SolveResult {
    /// Carry a search solution through unchanged, status included.
    fn from_solution(solution: mm_search::Solution) -> Self {
        Self {
            result: solution.result,
            steps: solution.steps,
            status: solution.status,
        }
    }

    /// Whether the trace replays from the input and every step was independently checked.
    pub fn is_fully_verified(&self) -> bool {
        self.status.is_fully_checked()
    }

    /// Get the number of steps.
    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }

    /// Check if no steps were needed.
    pub fn is_trivial(&self) -> bool {
        self.steps.is_empty()
    }

    /// Format the solution as a human-readable string.
    pub fn format(&self, _symbols: &SymbolTable) -> String {
        let mut output = String::new();

        if self.steps.is_empty() {
            output.push_str(&format!("Result: {:?}\n", self.result));
            output.push_str("(No simplification needed)\n");
        } else {
            for (i, step) in self.steps.iter().enumerate() {
                output.push_str(&format!(
                    "Step {}: {} ({})\n",
                    i + 1,
                    step.rule_name,
                    step.justification
                ));
                output.push_str(&format!("  → {:?}\n", step.after));
            }
            output.push_str(&format!("\nFinal Result: {:?}\n", self.result));
        }

        output.push_str(&format!("Verification: {}\n", self.status));

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_creation() {
        let solver = LemmaSolver::new();
        assert!(solver.num_rules() > 0);
    }

    #[test]
    fn test_simplify() {
        let mut solver = LemmaSolver::new();

        let result = solver.simplify("2 + 3").unwrap();
        assert_eq!(result.result.canonicalize(), Expr::int(5));
    }

    #[test]
    fn test_parse() {
        let mut solver = LemmaSolver::new();

        let expr = solver.parse("x + 1").unwrap();
        assert!(matches!(expr, Expr::Add(_, _)));
    }

    #[test]
    fn solve_for_reports_that_it_is_unimplemented() {
        let mut solver = LemmaSolver::new();

        let err = solver.solve_for("x + 1 = 3", "x").unwrap_err();
        assert!(
            matches!(err, MathError::NotImplemented(_)),
            "an unimplemented solver must not return an empty solution set, got {err:?}"
        );
    }

    #[test]
    fn solve_for_still_reports_parse_errors() {
        let mut solver = LemmaSolver::new();
        assert!(matches!(
            solver.solve_for("x + 1", "x").unwrap_err(),
            MathError::ParseError(_)
        ));
    }

    #[test]
    fn a_simplified_result_carries_a_replayable_trace() {
        let mut solver = LemmaSolver::new();
        let result = solver.simplify("2 + 3").unwrap();

        assert_eq!(result.result.canonicalize(), Expr::int(5));
        assert!(
            result.status.replays(),
            "status must reflect the recorded trace, got {}",
            result.status
        );
    }
}
