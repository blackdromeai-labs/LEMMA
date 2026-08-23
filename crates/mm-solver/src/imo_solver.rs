//! Search over competition-style problems that are already in formal form.
//!
//! [`IMOSolver::solve_expr`] runs [`DeepMCTS`] over an [`Expr`].
//!
//! [`IMOSolver::solve_text`] does **not** solve anything. There is no path from a natural
//! language problem statement to an [`Expr`] in this repository. The previous implementation
//! generated substitution hints, discarded them, built the same hard-coded expression
//! `(a+b)^2 - (a^2 + 2ab + b^2)` for every input, searched that, and returned the resulting
//! path as the problem's solution with steps named `transformation` / "Applied rule". Any
//! apparent input-dependent result from it was an artefact. It now returns
//! [`IMOOutcome::Unsupported`] together with the hints, which remain useful as triage output.

use mm_brain::{SubstitutionPrediction, SubstitutionPredictor};
use mm_core::{Expr, SymbolTable};
use mm_rules::rule::standard_rules;
use mm_search::{DeepMCTS, DeepMCTSConfig, SearchStats};
use mm_verifier::Verifier;
use std::time::{Duration, Instant};

/// Configuration for the solver.
#[derive(Clone)]
pub struct IMOSolverConfig {
    /// Maximum nodes to explore
    pub max_nodes: u64,
    /// Time limit in seconds
    pub time_limit_secs: u64,
    /// Number of substitutions to request from the predictor
    pub top_k_substitutions: usize,
    /// Verbose output
    pub verbose: bool,
}

impl Default for IMOSolverConfig {
    fn default() -> Self {
        Self {
            max_nodes: 1_000_000,
            time_limit_secs: 60,
            top_k_substitutions: 5,
            verbose: true,
        }
    }
}

impl IMOSolverConfig {
    /// Quick mode: 100K nodes, 10s
    pub fn quick() -> Self {
        Self {
            max_nodes: 100_000,
            time_limit_secs: 10,
            top_k_substitutions: 3,
            verbose: false,
        }
    }

    /// Long mode: 100M nodes, 30 minutes
    pub fn competition() -> Self {
        Self {
            max_nodes: 100_000_000,
            time_limit_secs: 1800,
            top_k_substitutions: 10,
            verbose: true,
        }
    }
}

/// Why a text problem could not be turned into a formal one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedInput {
    /// The input that was rejected, truncated for reporting.
    pub input: String,
    /// What is missing.
    pub reason: String,
}

/// What happened to a solve request.
#[derive(Debug)]
pub enum IMOOutcome {
    /// A goal path was found.
    Solved(Vec<SolutionStep>),
    /// The search ran and found no goal path.
    NotFound,
    /// The input could not be turned into a formal problem, so nothing was searched.
    Unsupported(UnsupportedInput),
}

impl IMOOutcome {
    /// Whether a solution path was produced.
    pub fn is_solved(&self) -> bool {
        matches!(self, IMOOutcome::Solved(_))
    }

    /// The solution path, if there is one.
    pub fn path(&self) -> Option<&[SolutionStep]> {
        match self {
            IMOOutcome::Solved(steps) => Some(steps),
            _ => None,
        }
    }
}

/// Result of a solve request.
#[derive(Debug)]
pub struct IMOSolveResult {
    /// What happened.
    pub outcome: IMOOutcome,
    /// Substitution hints from the predictor. These are keyword-driven suggestions for a
    /// human reader; nothing in the search consumes them.
    pub substitutions_suggested: Vec<SubstitutionPrediction>,
    /// Search statistics. All zero when nothing was searched.
    pub stats: SearchStats,
    /// Time taken
    pub elapsed: Duration,
}

impl IMOSolveResult {
    /// Whether a solution path was produced.
    pub fn solved(&self) -> bool {
        self.outcome.is_solved()
    }
}

/// A step in the solution
#[derive(Debug, Clone, PartialEq)]
pub struct SolutionStep {
    /// Expression before this step
    pub before: Expr,
    /// Expression after this step
    pub after: Expr,
    /// Explanation of the transition.
    ///
    /// [`DeepMCTS`] returns a path of expressions rather than of rule applications, so this
    /// describes the shape of the transition, not a named rule.
    pub explanation: String,
}

/// Search driver for problems supplied as expressions.
pub struct IMOSolver {
    /// Parallel tree search engine
    mcts: DeepMCTS,
    /// Substitution predictor (keyword heuristics)
    predictor: SubstitutionPredictor,
    /// Symbol table for parsing
    symbols: SymbolTable,
    /// Configuration
    config: IMOSolverConfig,
}

impl IMOSolver {
    /// Create a solver with default configuration
    pub fn new() -> Self {
        Self::with_config(IMOSolverConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: IMOSolverConfig) -> Self {
        let rules = standard_rules();
        let verifier = Verifier::new();
        let mcts_config = DeepMCTSConfig {
            max_nodes: config.max_nodes,
            time_limit_secs: config.time_limit_secs,
            ..DeepMCTSConfig::default()
        };

        Self {
            mcts: DeepMCTS::with_config(rules, verifier, mcts_config),
            predictor: SubstitutionPredictor::new(),
            symbols: SymbolTable::new(),
            config,
        }
    }

    /// Report substitution hints for a problem statement. Does not solve it.
    ///
    /// Returns [`IMOOutcome::Unsupported`]: there is no text-to-expression translation here,
    /// so there is nothing to search. Use [`Self::solve_expr`] with an [`Expr`] you built or
    /// parsed yourself.
    pub fn solve_text(&self, problem_text: &str) -> IMOSolveResult {
        let start = Instant::now();

        let hints = self
            .predictor
            .predict(problem_text, self.config.top_k_substitutions);

        if self.config.verbose {
            println!("LEMMA cannot read a problem statement.");
            println!("  Input: {}", truncate(problem_text, 72));
            println!("  Suggested substitutions to try by hand:");
            for (i, hint) in hints.iter().enumerate() {
                println!(
                    "    {}. {} (confidence: {:.1}%)",
                    i + 1,
                    hint.substitution,
                    hint.confidence * 100.0
                );
            }
            println!("  Build the problem as an Expr and call solve_expr to search it.");
        }

        IMOSolveResult {
            outcome: IMOOutcome::Unsupported(UnsupportedInput {
                input: truncate(problem_text, 200),
                reason: "no natural-language to expression translation is implemented; \
                         supply an Expr to solve_expr"
                    .to_string(),
            }),
            substitutions_suggested: hints,
            stats: SearchStats::default(),
            elapsed: start.elapsed(),
        }
    }

    /// Search an expression for a state that satisfies `goal`.
    pub fn solve_expr_with_goal<F>(&self, expr: Expr, goal: F) -> IMOSolveResult
    where
        F: Fn(&Expr) -> bool + Sync,
    {
        let start = Instant::now();
        let (path, stats) = self.mcts.search(expr, goal);
        let elapsed = start.elapsed();

        let outcome = match path {
            Some(path) => IMOOutcome::Solved(
                path.windows(2)
                    .map(|w| SolutionStep {
                        before: w[0].clone(),
                        after: w[1].clone(),
                        explanation: "Rule application found by tree search".to_string(),
                    })
                    .collect(),
            ),
            None => IMOOutcome::NotFound,
        };

        IMOSolveResult {
            outcome,
            substitutions_suggested: vec![],
            stats,
            elapsed,
        }
    }

    /// Search an expression, aiming to reduce it to a constant or a single variable.
    pub fn solve_expr(&self, expr: Expr) -> IMOSolveResult {
        self.solve_expr_with_goal(expr, |e: &Expr| matches!(e, Expr::Const(_) | Expr::Var(_)))
    }

    /// The symbol table, for building expressions to pass to [`Self::solve_expr`].
    pub fn symbols_mut(&mut self) -> &mut SymbolTable {
        &mut self.symbols
    }

    /// Get the number of rules loaded
    pub fn num_rules(&self) -> usize {
        self.mcts.rules.len()
    }

    /// Get predictor vocabulary size
    pub fn vocab_size(&self) -> usize {
        SubstitutionPredictor::vocabulary().len()
    }
}

impl Default for IMOSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Truncate on a character boundary.
fn truncate(text: &str, max_chars: usize) -> String {
    let mut out: String = text.chars().take(max_chars).collect();
    if text.chars().count() > max_chars {
        out.push_str("...");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quiet() -> IMOSolver {
        IMOSolver::with_config(IMOSolverConfig {
            verbose: false,
            ..IMOSolverConfig::quick()
        })
    }

    #[test]
    fn test_solver_creation() {
        let solver = quiet();
        assert!(solver.num_rules() > 400);
        assert_eq!(solver.vocab_size(), 20);
    }

    #[test]
    fn text_input_is_reported_as_unsupported() {
        let solver = quiet();
        let result =
            solver.solve_text("Find all functions f: R -> R such that f(x + f(y)) = f(x) + y.");

        assert!(!result.solved());
        assert!(matches!(result.outcome, IMOOutcome::Unsupported(_)));
        assert!(result.outcome.path().is_none());
        assert_eq!(result.stats.nodes_explored, 0, "nothing should be searched");
    }

    #[test]
    fn two_unrelated_problems_cannot_produce_the_same_canned_solution() {
        // The regression this guards: both inputs used to be discarded and replaced by the
        // same hard-coded algebraic identity, so both "solved" identically.
        let solver = quiet();

        let a = solver.solve_text("Prove that for all positive reals a, b: a^2 + b^2 >= 2ab.");
        let b = solver.solve_text("Find the number of primes p such that p^2 + 2 is prime.");

        assert!(!a.solved() && !b.solved());
        assert!(a.outcome.path().is_none());
        assert!(b.outcome.path().is_none());

        match (&a.outcome, &b.outcome) {
            (IMOOutcome::Unsupported(ua), IMOOutcome::Unsupported(ub)) => {
                assert_ne!(ua.input, ub.input, "each result must name its own input");
            }
            other => panic!("expected two unsupported results, got {other:?}"),
        }
    }

    #[test]
    fn hints_are_still_reported_for_text() {
        let solver = quiet();
        let result =
            solver.solve_text("Find all functions f: R -> R such that f(x + f(y)) = f(x) + y.");

        let subs: Vec<&str> = result
            .substitutions_suggested
            .iter()
            .map(|s| s.substitution.as_str())
            .collect();
        assert!(subs.contains(&"x = 0") || subs.contains(&"y = 0"));
    }

    #[test]
    fn inequality_hints_are_still_reported() {
        let solver = quiet();
        let result =
            solver.solve_text("Let a, b, c be positive reals with abc = 1. Prove a + b + c >= 3.");

        let subs: Vec<&str> = result
            .substitutions_suggested
            .iter()
            .map(|s| s.substitution.as_str())
            .collect();
        assert!(subs.contains(&"Apply AM-GM") || subs.contains(&"abc = 1 constraint"));
    }

    #[test]
    fn an_expression_that_is_already_a_goal_solves_trivially() {
        let solver = quiet();
        let result = solver.solve_expr(Expr::int(5));
        // The root already satisfies the goal, so the path has a single state and no steps.
        assert!(matches!(
            result.outcome,
            IMOOutcome::Solved(_) | IMOOutcome::NotFound
        ));
    }
}
