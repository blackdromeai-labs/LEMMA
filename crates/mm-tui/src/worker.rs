// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Author: Pushp Kharat

//! The solver boundary.
//!
//! LEMMA's search is CPU-bound and synchronous, and a simplify can take long enough to be
//! noticed. Running it on the render thread would freeze both drawing and key handling, so it
//! runs on one worker thread reached through channels. That is the whole reason this module
//! exists; there is no async runtime.
//!
//! The worker owns the `LemmaSolver`, and therefore owns the `SymbolTable` that gives
//! `Symbol` values their names. It formats every expression before replying, so what crosses
//! the channel is owned `String` data that stays correct no matter what the render thread
//! does next.
//!
//! Requests carry an id and replies echo it. The UI drops replies whose id is not the one it
//! is waiting for, so a slow job that finishes after the user has moved on cannot overwrite a
//! newer result.

use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use mm_core::{format_expr, Expr, MathError};
use mm_search::Step;
use mm_solver::LemmaSolver;
use mm_verifier::VerifyResult;

use crate::presentation::{
    badge_for_status, badge_for_verify_result, ErrorField, Mode, StatusBadge, UiError, UiEvidence,
    UiResult, UiStep,
};

/// A unit of work for the solver thread.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolveRequest {
    /// Monotonic id; the reply echoes it so stale replies can be dropped.
    pub id: u64,
    /// Operation to perform.
    pub mode: Mode,
    /// Expression source text.
    pub expression: String,
    /// Variable name, read by Differentiate and Verify candidate.
    pub variable: String,
    /// Candidate value source text, read by Verify candidate.
    pub candidate: String,
}

/// A reply from the solver thread.
#[derive(Debug, Clone, PartialEq)]
pub struct SolveResponse {
    /// Id of the request this answers.
    pub id: u64,
    /// Wall-clock time spent on it.
    pub elapsed: Duration,
    /// The rendered result, or the input error that stopped it.
    pub outcome: Result<UiResult, UiError>,
}

/// Handle to the solver thread.
pub struct Worker {
    requests: Sender<WorkerMessage>,
    responses: Receiver<SolveResponse>,
    handle: Option<JoinHandle<()>>,
}

enum WorkerMessage {
    Solve(Box<SolveRequest>),
    Shutdown,
}

impl Worker {
    /// Start the solver thread.
    ///
    /// Building `LemmaSolver` loads the whole rule registry, which takes long enough to be
    /// visible at startup, so it happens on the worker rather than before the first frame.
    pub fn spawn() -> Self {
        let (request_tx, request_rx) = std::sync::mpsc::channel::<WorkerMessage>();
        let (response_tx, response_rx) = std::sync::mpsc::channel::<SolveResponse>();

        let handle = std::thread::Builder::new()
            .name("mm-tui-solver".to_string())
            .spawn(move || run(request_rx, response_tx))
            .expect("failed to spawn the solver thread");

        Self {
            requests: request_tx,
            responses: response_rx,
            handle: Some(handle),
        }
    }

    /// Queue a request. Returns `false` if the worker has gone away.
    pub fn submit(&self, request: SolveRequest) -> bool {
        self.requests
            .send(WorkerMessage::Solve(Box::new(request)))
            .is_ok()
    }

    /// Take any replies that have arrived, without blocking.
    pub fn drain(&self) -> Vec<SolveResponse> {
        self.responses.try_iter().collect()
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        // Ask the worker to stop, then wait briefly. A job already running cannot be
        // interrupted — the search takes no cancellation token — so this does not claim to
        // cancel anything, it just avoids detaching the thread silently.
        let _ = self.requests.send(WorkerMessage::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn run(requests: Receiver<WorkerMessage>, responses: Sender<SolveResponse>) {
    let mut solver = LemmaSolver::new();

    loop {
        // A timeout rather than a blocking recv, so a dropped sender cannot wedge the thread.
        match requests.recv_timeout(Duration::from_millis(200)) {
            Ok(WorkerMessage::Solve(request)) => {
                let started = Instant::now();
                let outcome = execute(&mut solver, &request);
                let response = SolveResponse {
                    id: request.id,
                    elapsed: started.elapsed(),
                    outcome,
                };
                if responses.send(response).is_err() {
                    return;
                }
            }
            Ok(WorkerMessage::Shutdown) | Err(RecvTimeoutError::Disconnected) => return,
            Err(RecvTimeoutError::Timeout) => {}
        }
    }
}

/// Run one request against the solver and render the outcome.
fn execute(solver: &mut LemmaSolver, request: &SolveRequest) -> Result<UiResult, UiError> {
    match request.mode {
        Mode::Simplify => simplify(solver, request),
        Mode::Differentiate => differentiate(solver, request),
        Mode::VerifyCandidate => verify_candidate(solver, request),
    }
}

fn simplify(solver: &mut LemmaSolver, request: &SolveRequest) -> Result<UiResult, UiError> {
    let expression = require_expression(&request.expression)?;
    let parsed = parse_field(solver, expression, ErrorField::Expression)?;

    let result = solver.simplify_expr(parsed.clone());
    let (badge, reason) = badge_for_status(&result.status);

    Ok(UiResult {
        mode: request.mode,
        input: render(solver, &parsed),
        output: render(solver, &result.result),
        badge,
        reason,
        steps: render_steps(solver, &result.steps),
        elapsed: Duration::ZERO,
    })
}

fn differentiate(solver: &mut LemmaSolver, request: &SolveRequest) -> Result<UiResult, UiError> {
    let expression = require_expression(&request.expression)?;
    let variable = require_variable(&request.variable)?;

    // Parse first so a bad expression is reported against the expression field, then build the
    // derivative node directly. `LemmaSolver::differentiate` re-parses from a string, which
    // would attribute a parse error to the wrong place.
    let parsed = parse_field(solver, expression, ErrorField::Expression)?;
    let var_symbol = solver.symbols_mut().intern(variable);
    let derivative = Expr::Derivative {
        expr: Box::new(parsed),
        var: var_symbol,
    };

    let result = solver.simplify_expr(derivative.clone());
    let (badge, reason) = badge_for_status(&result.status);

    Ok(UiResult {
        mode: request.mode,
        input: render(solver, &derivative),
        output: render(solver, &result.result),
        badge,
        reason,
        steps: render_steps(solver, &result.steps),
        elapsed: Duration::ZERO,
    })
}

fn verify_candidate(solver: &mut LemmaSolver, request: &SolveRequest) -> Result<UiResult, UiError> {
    let expression = require_expression(&request.expression)?;
    let variable = require_variable(&request.variable)?;
    let candidate_text = request.candidate.trim();
    if candidate_text.is_empty() {
        return Err(UiError {
            field: ErrorField::Candidate,
            message: "Enter a candidate value to check.".to_string(),
        });
    }

    let equation = parse_field(solver, expression, ErrorField::Expression)?;
    let Expr::Equation { .. } = equation else {
        return Err(UiError {
            field: ErrorField::Expression,
            message: "Verify candidate needs an equation, for example x + 3 = 7.".to_string(),
        });
    };

    let candidate = parse_field(solver, candidate_text, ErrorField::Candidate)?;
    let var_symbol = solver.symbols_mut().intern(variable);

    let verdict: VerifyResult = solver
        .verifier()
        .verify_solution(&equation, var_symbol, &candidate);
    let (badge, reason) = badge_for_verify_result(&verdict);

    let equation_text = render(solver, &equation);
    let candidate_text = render(solver, &candidate);

    Ok(UiResult {
        mode: request.mode,
        input: equation_text,
        // A candidate check produces a verdict about a substitution, not a new expression.
        // Saying so is more honest than echoing the candidate as though it were an answer.
        output: format!("{variable} = {candidate_text}"),
        badge,
        reason,
        // `verify_solution` substitutes and compares; it records no transformation steps, and
        // inventing some would misrepresent what happened.
        steps: Vec::new(),
        elapsed: Duration::ZERO,
    })
}

fn require_expression(text: &str) -> Result<&str, UiError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(UiError {
            field: ErrorField::Expression,
            message: "Enter an expression in LEMMA syntax.".to_string(),
        });
    }
    Ok(trimmed)
}

fn require_variable(text: &str) -> Result<&str, UiError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(UiError {
            field: ErrorField::Variable,
            message: "Enter the variable to work with, for example x.".to_string(),
        });
    }
    if !trimmed.chars().all(|c| c.is_alphanumeric() || c == '_')
        || trimmed.chars().next().is_some_and(|c| c.is_ascii_digit())
    {
        return Err(UiError {
            field: ErrorField::Variable,
            message: format!("{trimmed:?} is not a variable name."),
        });
    }
    Ok(trimmed)
}

fn parse_field(solver: &mut LemmaSolver, text: &str, field: ErrorField) -> Result<Expr, UiError> {
    solver.parse(text).map_err(|error| UiError {
        field,
        message: describe_parse_error(&error),
    })
}

/// Turn a domain error into something a user can act on.
fn describe_parse_error(error: &MathError) -> String {
    match error {
        MathError::ParseError(detail) => detail.clone(),
        other => other.to_string(),
    }
}

fn render(solver: &LemmaSolver, expr: &Expr) -> String {
    format_expr(expr, solver.symbols())
}

fn render_steps(solver: &LemmaSolver, steps: &[Step]) -> Vec<UiStep> {
    steps
        .iter()
        .enumerate()
        .map(|(index, step)| {
            let rule = if step.is_rule_application() {
                qualified_rule_name(solver, step)
            } else {
                // Normalisation steps are not registry rules; naming them as one would give
                // them an identity they do not have.
                format!("normalisation::{}", step.rule_name)
            };

            UiStep {
                index: index + 1,
                before: render(solver, &step.before),
                after: render(solver, &step.after),
                rule,
                rule_id: step.is_rule_application().then_some(step.rule_id.0),
                justification: step.justification.clone(),
                evidence: UiEvidence::from_step(step.evidence),
            }
        })
        .collect()
}

/// Resolve a step's rule to its stable `module::name` key.
fn qualified_rule_name(solver: &LemmaSolver, step: &Step) -> String {
    solver
        .rules()
        .key_of(step.rule_id)
        .map(|key| key.to_string())
        .unwrap_or_else(|| step.rule_name.to_string())
}

/// Build the error shown when a request is rejected before it reaches the solver.
pub fn input_error_result(mode: Mode, error: &UiError, elapsed: Duration) -> UiResult {
    UiResult {
        mode,
        input: String::new(),
        output: String::new(),
        badge: StatusBadge::InputError,
        reason: format!("{}: {}", error.field.label(), error.message),
        steps: Vec::new(),
        elapsed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(mode: Mode, expression: &str, variable: &str, candidate: &str) -> SolveRequest {
        SolveRequest {
            id: 1,
            mode,
            expression: expression.to_string(),
            variable: variable.to_string(),
            candidate: candidate.to_string(),
        }
    }

    fn run_one(request: SolveRequest) -> Result<UiResult, UiError> {
        let mut solver = LemmaSolver::new();
        execute(&mut solver, &request)
    }

    #[test]
    fn simplify_renders_readable_maths_not_debug_output() {
        let result = run_one(request(Mode::Simplify, "(x + 0) * 1", "", "")).unwrap();

        assert_eq!(result.input, "(x + 0) * 1");
        assert!(
            !result.output.contains("Var("),
            "output leaked a Rust AST: {}",
            result.output
        );
        assert!(!result.output.contains("SymbolU32"));
    }

    #[test]
    fn simplify_reports_the_domain_status() {
        let result = run_one(request(Mode::Simplify, "2 + 3", "", "")).unwrap();
        assert_eq!(result.output, "5");
        // Whatever the status is, it must carry a reason and be a real badge.
        assert!(!result.reason.is_empty());
        assert!(!result.badge.label().is_empty());
    }

    #[test]
    fn every_step_carries_identity_and_evidence() {
        let result = run_one(request(Mode::Simplify, "(x + 0) * 1", "", "")).unwrap();

        for step in &result.steps {
            assert!(!step.rule.is_empty());
            assert!(!step.before.is_empty());
            assert!(!step.after.is_empty());
            assert!(!step.evidence.label.is_empty());
            assert!(
                step.rule.contains("::"),
                "rule identity must be qualified, got {}",
                step.rule
            );
        }
    }

    #[test]
    fn differentiate_builds_the_derivative_from_the_parsed_expression() {
        let result = run_one(request(Mode::Differentiate, "x^3", "x", "")).unwrap();
        assert_eq!(result.input, "diff(x^3, x)");
        assert!(!result.output.is_empty());
    }

    #[test]
    fn differentiate_requires_a_variable() {
        let error = run_one(request(Mode::Differentiate, "x^3", "", "")).unwrap_err();
        assert_eq!(error.field, ErrorField::Variable);
    }

    #[test]
    fn a_parse_error_is_attributed_to_its_field() {
        let error = run_one(request(Mode::Simplify, "x +", "", "")).unwrap_err();
        assert_eq!(error.field, ErrorField::Expression);
        assert!(!error.message.is_empty());

        let error = run_one(request(Mode::VerifyCandidate, "x + 3 = 7", "x", "((")).unwrap_err();
        assert_eq!(error.field, ErrorField::Candidate);
    }

    #[test]
    fn an_empty_expression_is_an_input_error_not_a_solver_run() {
        let error = run_one(request(Mode::Simplify, "   ", "", "")).unwrap_err();
        assert_eq!(error.field, ErrorField::Expression);
    }

    #[test]
    fn a_variable_field_rejects_things_that_are_not_names() {
        let error = run_one(request(Mode::Differentiate, "x^2", "2x", "")).unwrap_err();
        assert_eq!(error.field, ErrorField::Variable);
    }

    #[test]
    fn verify_candidate_accepts_a_true_candidate() {
        let result = run_one(request(Mode::VerifyCandidate, "x + 3 = 7", "x", "4")).unwrap();
        assert_eq!(result.badge, StatusBadge::CandidateValid);
        assert_eq!(result.output, "x = 4");
        assert!(
            result.steps.is_empty(),
            "a candidate check records no transformation trace"
        );
    }

    #[test]
    fn verify_candidate_rejects_a_false_candidate() {
        let result = run_one(request(Mode::VerifyCandidate, "x + 3 = 7", "x", "5")).unwrap();
        assert_eq!(result.badge, StatusBadge::CandidateInvalid);
        assert!(!result.badge.is_verified());
    }

    #[test]
    fn verify_candidate_needs_an_equation() {
        let error = run_one(request(Mode::VerifyCandidate, "x + 3", "x", "4")).unwrap_err();
        assert_eq!(error.field, ErrorField::Expression);
        assert!(error.message.contains("equation"));
    }

    #[test]
    fn the_worker_answers_on_its_own_thread_and_echoes_the_request_id() {
        let worker = Worker::spawn();
        assert!(worker.submit(SolveRequest {
            id: 77,
            mode: Mode::Simplify,
            expression: "2 + 3".to_string(),
            variable: String::new(),
            candidate: String::new(),
        }));

        // Poll rather than block, the way the render loop does.
        let deadline = Instant::now() + Duration::from_secs(60);
        let response = loop {
            if let Some(response) = worker.drain().into_iter().next() {
                break response;
            }
            assert!(Instant::now() < deadline, "worker did not reply in time");
            std::thread::sleep(Duration::from_millis(20));
        };

        assert_eq!(response.id, 77);
        assert_eq!(response.outcome.unwrap().output, "5");
    }

    #[test]
    fn input_errors_become_an_input_error_badge() {
        let error = UiError {
            field: ErrorField::Expression,
            message: "Unexpected end of input".to_string(),
        };
        let result = input_error_result(Mode::Simplify, &error, Duration::from_millis(1));
        assert_eq!(result.badge, StatusBadge::InputError);
        assert!(result.reason.contains("expression"));
        assert!(!result.badge.is_verified());
    }
}
