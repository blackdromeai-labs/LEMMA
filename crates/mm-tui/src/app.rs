// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Author: Pushp Kharat

//! Application state and the reducer that moves it.
//!
//! Rendering is immediate-mode, so all of the state lives here and `ui` is a pure function of
//! it. Keys become [`Action`]s first, which keeps the key map testable without a terminal and
//! keeps the "what does this do" decision in one place.

use std::collections::VecDeque;
use std::time::Duration;

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui_textarea::TextArea;

use crate::presentation::{ErrorField, Mode, StatusBadge, UiError, UiResult};
use crate::worker::{SolveRequest, SolveResponse};

/// How many past requests the history drawer keeps. In memory only.
pub const HISTORY_LIMIT: usize = 20;

/// Smallest terminal this UI will draw in. Below it, a notice is shown instead of a broken
/// layout.
pub const MIN_WIDTH: u16 = 80;
/// Minimum terminal height. See [`MIN_WIDTH`].
pub const MIN_HEIGHT: u16 = 24;
/// At or above this width the result, trace and detail panes sit side by side.
pub const WIDE_WIDTH: u16 = 110;

/// Which pane has keyboard focus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    /// The expression editor.
    Expression,
    /// The variable editor.
    Variable,
    /// The candidate editor.
    Candidate,
    /// The trace list.
    Trace,
}

impl Focus {
    /// Whether a text editor is consuming keys.
    ///
    /// While one is, bare letters must reach the editor rather than triggering shortcuts.
    pub fn is_editor(self) -> bool {
        !matches!(self, Focus::Trace)
    }
}

/// What the solver is doing.
#[derive(Debug, Clone, PartialEq)]
pub enum Job {
    /// Nothing has been run yet.
    Idle,
    /// A request is in flight.
    Running {
        /// Id of the request being awaited.
        id: u64,
    },
    /// A request finished.
    Complete(Box<UiResult>),
    /// A request was rejected before it reached the solver.
    Failed(UiError),
}

/// A modal layer drawn over the workbench.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Overlay {
    /// Key reference.
    Help,
    /// Recent requests.
    History,
}

/// One remembered request.
#[derive(Debug, Clone, PartialEq)]
pub struct HistoryEntry {
    /// Operation that was run.
    pub mode: Mode,
    /// Expression source text.
    pub expression: String,
    /// Variable source text.
    pub variable: String,
    /// Candidate source text.
    pub candidate: String,
    /// The completed result, so re-selecting it restores the trace without re-solving.
    pub result: UiResult,
}

/// Something the UI should do, derived from an input event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Leave the application.
    Quit,
    /// Switch operation.
    SetMode(Mode),
    /// Move focus forward.
    FocusNext,
    /// Move focus backward.
    FocusPrev,
    /// Run the current form.
    Submit,
    /// Empty the form and the result.
    Clear,
    /// Move the trace selection.
    TraceUp,
    /// Move the trace selection.
    TraceDown,
    /// Move the trace selection a page at a time.
    TracePageUp,
    /// Move the trace selection a page at a time.
    TracePageDown,
    /// Select the first trace step.
    TraceHome,
    /// Select the last trace step.
    TraceEnd,
    /// Show or hide the help overlay.
    ToggleHelp,
    /// Show or hide the history overlay.
    ToggleHistory,
    /// Dismiss the overlay, or step out of the focused pane.
    Escape,
    /// Restore the highlighted history entry.
    RecallHistory,
    /// Give the key to the focused editor.
    Edit(KeyEvent),
}

/// The whole application state.
pub struct App {
    /// Selected operation.
    pub mode: Mode,
    /// Focused pane.
    pub focus: Focus,
    /// Expression editor.
    pub expression: TextArea<'static>,
    /// Variable editor.
    pub variable: TextArea<'static>,
    /// Candidate editor.
    pub candidate: TextArea<'static>,
    /// Solver state.
    pub job: Job,
    /// Index of the highlighted trace step.
    pub selected_step: usize,
    /// Recent requests, newest first.
    pub history: VecDeque<HistoryEntry>,
    /// Highlighted row in the history overlay.
    pub history_cursor: usize,
    /// Active overlay, if any.
    pub overlay: Option<Overlay>,
    /// Whether the event loop should exit.
    pub should_quit: bool,
    /// Number of rules the solver reported, shown in the header.
    pub rule_count: usize,
    /// Frame counter, used only for the running indicator.
    pub ticks: u64,
    next_request_id: u64,
    /// The request currently in flight.
    ///
    /// History is written when the reply arrives, and by then the editors may hold something
    /// else entirely — the user is free to type while a job runs. Recording the live fields
    /// would file the new text against the old result. Keeping the request means history
    /// records what was actually asked.
    pending: Option<SolveRequest>,
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    /// A fresh workbench.
    pub fn new() -> Self {
        Self {
            mode: Mode::Simplify,
            focus: Focus::Expression,
            expression: single_line_editor(""),
            variable: single_line_editor("x"),
            candidate: single_line_editor(""),
            job: Job::Idle,
            selected_step: 0,
            history: VecDeque::new(),
            history_cursor: 0,
            overlay: None,
            should_quit: false,
            rule_count: 0,
            ticks: 0,
            next_request_id: 0,
            pending: None,
        }
    }

    /// Current contents of an editor, trimmed of the trailing newline `TextArea` keeps.
    pub fn field(&self, focus: Focus) -> String {
        let area = match focus {
            Focus::Expression => &self.expression,
            Focus::Variable => &self.variable,
            Focus::Candidate => &self.candidate,
            Focus::Trace => return String::new(),
        };
        area.lines().join(" ")
    }

    /// The focus order for the current mode, skipping fields the mode does not read.
    pub fn focus_order(&self) -> Vec<Focus> {
        let mut order = vec![Focus::Expression];
        if self.mode.uses_variable() {
            order.push(Focus::Variable);
        }
        if self.mode.uses_candidate() {
            order.push(Focus::Candidate);
        }
        order.push(Focus::Trace);
        order
    }

    /// The completed result, if there is one.
    pub fn result(&self) -> Option<&UiResult> {
        match &self.job {
            Job::Complete(result) => Some(result),
            _ => None,
        }
    }

    /// Number of steps in the current trace.
    pub fn step_count(&self) -> usize {
        self.result().map_or(0, |r| r.steps.len())
    }

    /// Whether a request is in flight.
    pub fn is_running(&self) -> bool {
        matches!(self.job, Job::Running { .. })
    }

    /// Apply an action. Returns a request to submit, if the action asked for one.
    pub fn apply(&mut self, action: Action) -> Option<SolveRequest> {
        match action {
            Action::Quit => self.should_quit = true,
            Action::SetMode(mode) => self.set_mode(mode),
            Action::FocusNext => self.cycle_focus(1),
            Action::FocusPrev => self.cycle_focus(-1),
            Action::Submit => return self.submit(),
            Action::Clear => self.clear(),
            Action::TraceUp => self.move_selection(-1),
            Action::TraceDown => self.move_selection(1),
            Action::TracePageUp => self.move_selection(-5),
            Action::TracePageDown => self.move_selection(5),
            Action::TraceHome => self.set_selection(0),
            Action::TraceEnd => self.set_selection(self.step_count().saturating_sub(1)),
            Action::ToggleHelp => self.toggle_overlay(Overlay::Help),
            Action::ToggleHistory => self.toggle_overlay(Overlay::History),
            Action::Escape => self.escape(),
            Action::RecallHistory => self.recall_history(),
            Action::Edit(key) => self.edit(key),
        }
        None
    }

    fn set_mode(&mut self, mode: Mode) {
        if self.mode == mode {
            return;
        }
        self.mode = mode;
        // The old result described a different operation, so keeping it on screen beside a new
        // mode label would misattribute it.
        self.job = Job::Idle;
        self.selected_step = 0;
        if !self.focus_order().contains(&self.focus) {
            self.focus = Focus::Expression;
        }
    }

    fn cycle_focus(&mut self, delta: isize) {
        let order = self.focus_order();
        let current = order.iter().position(|f| *f == self.focus).unwrap_or(0);
        let len = order.len() as isize;
        let next = (current as isize + delta).rem_euclid(len) as usize;
        self.focus = order[next];
    }

    fn submit(&mut self) -> Option<SolveRequest> {
        // One job at a time. The search takes no cancellation token, so starting a second
        // would leave the first consuming a core with nowhere to report.
        if self.is_running() {
            return None;
        }

        self.next_request_id += 1;
        let id = self.next_request_id;
        self.job = Job::Running { id };
        self.selected_step = 0;
        self.overlay = None;

        let request = SolveRequest {
            id,
            mode: self.mode,
            expression: self.field(Focus::Expression),
            variable: self.field(Focus::Variable),
            candidate: self.field(Focus::Candidate),
        };
        self.pending = Some(request.clone());
        Some(request)
    }

    fn clear(&mut self) {
        self.expression = single_line_editor("");
        self.candidate = single_line_editor("");
        self.job = Job::Idle;
        self.pending = None;
        self.selected_step = 0;
        self.focus = Focus::Expression;
    }

    fn move_selection(&mut self, delta: isize) {
        let count = self.step_count();
        if count == 0 {
            self.selected_step = 0;
            return;
        }
        let next = (self.selected_step as isize + delta).clamp(0, count as isize - 1);
        self.selected_step = next as usize;
    }

    fn set_selection(&mut self, index: usize) {
        let count = self.step_count();
        self.selected_step = if count == 0 { 0 } else { index.min(count - 1) };
    }

    fn toggle_overlay(&mut self, overlay: Overlay) {
        self.overlay = if self.overlay == Some(overlay) {
            None
        } else {
            if overlay == Overlay::History {
                self.history_cursor = 0;
            }
            Some(overlay)
        };
    }

    fn escape(&mut self) {
        if self.overlay.is_some() {
            self.overlay = None;
        } else if self.focus == Focus::Trace {
            self.focus = Focus::Expression;
        }
    }

    fn recall_history(&mut self) {
        let Some(entry) = self.history.get(self.history_cursor).cloned() else {
            return;
        };
        self.mode = entry.mode;
        self.expression = single_line_editor(&entry.expression);
        self.variable = single_line_editor(&entry.variable);
        self.candidate = single_line_editor(&entry.candidate);
        self.job = Job::Complete(Box::new(entry.result));
        self.selected_step = 0;
        self.overlay = None;
        self.focus = Focus::Expression;
    }

    fn edit(&mut self, key: KeyEvent) {
        // `ratatui-textarea` accepts a Crossterm key event directly. Single-line editors
        // ignore Enter so a stray newline cannot turn a field into two lines.
        if key.code == KeyCode::Enter {
            return;
        }
        let area = match self.focus {
            Focus::Expression => &mut self.expression,
            Focus::Variable => &mut self.variable,
            Focus::Candidate => &mut self.candidate,
            Focus::Trace => return,
        };
        area.input(key);
    }

    /// Move the history cursor while the overlay is open.
    pub fn move_history_cursor(&mut self, delta: isize) {
        if self.history.is_empty() {
            self.history_cursor = 0;
            return;
        }
        let last = self.history.len() as isize - 1;
        self.history_cursor = (self.history_cursor as isize + delta).clamp(0, last) as usize;
    }

    /// Accept a reply from the worker.
    ///
    /// Replies whose id is not the one being awaited are dropped: they belong to a request the
    /// user has already replaced.
    pub fn accept(&mut self, response: SolveResponse) {
        let Job::Running { id } = self.job else {
            return;
        };
        if response.id != id {
            return;
        }

        let request = self.pending.take();

        match response.outcome {
            Ok(mut result) => {
                result.elapsed = response.elapsed;
                if let Some(request) = request {
                    self.remember(&request, &result);
                }
                self.job = Job::Complete(Box::new(result));
            }
            Err(error) => self.job = Job::Failed(error),
        }
        self.selected_step = 0;
    }

    /// File a completed request in history, described by the inputs it was sent with.
    fn remember(&mut self, request: &SolveRequest, result: &UiResult) {
        self.history.push_front(HistoryEntry {
            mode: request.mode,
            expression: request.expression.clone(),
            variable: request.variable.clone(),
            candidate: request.candidate.clone(),
            result: result.clone(),
        });
        while self.history.len() > HISTORY_LIMIT {
            self.history.pop_back();
        }
    }

    /// The badge to draw for the current job, if any.
    pub fn badge(&self) -> Option<StatusBadge> {
        match &self.job {
            Job::Complete(result) => Some(result.badge),
            Job::Failed(_) => Some(StatusBadge::InputError),
            _ => None,
        }
    }

    /// The field an input error belongs to, for highlighting.
    pub fn error_field(&self) -> Option<ErrorField> {
        match &self.job {
            Job::Failed(error) => Some(error.field),
            _ => None,
        }
    }

    /// Elapsed time of the completed request.
    pub fn elapsed(&self) -> Option<Duration> {
        self.result().map(|r| r.elapsed)
    }
}

/// Build a single-line editor holding `text`.
fn single_line_editor(text: &str) -> TextArea<'static> {
    let mut area = TextArea::new(vec![text.to_string()]);
    area.move_cursor(ratatui_textarea::CursorMove::End);
    area
}

/// Translate a key event into an action.
///
/// Returns `None` for events that should be ignored, which includes every non-press event:
/// on Windows, Crossterm reports press *and* release, and treating both as input would submit
/// each job twice.
pub fn action_for_key(app: &App, key: KeyEvent) -> Option<Action> {
    if key.kind != KeyEventKind::Press {
        return None;
    }

    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);

    // Ctrl chords work everywhere, including inside an editor.
    if ctrl {
        return match key.code {
            KeyCode::Char('q') | KeyCode::Char('Q') => Some(Action::Quit),
            KeyCode::Char('l') | KeyCode::Char('L') => Some(Action::Clear),
            KeyCode::Enter | KeyCode::Char('r') | KeyCode::Char('R') => Some(Action::Submit),
            _ => Some(Action::Edit(key)),
        };
    }

    // The history overlay is a list, so arrows drive it rather than the trace.
    if app.overlay == Some(Overlay::History) {
        return match key.code {
            KeyCode::Esc | KeyCode::Char('h') => Some(Action::ToggleHistory),
            KeyCode::Char('?') => Some(Action::ToggleHelp),
            KeyCode::Enter => Some(Action::RecallHistory),
            _ => None,
        };
    }
    if app.overlay == Some(Overlay::Help) {
        return match key.code {
            KeyCode::Esc | KeyCode::Char('?') => Some(Action::ToggleHelp),
            _ => None,
        };
    }

    match key.code {
        KeyCode::Tab => Some(Action::FocusNext),
        KeyCode::BackTab => Some(Action::FocusPrev),
        KeyCode::Esc => Some(Action::Escape),
        KeyCode::Up => Some(Action::TraceUp),
        KeyCode::Down => Some(Action::TraceDown),
        KeyCode::PageUp => Some(Action::TracePageUp),
        KeyCode::PageDown => Some(Action::TracePageDown),

        // Home/End belong to the editor when one is focused, and to the trace otherwise.
        KeyCode::Home if !app.focus.is_editor() => Some(Action::TraceHome),
        KeyCode::End if !app.focus.is_editor() => Some(Action::TraceEnd),

        // Bare letters are shortcuts only when no editor is consuming input. Otherwise `q`
        // would quit in the middle of typing `sqrt(x)`.
        KeyCode::Char('1') if !app.focus.is_editor() => Some(Action::SetMode(Mode::Simplify)),
        KeyCode::Char('2') if !app.focus.is_editor() => Some(Action::SetMode(Mode::Differentiate)),
        KeyCode::Char('3') if !app.focus.is_editor() => {
            Some(Action::SetMode(Mode::VerifyCandidate))
        }
        KeyCode::Char('q') if !app.focus.is_editor() => Some(Action::Quit),
        KeyCode::Char('h') if !app.focus.is_editor() => Some(Action::ToggleHistory),

        // `?` opens help from anywhere: it is not a character any LEMMA expression uses.
        KeyCode::Char('?') => Some(Action::ToggleHelp),

        KeyCode::Enter if !app.focus.is_editor() => Some(Action::Submit),

        _ if app.focus.is_editor() => Some(Action::Edit(key)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presentation::{StatusBadge, UiStep};

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn ctrl(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::CONTROL)
    }

    fn result_with_steps(count: usize) -> UiResult {
        UiResult {
            mode: Mode::Simplify,
            input: "x + 0".to_string(),
            output: "x".to_string(),
            badge: StatusBadge::Checked,
            reason: "checked".to_string(),
            steps: (1..=count)
                .map(|index| UiStep {
                    index,
                    before: "x + 0".to_string(),
                    after: "x".to_string(),
                    rule: "algebra::identity_add_zero".to_string(),
                    rule_id: Some(2),
                    justification: "identity".to_string(),
                    evidence: crate::presentation::UiEvidence {
                        label: "checked (symbolic equivalence)".to_string(),
                        checked: true,
                        independent: true,
                    },
                })
                .collect(),
            elapsed: Duration::from_millis(12),
        }
    }

    #[test]
    fn key_release_events_are_ignored() {
        // Windows terminals report press and release. Acting on both submits every job twice.
        let app = App::new();
        let mut release = ctrl(KeyCode::Enter);
        release.kind = KeyEventKind::Release;
        assert_eq!(action_for_key(&app, release), None);

        let mut repeat = ctrl(KeyCode::Enter);
        repeat.kind = KeyEventKind::Repeat;
        assert_eq!(action_for_key(&app, repeat), None);

        assert_eq!(
            action_for_key(&app, ctrl(KeyCode::Enter)),
            Some(Action::Submit)
        );
    }

    #[test]
    fn letters_reach_the_editor_instead_of_triggering_shortcuts() {
        let mut app = App::new();
        app.focus = Focus::Expression;

        // `q` while typing must not quit, and `1` must not switch mode.
        assert!(matches!(
            action_for_key(&app, press(KeyCode::Char('q'))),
            Some(Action::Edit(_))
        ));
        assert!(matches!(
            action_for_key(&app, press(KeyCode::Char('1'))),
            Some(Action::Edit(_))
        ));

        app.focus = Focus::Trace;
        assert_eq!(
            action_for_key(&app, press(KeyCode::Char('q'))),
            Some(Action::Quit)
        );
        assert_eq!(
            action_for_key(&app, press(KeyCode::Char('1'))),
            Some(Action::SetMode(Mode::Simplify))
        );
    }

    #[test]
    fn typing_sqrt_does_not_quit() {
        let mut app = App::new();
        app.focus = Focus::Expression;
        for c in "sqrt(x)".chars() {
            if let Some(action) = action_for_key(&app, press(KeyCode::Char(c))) {
                app.apply(action);
            }
        }
        assert!(!app.should_quit);
        assert_eq!(app.field(Focus::Expression), "sqrt(x)");
    }

    #[test]
    fn ctrl_q_quits_even_while_editing() {
        let mut app = App::new();
        app.focus = Focus::Expression;
        let action = action_for_key(&app, ctrl(KeyCode::Char('q'))).unwrap();
        app.apply(action);
        assert!(app.should_quit);
    }

    #[test]
    fn focus_order_follows_the_mode() {
        let mut app = App::new();
        assert_eq!(app.focus_order(), vec![Focus::Expression, Focus::Trace]);

        app.apply(Action::SetMode(Mode::Differentiate));
        assert_eq!(
            app.focus_order(),
            vec![Focus::Expression, Focus::Variable, Focus::Trace]
        );

        app.apply(Action::SetMode(Mode::VerifyCandidate));
        assert_eq!(
            app.focus_order(),
            vec![
                Focus::Expression,
                Focus::Variable,
                Focus::Candidate,
                Focus::Trace
            ]
        );
    }

    #[test]
    fn tab_cycles_focus_and_wraps() {
        let mut app = App::new();
        app.apply(Action::SetMode(Mode::Differentiate));

        assert_eq!(app.focus, Focus::Expression);
        app.apply(Action::FocusNext);
        assert_eq!(app.focus, Focus::Variable);
        app.apply(Action::FocusNext);
        assert_eq!(app.focus, Focus::Trace);
        app.apply(Action::FocusNext);
        assert_eq!(app.focus, Focus::Expression);

        app.apply(Action::FocusPrev);
        assert_eq!(app.focus, Focus::Trace);
    }

    #[test]
    fn switching_mode_drops_a_result_from_the_previous_operation() {
        let mut app = App::new();
        app.job = Job::Complete(Box::new(result_with_steps(2)));

        app.apply(Action::SetMode(Mode::Differentiate));
        assert!(matches!(app.job, Job::Idle));
    }

    #[test]
    fn focus_moves_off_a_field_the_new_mode_does_not_use() {
        let mut app = App::new();
        app.apply(Action::SetMode(Mode::VerifyCandidate));
        app.focus = Focus::Candidate;

        app.apply(Action::SetMode(Mode::Simplify));
        assert_eq!(app.focus, Focus::Expression);
    }

    #[test]
    fn submitting_while_running_does_not_queue_a_second_job() {
        let mut app = App::new();
        let first = app.apply(Action::Submit);
        assert!(first.is_some());
        assert!(app.is_running());

        let second = app.apply(Action::Submit);
        assert!(
            second.is_none(),
            "a second submit must not start while one is in flight"
        );
    }

    #[test]
    fn each_submission_gets_a_new_id() {
        let mut app = App::new();
        let first = app.apply(Action::Submit).unwrap();
        app.accept(SolveResponse {
            id: first.id,
            elapsed: Duration::from_millis(1),
            outcome: Ok(result_with_steps(1)),
        });
        let second = app.apply(Action::Submit).unwrap();
        assert_ne!(first.id, second.id);
    }

    #[test]
    fn a_stale_response_is_ignored() {
        let mut app = App::new();
        let request = app.apply(Action::Submit).unwrap();

        app.accept(SolveResponse {
            id: request.id + 999,
            elapsed: Duration::from_millis(1),
            outcome: Ok(result_with_steps(3)),
        });
        assert!(
            app.is_running(),
            "a reply for another request must not complete this one"
        );

        app.accept(SolveResponse {
            id: request.id,
            elapsed: Duration::from_millis(2),
            outcome: Ok(result_with_steps(3)),
        });
        assert_eq!(app.step_count(), 3);
    }

    #[test]
    fn trace_selection_stays_in_range() {
        let mut app = App::new();
        app.job = Job::Complete(Box::new(result_with_steps(3)));

        app.apply(Action::TraceUp);
        assert_eq!(app.selected_step, 0, "cannot select above the first step");

        for _ in 0..10 {
            app.apply(Action::TraceDown);
        }
        assert_eq!(app.selected_step, 2, "cannot select past the last step");

        app.apply(Action::TraceHome);
        assert_eq!(app.selected_step, 0);
        app.apply(Action::TraceEnd);
        assert_eq!(app.selected_step, 2);

        app.apply(Action::TracePageUp);
        assert_eq!(app.selected_step, 0);
    }

    #[test]
    fn trace_navigation_is_safe_with_no_steps() {
        let mut app = App::new();
        app.apply(Action::TraceDown);
        app.apply(Action::TraceEnd);
        assert_eq!(app.selected_step, 0);
    }

    #[test]
    fn history_keeps_the_most_recent_entries_only() {
        let mut app = App::new();
        for _ in 0..(HISTORY_LIMIT + 5) {
            let request = app.apply(Action::Submit).unwrap();
            app.accept(SolveResponse {
                id: request.id,
                elapsed: Duration::from_millis(1),
                outcome: Ok(result_with_steps(1)),
            });
        }
        assert_eq!(app.history.len(), HISTORY_LIMIT);
    }

    #[test]
    fn a_failed_request_is_not_remembered_as_a_result() {
        let mut app = App::new();
        let request = app.apply(Action::Submit).unwrap();
        app.accept(SolveResponse {
            id: request.id,
            elapsed: Duration::from_millis(1),
            outcome: Err(UiError {
                field: ErrorField::Expression,
                message: "Unexpected end of input".to_string(),
            }),
        });

        assert!(matches!(app.job, Job::Failed(_)));
        assert_eq!(app.badge(), Some(StatusBadge::InputError));
        assert_eq!(app.error_field(), Some(ErrorField::Expression));
        assert!(
            app.history.is_empty(),
            "a rejected input is not a completed request"
        );
    }

    #[test]
    fn history_records_the_inputs_the_request_was_sent_with() {
        // A job runs on another thread and the user can keep typing while it does. History is
        // written when the reply lands, so reading the editors at that moment would file the
        // newly typed text against the previous run's result.
        let mut app = App::new();
        app.expression = single_line_editor("2 + 3");

        let request = app.apply(Action::Submit).unwrap();
        assert_eq!(request.expression, "2 + 3");

        // The user types something else while the solver is busy.
        app.expression = single_line_editor("completely different");

        app.accept(SolveResponse {
            id: request.id,
            elapsed: Duration::from_millis(3),
            outcome: Ok(result_with_steps(1)),
        });

        assert_eq!(
            app.history[0].expression, "2 + 3",
            "history must record the submitted input, not whatever is in the editor now"
        );
    }

    #[test]
    fn history_records_the_mode_the_request_was_sent_with() {
        let mut app = App::new();
        app.expression = single_line_editor("x^2");
        app.apply(Action::SetMode(Mode::Differentiate));

        let request = app.apply(Action::Submit).unwrap();
        assert_eq!(request.mode, Mode::Differentiate);

        app.accept(SolveResponse {
            id: request.id,
            elapsed: Duration::from_millis(3),
            outcome: Ok(result_with_steps(1)),
        });

        assert_eq!(app.history[0].mode, Mode::Differentiate);
    }

    #[test]
    fn recalling_history_restores_the_form_and_the_trace() {
        let mut app = App::new();
        app.expression = single_line_editor("x + 0");
        let request = app.apply(Action::Submit).unwrap();
        app.accept(SolveResponse {
            id: request.id,
            elapsed: Duration::from_millis(4),
            outcome: Ok(result_with_steps(2)),
        });

        app.expression = single_line_editor("something else");
        app.job = Job::Idle;

        app.apply(Action::ToggleHistory);
        app.apply(Action::RecallHistory);

        assert_eq!(app.field(Focus::Expression), "x + 0");
        assert_eq!(app.step_count(), 2);
        assert_eq!(app.overlay, None);
    }

    #[test]
    fn overlays_toggle_and_escape_closes_them() {
        let mut app = App::new();
        app.apply(Action::ToggleHelp);
        assert_eq!(app.overlay, Some(Overlay::Help));
        app.apply(Action::ToggleHelp);
        assert_eq!(app.overlay, None);

        app.apply(Action::ToggleHistory);
        assert_eq!(app.overlay, Some(Overlay::History));
        app.apply(Action::Escape);
        assert_eq!(app.overlay, None);
    }

    #[test]
    fn escape_from_the_trace_returns_to_the_form() {
        let mut app = App::new();
        app.focus = Focus::Trace;
        app.apply(Action::Escape);
        assert_eq!(app.focus, Focus::Expression);
    }

    #[test]
    fn clear_empties_the_form_and_the_result() {
        let mut app = App::new();
        app.expression = single_line_editor("x + 0");
        app.job = Job::Complete(Box::new(result_with_steps(1)));

        app.apply(Action::Clear);
        assert_eq!(app.field(Focus::Expression), "");
        assert!(matches!(app.job, Job::Idle));
        assert_eq!(app.focus, Focus::Expression);
    }

    #[test]
    fn clear_keeps_the_variable_so_the_next_run_does_not_need_retyping() {
        let mut app = App::new();
        app.apply(Action::SetMode(Mode::Differentiate));
        app.apply(Action::Clear);
        assert_eq!(app.field(Focus::Variable), "x");
    }

    #[test]
    fn enter_inside_an_editor_does_not_create_a_second_line() {
        let mut app = App::new();
        app.focus = Focus::Expression;
        app.apply(Action::Edit(press(KeyCode::Char('x'))));
        app.apply(Action::Edit(press(KeyCode::Enter)));
        app.apply(Action::Edit(press(KeyCode::Char('y'))));
        assert_eq!(app.expression.lines().len(), 1);
        assert_eq!(app.field(Focus::Expression), "xy");
    }

    #[test]
    fn history_cursor_stays_in_range() {
        let mut app = App::new();
        app.move_history_cursor(1);
        assert_eq!(app.history_cursor, 0);

        for _ in 0..3 {
            let request = app.apply(Action::Submit).unwrap();
            app.accept(SolveResponse {
                id: request.id,
                elapsed: Duration::from_millis(1),
                outcome: Ok(result_with_steps(1)),
            });
        }
        app.move_history_cursor(10);
        assert_eq!(app.history_cursor, 2);
        app.move_history_cursor(-10);
        assert_eq!(app.history_cursor, 0);
    }

    #[test]
    fn help_opens_from_an_editor_because_question_mark_is_not_expression_syntax() {
        let mut app = App::new();
        app.focus = Focus::Expression;
        assert_eq!(
            action_for_key(&app, press(KeyCode::Char('?'))),
            Some(Action::ToggleHelp)
        );
    }
}
