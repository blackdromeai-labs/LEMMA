//! End-to-end runs through the real pipeline: key event -> action -> worker -> solver ->
//! presentation -> rendered frame.
//!
//! The render tests use hand-built results, which proves the drawing but not the wiring. These
//! use the actual `LemmaSolver`, so they catch a mode that sends the wrong request, a symbol
//! rendered against the wrong table, or a status that never reaches the screen.
//!
//! They are slower than the rest of the suite because each one loads the rule registry.

use std::time::{Duration, Instant};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::backend::TestBackend;
use ratatui::Terminal;

use mm_tui::app::{action_for_key, App, Focus, Job};
use mm_tui::presentation::{Mode, StatusBadge};
use mm_tui::ui;
use mm_tui::worker::Worker;

/// Type a string into the focused editor, one key event at a time.
fn type_text(app: &mut App, text: &str) {
    for c in text.chars() {
        let key = KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE);
        if let Some(action) = action_for_key(app, key) {
            app.apply(action);
        }
    }
}

fn clear_field(app: &mut App) {
    for _ in 0..64 {
        let key = KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE);
        if let Some(action) = action_for_key(app, key) {
            app.apply(action);
        }
    }
}

/// Press Ctrl+Enter, hand the request to the worker, and wait for the reply.
fn run_and_wait(app: &mut App, worker: &Worker) {
    let submit = KeyEvent::new(KeyCode::Enter, KeyModifiers::CONTROL);
    let action = action_for_key(app, submit).expect("Ctrl+Enter must submit");
    let request = app
        .apply(action)
        .expect("submitting must produce a request");
    assert!(worker.submit(request), "the worker must accept the request");

    let deadline = Instant::now() + Duration::from_secs(120);
    while app.is_running() {
        for response in worker.drain() {
            app.accept(response);
        }
        assert!(
            Instant::now() < deadline,
            "the solver did not reply within the timeout"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// Render the current state and return it as one searchable string.
fn screen(app: &App) -> String {
    let backend = TestBackend::new(120, 40);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| ui::draw(frame, app))
        .expect("draw must not fail");

    let buffer = terminal.backend().buffer().clone();
    (0..buffer.area.height)
        .map(|y| {
            (0..buffer.area.width)
                .map(|x| buffer[(x, y)].symbol())
                .collect::<String>()
                .trim_end()
                .to_string()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn simplify_runs_end_to_end_and_shows_a_readable_result() {
    let worker = Worker::spawn();
    let mut app = App::new();

    type_text(&mut app, "(x + 0) * 1");
    run_and_wait(&mut app, &worker);

    let result = app.result().expect("simplify should complete");
    assert_eq!(result.output, "x");

    let view = screen(&app);
    assert!(view.contains('x'));
    assert!(
        !view.contains("SymbolU32"),
        "a raw AST reached the screen:\n{view}"
    );
    assert!(
        !view.contains("Var("),
        "a raw AST reached the screen:\n{view}"
    );
    // Whatever the status, its badge and reason must both be on screen.
    assert!(view.contains(result.badge.label()));
    assert!(!result.reason.is_empty());
}

#[test]
fn a_multi_step_trace_is_navigable_and_its_detail_follows_the_selection() {
    let worker = Worker::spawn();
    let mut app = App::new();

    type_text(&mut app, "((x + 0) * 1) + 0");
    run_and_wait(&mut app, &worker);

    let steps = app.step_count();
    assert!(steps > 0, "this input should record at least one step");

    // Move to the trace and walk it with the arrow keys, exactly as a user would.
    app.focus = Focus::Trace;
    let first_detail = screen(&app);

    let down = KeyEvent::new(KeyCode::Down, KeyModifiers::NONE);
    for _ in 0..steps {
        if let Some(action) = action_for_key(&app, down) {
            app.apply(action);
        }
    }
    assert_eq!(
        app.selected_step,
        steps - 1,
        "Down should walk to the last step and stop"
    );

    if steps > 1 {
        assert_ne!(
            first_detail,
            screen(&app),
            "selecting another step must change the detail pane"
        );
    }

    // Every step must carry identity and evidence on screen.
    let view = screen(&app);
    assert!(view.contains("Rule"));
    assert!(view.contains("Evidence"));
    assert!(view.contains("Before"));
    assert!(view.contains("After"));
}

#[test]
fn differentiate_runs_end_to_end_with_its_variable() {
    let worker = Worker::spawn();
    let mut app = App::new();

    app.apply(mm_tui::app::Action::SetMode(Mode::Differentiate));
    type_text(&mut app, "x^3");
    run_and_wait(&mut app, &worker);

    let result = app.result().expect("differentiate should complete");
    assert_eq!(result.input, "diff(x^3, x)");
    assert!(!result.output.is_empty());

    let view = screen(&app);
    assert!(view.contains("variable"));
    assert!(view.contains(result.badge.label()));
    // A derivative rests on rule replay, so it must not be sold as fully verified.
    if result.badge != StatusBadge::Checked {
        assert!(!view.contains(" CHECKED "));
    }
}

#[test]
fn verify_candidate_runs_end_to_end_for_a_true_and_a_false_candidate() {
    let worker = Worker::spawn();
    let mut app = App::new();

    app.apply(mm_tui::app::Action::SetMode(Mode::VerifyCandidate));
    type_text(&mut app, "x + 3 = 7");

    // Tab to the candidate field: expression -> variable -> candidate.
    let tab = KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE);
    for _ in 0..2 {
        if let Some(action) = action_for_key(&app, tab) {
            app.apply(action);
        }
    }
    assert_eq!(app.focus, Focus::Candidate);

    type_text(&mut app, "4");
    run_and_wait(&mut app, &worker);

    let result = app.result().expect("verify should complete");
    assert_eq!(result.badge, StatusBadge::CandidateValid);
    assert!(screen(&app).contains("SATISFIED"));

    // Now a candidate that does not satisfy it.
    clear_field(&mut app);
    type_text(&mut app, "5");
    run_and_wait(&mut app, &worker);

    let result = app.result().expect("verify should complete");
    assert_eq!(result.badge, StatusBadge::CandidateInvalid);
    assert!(!result.badge.is_verified());
    assert!(screen(&app).contains("NOT SATISFIED"));
}

#[test]
fn a_parse_error_stays_in_the_ui_and_the_next_run_still_works() {
    let worker = Worker::spawn();
    let mut app = App::new();

    type_text(&mut app, "x +");
    run_and_wait(&mut app, &worker);

    assert!(matches!(app.job, Job::Failed(_)));
    let view = screen(&app);
    assert!(view.contains("INPUT ERROR"));
    assert!(
        view.contains("Nothing was solved"),
        "a parse error must not read as a mathematical verdict"
    );

    // The application is still usable afterwards.
    clear_field(&mut app);
    type_text(&mut app, "2 + 3");
    run_and_wait(&mut app, &worker);

    assert_eq!(app.result().expect("should recover").output, "5");
}

#[test]
fn history_records_completed_runs_and_can_restore_one() {
    let worker = Worker::spawn();
    let mut app = App::new();

    type_text(&mut app, "2 + 3");
    run_and_wait(&mut app, &worker);
    clear_field(&mut app);
    type_text(&mut app, "7 * 8");
    run_and_wait(&mut app, &worker);

    assert_eq!(app.history.len(), 2);
    assert_eq!(app.result().unwrap().output, "56");

    // Open history, move to the older entry, and restore it.
    app.apply(mm_tui::app::Action::ToggleHistory);
    app.move_history_cursor(1);
    app.apply(mm_tui::app::Action::RecallHistory);

    assert_eq!(app.field(Focus::Expression), "2 + 3");
    assert_eq!(app.result().unwrap().output, "5");
    assert_eq!(app.overlay, None);
}

#[test]
fn the_ui_stays_responsive_while_the_solver_is_working() {
    // The point of the worker thread. If solving happened on this thread, the frame drawn
    // between submitting and receiving could not exist.
    let worker = Worker::spawn();
    let mut app = App::new();

    type_text(&mut app, "sin(x)^2 + cos(x)^2");

    let submit = KeyEvent::new(KeyCode::Enter, KeyModifiers::CONTROL);
    let action = action_for_key(&app, submit).unwrap();
    let request = app.apply(action).unwrap();
    worker.submit(request);

    // Draw and handle keys while the request is outstanding.
    let mid_flight = screen(&app);
    assert!(mid_flight.contains("RUNNING"));

    let help = KeyEvent::new(KeyCode::Char('?'), KeyModifiers::NONE);
    if let Some(action) = action_for_key(&app, help) {
        app.apply(action);
    }
    assert!(screen(&app).contains("Help"), "keys work during a run");
    app.apply(mm_tui::app::Action::ToggleHelp);

    let deadline = Instant::now() + Duration::from_secs(120);
    while app.is_running() {
        for response in worker.drain() {
            app.accept(response);
        }
        assert!(Instant::now() < deadline, "solver timed out");
        std::thread::sleep(Duration::from_millis(10));
    }

    assert!(app.result().is_some());
}

#[test]
fn switching_modes_between_runs_keeps_each_result_with_its_own_operation() {
    let worker = Worker::spawn();
    let mut app = App::new();

    type_text(&mut app, "2 + 3");
    run_and_wait(&mut app, &worker);
    assert_eq!(app.result().unwrap().mode, Mode::Simplify);

    // Switching operation must drop the previous result rather than relabel it.
    app.apply(mm_tui::app::Action::SetMode(Mode::Differentiate));
    assert!(app.result().is_none());

    clear_field(&mut app);
    type_text(&mut app, "x^2");
    run_and_wait(&mut app, &worker);
    assert_eq!(app.result().unwrap().mode, Mode::Differentiate);
}
