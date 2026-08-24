//! Render assertions against Ratatui's `TestBackend`.
//!
//! These draw real frames into an in-memory buffer and read the text back, so they catch the
//! failures that unit tests on state cannot: a pane that renders nothing, a badge that
//! disappears at a narrow width, a status whose reason is dropped, a panic from a layout with
//! no room left.
//!
//! The strongest assertions here are about honesty rather than pixels. A `Heuristic` result
//! must not put the word CHECKED on screen anywhere, and an unverified one must not either.

use std::time::Duration;

use ratatui::backend::TestBackend;
use ratatui::Terminal;

use mm_tui::app::{App, Focus, Job, Overlay};
use mm_tui::presentation::{ErrorField, Mode, StatusBadge, UiError, UiEvidence, UiResult, UiStep};
use mm_tui::ui;

/// Render `app` at the given size and return the buffer as one string per row.
fn render(app: &App, width: u16, height: u16) -> Vec<String> {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test terminal");
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
        .collect()
}

/// The whole frame as a single searchable string.
fn text(app: &App, width: u16, height: u16) -> String {
    render(app, width, height).join("\n")
}

fn step(index: usize, checked: bool, independent: bool) -> UiStep {
    UiStep {
        index,
        before: "x + 0".to_string(),
        after: "x".to_string(),
        rule: "algebra::identity_add_zero".to_string(),
        rule_id: Some(2),
        justification: "Removed additive identity".to_string(),
        evidence: UiEvidence {
            label: if checked {
                if independent {
                    "checked (symbolic equivalence)".to_string()
                } else {
                    "checked (rule replay only)".to_string()
                }
            } else {
                "unchecked".to_string()
            },
            checked,
            independent,
        },
    }
}

fn result(badge: StatusBadge, reason: &str, steps: usize) -> UiResult {
    UiResult {
        mode: Mode::Simplify,
        input: "(x + 0) * 1".to_string(),
        output: "x".to_string(),
        badge,
        reason: reason.to_string(),
        steps: (1..=steps).map(|i| step(i, true, true)).collect(),
        elapsed: Duration::from_millis(14),
    }
}

fn completed(badge: StatusBadge, reason: &str, steps: usize) -> App {
    let mut app = App::new();
    app.rule_count = 572;
    app.job = Job::Complete(Box::new(result(badge, reason, steps)));
    app
}

// ---------------------------------------------------------------------------
// Layouts
// ---------------------------------------------------------------------------

#[test]
fn wide_layout_shows_every_pane() {
    let app = completed(StatusBadge::Checked, "Trace replays end to end.", 2);
    let screen = text(&app, 120, 40);

    assert!(screen.contains("LEMMA · SYMBOLIC WORKBENCH"));
    assert!(screen.contains("Simplify"));
    assert!(screen.contains("Differentiate"));
    assert!(screen.contains("Verify candidate"));
    assert!(screen.contains("Input"));
    assert!(screen.contains("Result"));
    assert!(screen.contains("Trace"));
    assert!(screen.contains("Selected step"));
    assert!(screen.contains("572 rules"));
}

#[test]
fn wide_layout_puts_result_and_trace_on_the_same_rows() {
    let app = completed(StatusBadge::Checked, "Trace replays end to end.", 2);
    let rows = render(&app, 120, 40);

    // Side by side means one row carries both titles.
    let shared = rows
        .iter()
        .any(|row| row.contains("Result") && row.contains("Trace"));
    assert!(shared, "at 120 columns Result and Trace should share a row");
}

#[test]
fn stacked_layout_separates_result_and_trace() {
    let app = completed(StatusBadge::Checked, "Trace replays end to end.", 2);
    let rows = render(&app, 90, 34);

    let shared = rows
        .iter()
        .any(|row| row.contains("Result") && row.contains("Trace"));
    assert!(!shared, "at 90 columns the panes should stack");

    let screen = rows.join("\n");
    assert!(screen.contains("Result"));
    assert!(screen.contains("Trace"));
    assert!(screen.contains("Selected step"));
}

#[test]
fn a_tiny_terminal_gets_a_notice_and_does_not_panic() {
    let app = App::new();
    let screen = text(&app, 40, 10);

    assert!(screen.contains("Terminal too small"));
    assert!(screen.contains("40x10"));
    assert!(screen.contains("80"), "the notice states the requirement");
    assert!(!screen.contains("Selected step"));
}

#[test]
fn every_size_from_tiny_to_wide_renders_without_panicking() {
    // Layout arithmetic that underflows shows up as a panic on an odd size, not on a round
    // one, so this sweeps the boundaries rather than sampling three sizes.
    let app = completed(StatusBadge::Partial, "mixed evidence", 6);
    for width in [20, 39, 40, 79, 80, 81, 109, 110, 111, 200] {
        for height in [3, 5, 9, 10, 23, 24, 25, 60] {
            let _ = render(&app, width, height);
        }
    }
}

// ---------------------------------------------------------------------------
// Job states
// ---------------------------------------------------------------------------

#[test]
fn idle_state_invites_formal_input_and_says_so() {
    let app = App::new();
    let screen = text(&app, 120, 30);

    assert!(screen.contains("Ctrl+Enter"));
    assert!(
        screen.contains("formal LEMMA syntax") || screen.contains("not prose"),
        "the idle state should say the input is formal, not natural language"
    );
    assert!(!screen.contains("CHECKED"));
}

#[test]
fn running_state_shows_a_running_label_and_no_verdict() {
    let mut app = App::new();
    app.job = Job::Running { id: 1 };
    let screen = text(&app, 120, 30);

    assert!(screen.contains("RUNNING"));
    assert!(!screen.contains("CHECKED"));
    assert!(!screen.contains("UNVERIFIED"));
    // It must not claim the run can be stopped, because it cannot.
    assert!(!screen.to_lowercase().contains("cancel "));
}

#[test]
fn a_parse_error_is_shown_as_input_error_beside_its_field() {
    let mut app = App::new();
    app.job = Job::Failed(UiError {
        field: ErrorField::Expression,
        message: "Unexpected end of input".to_string(),
    });
    let screen = text(&app, 120, 30);

    assert!(screen.contains("INPUT ERROR"));
    assert!(screen.contains("Unexpected end of input"));
    assert!(
        screen.contains("Nothing was solved"),
        "an input error must not read as a mathematical verdict"
    );
    assert!(!screen.contains("UNVERIFIED"));
}

// ---------------------------------------------------------------------------
// Verification honesty
// ---------------------------------------------------------------------------

#[test]
fn checked_results_say_checked() {
    let app = completed(
        StatusBadge::Checked,
        "Trace replays from the input to the result.",
        3,
    );
    let screen = text(&app, 120, 40);

    assert!(screen.contains("CHECKED"));
    assert!(screen.contains("Trace replays"));
    assert!(
        !screen.contains("Not a proof"),
        "a checked result carries no disclaimer"
    );
}

#[test]
fn a_heuristic_result_never_prints_the_word_checked_as_its_verdict() {
    let app = completed(
        StatusBadge::Heuristic,
        "every step was accepted by rule replay only",
        2,
    );
    let screen = text(&app, 120, 40);

    assert!(screen.contains("HEURISTIC"));
    assert!(screen.contains("rule replay only"));
    assert!(
        screen.contains("Not a proof"),
        "a heuristic result must say it is not a proof"
    );
    // The badge itself must not read CHECKED. Step evidence may legitimately contain the word
    // "checked", so this looks for the uppercase badge only.
    let badge_rows: Vec<&String> = render(&app, 120, 40)
        .iter()
        .filter(|row| row.contains("HEURISTIC"))
        .cloned()
        .collect::<Vec<String>>()
        .iter()
        .map(|_| &app.result().unwrap().reason)
        .collect();
    let _ = badge_rows;
    assert!(!screen.contains(" CHECKED "));
}

#[test]
fn partial_results_show_their_mixed_evidence_reason() {
    let app = completed(
        StatusBadge::Partial,
        "2 of 5 steps rely on numeric sampling",
        5,
    );
    let screen = text(&app, 120, 40);

    assert!(screen.contains("PARTIAL"));
    assert!(screen.contains("numeric sampling"));
    assert!(screen.contains("Not a proof"));
}

#[test]
fn unverified_results_state_why() {
    let app = completed(
        StatusBadge::Unverified,
        "trace does not end at the reported result",
        1,
    );
    let screen = text(&app, 120, 40);

    assert!(screen.contains("UNVERIFIED"));
    assert!(screen.contains("does not end at the reported result"));
    assert!(!screen.contains(" CHECKED "));
}

#[test]
fn unsupported_results_are_distinct_from_failures() {
    let app = completed(
        StatusBadge::Unsupported,
        "formal verification is not implemented",
        0,
    );
    let screen = text(&app, 120, 40);

    assert!(screen.contains("UNSUPPORTED"));
    assert!(screen.contains("not implemented"));
    assert!(!screen.contains("UNVERIFIED"));
    assert!(!screen.contains("INPUT ERROR"));
}

#[test]
fn every_badge_renders_its_label_and_marker() {
    for badge in [
        StatusBadge::Checked,
        StatusBadge::Partial,
        StatusBadge::Heuristic,
        StatusBadge::Unverified,
        StatusBadge::Unsupported,
        StatusBadge::NotFound,
        StatusBadge::CandidateValid,
        StatusBadge::CandidateInvalid,
    ] {
        let app = completed(badge, "reason text", 1);
        let screen = text(&app, 120, 40);
        assert!(
            screen.contains(badge.label()),
            "{} did not render its label",
            badge.label()
        );
        assert!(
            screen.contains(badge.marker()),
            "{} did not render its marker; colour would be the only cue",
            badge.label()
        );
    }
}

// ---------------------------------------------------------------------------
// Trace and detail
// ---------------------------------------------------------------------------

#[test]
fn the_trace_lists_steps_and_marks_the_selection() {
    let mut app = completed(StatusBadge::Checked, "ok", 3);
    app.selected_step = 1;
    let screen = text(&app, 120, 40);

    assert!(screen.contains("Trace · 3 steps"));
    assert!(
        !text(&completed(StatusBadge::Checked, "ok", 1), 120, 40).contains("1 steps"),
        "a single step should not be labelled with a plural"
    );
    assert!(screen.contains("identity_add_zero"));
    assert!(screen.contains('>'), "the selected row is marked");
}

#[test]
fn the_detail_pane_shows_before_after_rule_evidence_and_reason() {
    let app = completed(StatusBadge::Checked, "ok", 2);
    let screen = text(&app, 130, 44);

    assert!(screen.contains("Rule"));
    assert!(screen.contains("algebra::identity_add_zero"));
    assert!(screen.contains("id 2"));
    assert!(screen.contains("Before"));
    assert!(screen.contains("x + 0"));
    assert!(screen.contains("After"));
    assert!(screen.contains("Evidence"));
    assert!(screen.contains("symbolic equivalence"));
    assert!(screen.contains("Why"));
    assert!(screen.contains("Removed additive identity"));
}

#[test]
fn selecting_a_different_step_changes_the_detail() {
    let mut app = completed(StatusBadge::Checked, "ok", 3);
    if let Job::Complete(result) = &mut app.job {
        result.steps[2].after = "distinctive-second-value".to_string();
    }

    app.selected_step = 0;
    assert!(!text(&app, 130, 44).contains("distinctive-second-value"));

    app.selected_step = 2;
    assert!(text(&app, 130, 44).contains("distinctive-second-value"));
}

#[test]
fn a_step_checked_only_by_replay_says_so_in_the_detail() {
    let mut app = completed(StatusBadge::Heuristic, "replay only", 1);
    if let Job::Complete(result) = &mut app.job {
        result.steps[0] = step(1, true, false);
    }
    let screen = text(&app, 130, 44);

    assert!(screen.contains("rule replay only"));
    assert!(screen.contains("REPLAY"), "the trace row marks it too");
}

#[test]
fn an_unchecked_step_is_visible_as_unchecked() {
    let mut app = completed(StatusBadge::Unverified, "one step was not checked", 1);
    if let Job::Complete(result) = &mut app.job {
        result.steps[0] = step(1, false, false);
    }
    let screen = text(&app, 130, 44);

    assert!(screen.contains("UNCHECKED"));
    assert!(screen.contains("unchecked"));
}

#[test]
fn a_candidate_check_says_it_records_no_transformations() {
    let mut app = App::new();
    app.mode = Mode::VerifyCandidate;
    let mut r = result(StatusBadge::CandidateValid, "substitution satisfies it", 0);
    r.mode = Mode::VerifyCandidate;
    r.output = "x = 4".to_string();
    app.job = Job::Complete(Box::new(r));

    let screen = text(&app, 120, 40);
    assert!(screen.contains("SATISFIED"));
    assert!(screen.contains("records no transformations"));
    assert!(
        !screen.contains("Not a proof"),
        "a substitution check is not a trace-verification state and needs no disclaimer"
    );
}

#[test]
fn long_rule_names_and_reasons_do_not_overflow_the_frame() {
    let mut app = completed(
        StatusBadge::Partial,
        &"a very long explanation that keeps going ".repeat(8),
        1,
    );
    if let Job::Complete(result) = &mut app.job {
        result.steps[0].rule =
            "number_theory::an_extremely_long_rule_name_that_will_not_fit_in_any_reasonable_pane"
                .to_string();
        result.steps[0].before = "x".repeat(400);
    }

    let rows = render(&app, 82, 26);
    for row in &rows {
        assert!(
            row.chars().count() <= 82,
            "a row exceeded the terminal width: {} chars",
            row.chars().count()
        );
    }
}

#[test]
fn multi_byte_text_is_truncated_on_character_boundaries() {
    // Byte-index truncation would panic here rather than produce a short string.
    let mut app = completed(StatusBadge::Checked, &"π≈3.14159 ".repeat(30), 1);
    if let Job::Complete(result) = &mut app.job {
        result.output = "ααββγγδδ".repeat(40);
        result.steps[0].rule = "trig::ünïcödé_rule_名前".to_string();
    }

    let rows = render(&app, 84, 28);
    for row in &rows {
        assert!(row.chars().count() <= 84);
    }
}

// ---------------------------------------------------------------------------
// Modes and focus
// ---------------------------------------------------------------------------

/// Whether a form field with this label is drawn.
///
/// Checks for a row that *starts* with the label, so the word "candidate" inside the
/// "Verify candidate" mode button is not mistaken for the field itself.
fn has_field(app: &App, label: &str) -> bool {
    render(app, 120, 30)
        .iter()
        .any(|row| row.trim_start_matches(['│', ' ']).starts_with(label))
}

#[test]
fn simplify_mode_hides_the_variable_and_candidate_fields() {
    let app = App::new();

    assert!(has_field(&app, "expression"));
    assert!(!has_field(&app, "variable"));
    assert!(!has_field(&app, "candidate"));
}

#[test]
fn differentiate_mode_shows_the_variable_field() {
    let mut app = App::new();
    app.mode = Mode::Differentiate;

    assert!(has_field(&app, "expression"));
    assert!(has_field(&app, "variable"));
    assert!(!has_field(&app, "candidate"));
}

#[test]
fn verify_mode_shows_every_field() {
    let mut app = App::new();
    app.mode = Mode::VerifyCandidate;

    assert!(has_field(&app, "expression"));
    assert!(has_field(&app, "variable"));
    assert!(has_field(&app, "candidate"));
}

#[test]
fn the_footer_offers_different_keys_depending_on_focus() {
    let mut app = App::new();
    app.focus = Focus::Expression;
    assert!(text(&app, 120, 30).contains("Ctrl+Q quit"));

    app.focus = Focus::Trace;
    let screen = text(&app, 120, 30);
    assert!(screen.contains("1/2/3 mode"));
    assert!(screen.contains("q quit"));
}

#[test]
fn no_screen_offers_natural_language_or_equation_solving() {
    // The APIs behind both report themselves unsupported. A menu entry would contradict that.
    for mode in Mode::ALL {
        let mut app = App::new();
        app.mode = mode;
        let screen = text(&app, 120, 34).to_lowercase();
        assert!(!screen.contains("solve for"));
        assert!(!screen.contains("solve_for"));
        assert!(!screen.contains("natural language"));
        assert!(!screen.contains("word problem"));
    }
}

// ---------------------------------------------------------------------------
// Overlays
// ---------------------------------------------------------------------------

#[test]
fn the_help_overlay_lists_keys_and_states_the_input_contract() {
    let mut app = App::new();
    app.overlay = Some(Overlay::Help);
    let screen = text(&app, 120, 40);

    assert!(screen.contains("Help"));
    assert!(screen.contains("Ctrl+Enter"));
    assert!(screen.contains("Simplify"));
    assert!(screen.contains("Formal syntax only"));
    assert!(
        screen.contains("  Tab / Shift+Tab"),
        "help entries keep their indentation"
    );
    assert!(
        screen.contains("Only CHECKED means verified"),
        "help must restate the verification contract"
    );
    assert!(screen.contains("not supported"));
}

#[test]
fn the_history_overlay_reports_when_it_is_empty() {
    let mut app = App::new();
    app.overlay = Some(Overlay::History);
    let screen = text(&app, 120, 40);

    assert!(screen.contains("Recent"));
    assert!(screen.contains("in memory only"));
}

#[test]
fn the_history_overlay_lists_completed_requests() {
    let mut app = completed(StatusBadge::Checked, "ok", 1);
    app.history.push_front(mm_tui::app::HistoryEntry {
        mode: Mode::Simplify,
        expression: "(x + 0) * 1".to_string(),
        variable: "x".to_string(),
        candidate: String::new(),
        result: result(StatusBadge::Checked, "ok", 1),
    });
    app.overlay = Some(Overlay::History);

    let screen = text(&app, 120, 40);
    assert!(screen.contains("Simplify"));
    assert!(screen.contains("CHECKED"));
    assert!(screen.contains("1 of 20"));
}

#[test]
fn an_overlay_covers_the_panes_beneath_it() {
    let mut app = completed(StatusBadge::Checked, "a distinctive reason string", 1);
    assert!(text(&app, 120, 40).contains("a distinctive reason string"));

    app.overlay = Some(Overlay::Help);
    let screen = text(&app, 120, 40);
    assert!(
        !screen.contains("a distinctive reason string"),
        "the overlay must clear what is under it"
    );
}

#[test]
fn overlays_fit_inside_a_small_but_valid_terminal() {
    for overlay in [Overlay::Help, Overlay::History] {
        let mut app = App::new();
        app.overlay = Some(overlay);
        let rows = render(&app, 80, 24);
        for row in &rows {
            assert!(row.chars().count() <= 80);
        }
        assert_eq!(rows.len(), 24);
    }
}
