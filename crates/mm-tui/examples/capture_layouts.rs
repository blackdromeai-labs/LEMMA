//! Print the three layouts to stdout, using the same `TestBackend` the render tests use.
//!
//! This exists so a reviewer can see the layouts without a terminal, and so a change to them
//! can be inspected in a diff. It renders a fixed, hand-built result rather than running the
//! solver, so the output is stable.
//!
//! Usage: `cargo run -p mm-tui --example capture_layouts`

use std::time::Duration;

use ratatui::backend::TestBackend;
use ratatui::Terminal;

use mm_tui::app::{App, Job, Overlay};
use mm_tui::presentation::{Mode, StatusBadge, UiEvidence, UiResult, UiStep};
use mm_tui::ui;

fn step(
    index: usize,
    rule: &str,
    before: &str,
    after: &str,
    why: &str,
    independent: bool,
) -> UiStep {
    UiStep {
        index,
        before: before.to_string(),
        after: after.to_string(),
        rule: rule.to_string(),
        rule_id: Some(2),
        justification: why.to_string(),
        evidence: UiEvidence {
            label: if independent {
                "checked (symbolic equivalence)".to_string()
            } else {
                "checked (rule replay only)".to_string()
            },
            checked: true,
            independent,
        },
    }
}

fn sample() -> App {
    let mut app = App::new();
    app.rule_count = 572;
    app.expression = {
        let mut area = ratatui_textarea::TextArea::new(vec!["(x + 0) * 1".to_string()]);
        area.move_cursor(ratatui_textarea::CursorMove::End);
        area
    };
    app.job = Job::Complete(Box::new(UiResult {
        mode: Mode::Simplify,
        input: "(x + 0) * 1".to_string(),
        output: "x".to_string(),
        badge: StatusBadge::Checked,
        reason: "Trace replays from the input to the result and every step was independently \
                 checked."
            .to_string(),
        steps: vec![
            step(
                1,
                "algebra::identity_add_zero",
                "(x + 0) * 1",
                "x * 1",
                "Removed additive identity",
                true,
            ),
            step(
                2,
                "algebra::identity_mul_one",
                "x * 1",
                "x",
                "Removed multiplicative identity",
                true,
            ),
        ],
        elapsed: Duration::from_millis(14),
    }));
    app
}

fn capture(title: &str, app: &App, width: u16, height: u16) {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| ui::draw(frame, app))
        .expect("draw must not fail");

    let buffer = terminal.backend().buffer().clone();
    println!("\n### {title} ({width}x{height})\n");
    println!("+{}+", "-".repeat(width as usize));
    for y in 0..buffer.area.height {
        let row: String = (0..buffer.area.width)
            .map(|x| buffer[(x, y)].symbol())
            .collect();
        println!("|{row}|");
    }
    println!("+{}+", "-".repeat(width as usize));
}

fn main() {
    let app = sample();

    capture("Wide: result and trace side by side", &app, 118, 30);
    capture("Stacked: medium terminal", &app, 90, 30);
    capture("Too small: minimum-size notice", &App::new(), 60, 14);

    let mut with_help = sample();
    with_help.overlay = Some(Overlay::Help);
    capture("Help overlay", &with_help, 118, 30);

    let mut verify = App::new();
    verify.rule_count = 572;
    verify.mode = Mode::VerifyCandidate;
    verify.job = Job::Complete(Box::new(UiResult {
        mode: Mode::VerifyCandidate,
        input: "x + 3 = 7".to_string(),
        output: "x = 4".to_string(),
        badge: StatusBadge::CandidateValid,
        reason: "Substituting the candidate satisfies the equation, established by symbolic \
                 equivalence (confidence 1.000)."
            .to_string(),
        steps: Vec::new(),
        elapsed: Duration::from_millis(3),
    }));
    capture("Verify candidate", &verify, 118, 30);

    let mut heuristic = sample();
    if let Job::Complete(result) = &mut heuristic.job {
        result.badge = StatusBadge::Heuristic;
        result.reason = "every step was accepted by rule replay only".to_string();
        result.output = "3 * x^2".to_string();
        result.steps = vec![step(
            1,
            "calculus::power_rule",
            "diff(x^3, x)",
            "3 * x^2",
            "d/dx[x^n] = n*x^(n-1)",
            false,
        )];
    }
    capture("Heuristic result: evidence, not proof", &heuristic, 118, 30);
}
