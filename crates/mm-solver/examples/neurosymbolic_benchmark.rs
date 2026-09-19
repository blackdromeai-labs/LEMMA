//! Reproducible trained-versus-uniform benchmark for LEMMA's neural MCTS.
//!
//! Commands:
//!   neurosymbolic_benchmark generate [corpus.jsonl]
//!   neurosymbolic_benchmark validate [corpus.jsonl]
//!   neurosymbolic_benchmark run [corpus.jsonl] [model.safetensors] [simulations]

use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use candle_core::Device;
use mm_brain::PolicyNetwork;
use mm_core::format_expr;
use mm_core::parse::Parser;
use mm_core::sampling::seed_sampling_rng;
use mm_core::{Expr, Symbol, SymbolTable};
use mm_rules::{standard_rules, ActionVocabulary, RuleContext, RuleSet};
use mm_search::{MCTSConfig, NeuralMCTS};
use mm_verifier::Verifier;
use serde::{Deserialize, Serialize};

const DEFAULT_CORPUS: &str = "experiments/corpus/problems.jsonl";
const DEFAULT_MODEL: &str = "experiments/models/policy.safetensors";
const GENERATOR_SEED: u64 = 0x4c45_4d4d_415f_0001;
const DEPTHS: [usize; 5] = [2, 3, 4, 6, 8];
const PER_CELL: usize = 40;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProblemRecord {
    id: String,
    family: String,
    split: String,
    partition: String,
    construction_depth: usize,
    generator_seed: u64,
    input: String,
    expected: String,
    reference_rules: Vec<String>,
}

#[derive(Debug, Default)]
struct ValidationSummary {
    problems: usize,
    reference_steps: usize,
    verified_steps: usize,
    branch_sum: usize,
    branch_min: usize,
    branch_max: usize,
}

#[derive(Debug, Default, Serialize)]
struct ArmSummary {
    arm: String,
    solved: usize,
    total: usize,
    elapsed_ms: u128,
    total_steps: usize,
    by_split_depth: BTreeMap<String, CellSummary>,
}

#[derive(Debug, Default, Serialize)]
struct CellSummary {
    solved: usize,
    total: usize,
}

fn bx(expr: Expr) -> Box<Expr> {
    Box::new(expr)
}

fn add(a: Expr, b: Expr) -> Expr {
    Expr::Add(bx(a), bx(b))
}

fn sub(a: Expr, b: Expr) -> Expr {
    Expr::Sub(bx(a), bx(b))
}

fn mul(a: Expr, b: Expr) -> Expr {
    Expr::Mul(bx(a), bx(b))
}

fn div(a: Expr, b: Expr) -> Expr {
    Expr::Div(bx(a), bx(b))
}

fn pow(a: Expr, b: Expr) -> Expr {
    Expr::Pow(bx(a), bx(b))
}

fn wrap_id(mut expr: Expr, count: usize, salt: usize) -> (Expr, Vec<String>) {
    let mut built = Vec::new();
    for i in 0..count {
        if (i + salt) % 2 == 0 {
            expr = add(expr, Expr::int(0));
            built.push("algebra::identity_add_zero".to_string());
        } else {
            expr = mul(Expr::int(1), expr);
            built.push("algebra::identity_mul_one".to_string());
        }
    }
    built.reverse();
    (expr, built)
}

fn wrap_ood(mut expr: Expr, count: usize, salt: usize) -> (Expr, Vec<String>) {
    let mut built = Vec::new();
    for i in 0..count {
        match (i + salt) % 2 {
            0 => {
                expr = pow(expr, Expr::int(1));
                built.push("algebra::power_of_one".to_string());
            }
            _ => {
                expr = Expr::Ln(bx(Expr::Exp(bx(expr))));
                built.push("algebra::ln_exp".to_string());
            }
        }
    }
    built.reverse();
    (expr, built)
}

fn record(
    symbols: &SymbolTable,
    id: String,
    family: &str,
    split: &str,
    depth: usize,
    input: Expr,
    expected: Expr,
    rules: Vec<String>,
) -> ProblemRecord {
    assert_eq!(rules.len(), depth);
    ProblemRecord {
        id,
        family: family.to_string(),
        split: split.to_string(),
        partition: "locked-test".to_string(),
        construction_depth: depth,
        generator_seed: GENERATOR_SEED,
        input: format_expr(&input, symbols),
        expected: format_expr(&expected, symbols),
        reference_rules: rules,
    }
}

fn id_problem(
    symbols: &SymbolTable,
    x: Symbol,
    y: Symbol,
    depth: usize,
    index: usize,
) -> ProblemRecord {
    let a = 2 + (index % 7) as i64;
    let b = 3 + ((index * 3) % 11) as i64;
    match index % 5 {
        0 => {
            let expected = add(mul(Expr::int(a), Expr::Var(x)), Expr::Var(y));
            let (input, rules) = wrap_id(expected.clone(), depth, index);
            record(
                symbols,
                format!("id-identity-d{depth}-{index:03}"),
                "identity-labyrinth",
                "ID",
                depth,
                input,
                expected,
                rules,
            )
        }
        1 => {
            let core = add(
                mul(Expr::int(a), Expr::Var(x)),
                mul(Expr::int(b), Expr::Var(x)),
            );
            let expected = mul(Expr::int(a + b), Expr::Var(x));
            let (input, mut rules) = wrap_id(core, depth - 1, index);
            rules.push("algebra::collect_like_terms".to_string());
            record(
                symbols,
                format!("id-collect-d{depth}-{index:03}"),
                "collect-after-noise",
                "ID",
                depth,
                input,
                expected,
                rules,
            )
        }
        2 => {
            let core = mul(Expr::int(a), add(Expr::Var(x), Expr::int(b)));
            let expected = add(
                mul(Expr::int(a), Expr::Var(x)),
                mul(Expr::int(a), Expr::int(b)),
            );
            let (input, mut rules) = wrap_id(core, depth - 1, index);
            rules.push("algebra::distribute".to_string());
            record(
                symbols,
                format!("id-distribute-d{depth}-{index:03}"),
                "distribute-after-noise",
                "ID",
                depth,
                input,
                expected,
                rules,
            )
        }
        3 => {
            let core = sub(
                pow(Expr::Var(x), Expr::int(2)),
                pow(Expr::int(b), Expr::int(2)),
            );
            let expected = mul(
                add(Expr::Var(x), Expr::int(b)),
                sub(Expr::Var(x), Expr::int(b)),
            );
            let (input, mut rules) = wrap_id(core, depth - 1, index);
            rules.push("algebra::difference_of_squares".to_string());
            record(
                symbols,
                format!("id-diffsq-d{depth}-{index:03}"),
                "difference-square-after-noise",
                "ID",
                depth,
                input,
                expected,
                rules,
            )
        }
        _ => equation_problem(symbols, x, depth, index, a, b),
    }
}

fn equation_problem(
    symbols: &SymbolTable,
    x: Symbol,
    depth: usize,
    index: usize,
    a: i64,
    b: i64,
) -> ProblemRecord {
    let mut lhs = Expr::Var(x);
    let mut operations = Vec::new();
    for i in 0..depth {
        let c = 2 + ((index + i) % 5) as i64;
        match (i + index) % 4 {
            0 => {
                lhs = add(lhs, Expr::int(c));
                operations.push(("algebra", c));
            }
            1 => {
                lhs = sub(lhs, Expr::int(c));
                operations.push(("subtract", c));
            }
            2 => {
                lhs = mul(Expr::int(c), lhs);
                operations.push(("multiply", c));
            }
            _ => {
                lhs = div(lhs, Expr::int(c));
                operations.push(("divide", c));
            }
        }
    }
    let mut rhs = Expr::int(a * b + 17);
    let mut rules = Vec::new();
    for (op, c) in operations.iter().rev() {
        match *op {
            "algebra" => {
                rhs = sub(rhs, Expr::int(*c));
                rules.push("equations::cancel_addition".to_string());
            }
            "subtract" => {
                rhs = add(rhs, Expr::int(*c));
                rules.push("equations::cancel_subtraction".to_string());
            }
            "multiply" => {
                rhs = div(rhs, Expr::int(*c));
                rules.push("equations::cancel_multiplication".to_string());
            }
            _ => {
                rhs = mul(rhs, Expr::int(*c));
                rules.push("equations::cancel_division".to_string());
            }
        }
    }
    let input = Expr::Equation {
        lhs: bx(lhs),
        rhs: bx(Expr::int(a * b + 17)),
    };
    let expected = Expr::Equation {
        lhs: bx(Expr::Var(x)),
        rhs: bx(rhs),
    };
    record(
        symbols,
        format!("id-equation-d{depth}-{index:03}"),
        "nested-equation-cancellation",
        "ID",
        depth,
        input,
        expected,
        rules,
    )
}

fn ood_problem(symbols: &SymbolTable, x: Symbol, depth: usize, index: usize) -> ProblemRecord {
    let c = 2 + (index % 13) as i64;
    let base = add(Expr::Var(x), Expr::int(c));
    match index % 5 {
        0 => {
            let mut input = base.clone();
            for _ in 0..depth {
                input = pow(input, Expr::int(1));
            }
            record(
                symbols,
                format!("ood-power1-d{depth}-{index:03}"),
                "nested-power-of-one",
                "OOD",
                depth,
                input,
                base,
                vec!["algebra::power_of_one".to_string(); depth],
            )
        }
        1 => {
            let mut input = base.clone();
            for _ in 0..depth {
                input = Expr::Neg(bx(Expr::Neg(bx(input))));
            }
            record(
                symbols,
                format!("ood-neg-d{depth}-{index:03}"),
                "nested-double-negative",
                "OOD",
                depth,
                input,
                base,
                vec!["algebra::double_negative".to_string(); depth],
            )
        }
        2 => {
            let mut input = base.clone();
            for _ in 0..depth {
                input = Expr::Ln(bx(Expr::Exp(bx(input))));
            }
            record(
                symbols,
                format!("ood-lnexp-d{depth}-{index:03}"),
                "nested-log-exp",
                "OOD",
                depth,
                input,
                base,
                vec!["algebra::ln_exp".to_string(); depth],
            )
        }
        3 => {
            let (input, rules) = wrap_ood(base.clone(), depth, index);
            record(
                symbols,
                format!("ood-mixed-d{depth}-{index:03}"),
                "mixed-unseen-wrappers",
                "OOD",
                depth,
                input,
                base,
                rules,
            )
        }
        _ => {
            let core = add(
                pow(Expr::Sin(bx(base.clone())), Expr::int(2)),
                pow(Expr::Cos(bx(base)), Expr::int(2)),
            );
            let (input, mut rules) = wrap_ood(core, depth - 1, index);
            rules.push("trig::pythagorean_identity".to_string());
            record(
                symbols,
                format!("ood-trig-d{depth}-{index:03}"),
                "pythagorean-after-unseen-noise",
                "OOD",
                depth,
                input,
                Expr::int(1),
                rules,
            )
        }
    }
}

fn generate() -> Vec<ProblemRecord> {
    let mut symbols = SymbolTable::new();
    let x = symbols.intern("x");
    let y = symbols.intern("y");
    let mut problems = Vec::with_capacity(DEPTHS.len() * PER_CELL * 2);
    for depth in DEPTHS {
        for index in 0..PER_CELL {
            problems.push(id_problem(&symbols, x, y, depth, index));
            problems.push(ood_problem(&symbols, x, depth, index));
        }
    }
    problems.sort_by(|a, b| a.id.cmp(&b.id));
    problems
}

fn write_corpus(path: &Path, problems: &[ProblemRecord]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let mut out = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);
    for problem in problems {
        serde_json::to_writer(&mut out, problem).map_err(|e| e.to_string())?;
        writeln!(out).map_err(|e| e.to_string())?;
    }
    out.flush().map_err(|e| e.to_string())
}

fn read_corpus(path: &Path) -> Result<Vec<ProblemRecord>, String> {
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    BufReader::new(file)
        .lines()
        .enumerate()
        .map(|(i, line)| {
            let line = line.map_err(|e| e.to_string())?;
            serde_json::from_str(&line)
                .map_err(|e| format!("{} line {}: {e}", path.display(), i + 1))
        })
        .collect()
}

fn find_rule<'a>(rules: &'a RuleSet, key: &str) -> Option<&'a mm_rules::Rule> {
    rules
        .keys()
        .iter()
        .zip(rules.all())
        .find(|(k, _)| k.to_string() == key)
        .map(|(_, r)| r)
}

fn verified_branch_count(rules: &RuleSet, verifier: &Verifier, state: &Expr) -> usize {
    let ctx = RuleContext::default();
    rules
        .all()
        .iter()
        .map(|rule| {
            if !rule.can_apply(state, &ctx) {
                return 0;
            }
            rule.apply(state, &ctx)
                .into_iter()
                .filter(|app| {
                    app.result != *state
                        && verifier
                            .verify_step(state, &app.result, rule, &ctx)
                            .is_valid()
                })
                .count()
        })
        .sum()
}

fn validate(problems: &[ProblemRecord]) -> Result<ValidationSummary, String> {
    let rules = standard_rules();
    let verifier = Verifier::new();
    let ctx = RuleContext::default();
    let trained: HashSet<&str> = [
        "algebra::const_fold",
        "algebra::identity_add_zero",
        "algebra::identity_mul_one",
        "algebra::zero_mul",
        "algebra::collect_like_terms",
        "algebra::distribute",
        "algebra::factor_common",
        "algebra::difference_of_squares",
        "calculus::power_rule",
        "calculus::constant_rule",
        "calculus::sum_rule",
        "calculus::product_rule",
        "calculus::quotient_rule",
        "calculus::sin_chain_rule",
        "calculus::cos_chain_rule",
        "calculus::exp_derivative",
        "calculus::ln_derivative",
        "equations::isolate_variable",
        "equations::cancel_addition",
        "equations::cancel_subtraction",
        "equations::cancel_multiplication",
        "equations::cancel_division",
        "equations::linear_solve",
        "equations::quadratic_formula",
    ]
    .into_iter()
    .collect();
    let mut ids = HashSet::new();
    let mut exact_pairs = HashSet::new();
    let mut cells: BTreeMap<(String, usize), usize> = BTreeMap::new();
    let mut summary = ValidationSummary {
        branch_min: usize::MAX,
        ..Default::default()
    };

    for problem in problems {
        if !ids.insert(problem.id.clone()) {
            return Err(format!("duplicate id: {}", problem.id));
        }
        if !exact_pairs.insert((problem.input.clone(), problem.expected.clone())) {
            return Err(format!("duplicate pair: {}", problem.id));
        }
        if problem.partition != "locked-test" {
            return Err(format!("{} is not locked-test", problem.id));
        }
        if problem.generator_seed != GENERATOR_SEED {
            return Err(format!("{} has wrong generator seed", problem.id));
        }
        if problem.reference_rules.len() != problem.construction_depth {
            return Err(format!("{} depth/path mismatch", problem.id));
        }
        *cells
            .entry((problem.split.clone(), problem.construction_depth))
            .or_default() += 1;

        let mut symbols = SymbolTable::new();
        let mut parser = Parser::new(&mut symbols);
        let mut state = parser
            .parse(&problem.input)
            .map_err(|e| format!("{} input: {e}", problem.id))?;
        let expected = parser
            .parse(&problem.expected)
            .map_err(|e| format!("{} expected: {e}", problem.id))?;
        if format_expr(&state, &symbols) != problem.input
            || format_expr(&expected, &symbols) != problem.expected
        {
            return Err(format!("{} does not round-trip exactly", problem.id));
        }

        for key in &problem.reference_rules {
            let is_trained = trained.contains(key.as_str());
            if (problem.split == "ID") != is_trained {
                return Err(format!("{} split violation at {key}", problem.id));
            }
            let branch = verified_branch_count(&rules, &verifier, &state);
            summary.branch_sum += branch;
            summary.branch_min = summary.branch_min.min(branch);
            summary.branch_max = summary.branch_max.max(branch);
            let rule = find_rule(&rules, key)
                .ok_or_else(|| format!("{} missing rule {key}", problem.id))?;
            if !rule.can_apply(&state, &ctx) {
                return Err(format!(
                    "{} rule {key} cannot apply to {}",
                    problem.id,
                    format_expr(&state, &symbols)
                ));
            }
            let applications = rule.apply(&state, &ctx);
            if applications.len() != 1 {
                return Err(format!(
                    "{} rule {key} produced {} applications",
                    problem.id,
                    applications.len()
                ));
            }
            let next = applications[0].result.clone();
            if !verifier.verify_step(&state, &next, rule, &ctx).is_valid() {
                return Err(format!("{} verifier rejected {key}", problem.id));
            }
            state = next;
            summary.verified_steps += 1;
        }
        if state != expected {
            return Err(format!(
                "{} reference path ended at {}, expected {}",
                problem.id,
                format_expr(&state, &symbols),
                problem.expected
            ));
        }
        summary.problems += 1;
        summary.reference_steps += problem.reference_rules.len();
    }
    for split in ["ID", "OOD"] {
        for depth in DEPTHS {
            if cells.get(&(split.to_string(), depth)).copied() != Some(PER_CELL) {
                return Err(format!(
                    "cell {split}/depth-{depth} does not contain {PER_CELL} problems"
                ));
            }
        }
    }
    Ok(summary)
}

fn make_solver(model: Option<&Path>, simulations: usize) -> Result<NeuralMCTS, String> {
    let rules = standard_rules();
    let verifier = Verifier::new();
    let config = MCTSConfig {
        simulations,
        max_depth: 12,
        ..Default::default()
    };
    let solver = NeuralMCTS::with_config(rules, verifier, config);
    if let Some(path) = model {
        #[cfg(feature = "cuda")]
        let device = Device::new_cuda(0).map_err(|e| e.to_string())?;
        #[cfg(not(feature = "cuda"))]
        let device = Device::Cpu;
        let policy = PolicyNetwork::load(path, ActionVocabulary::standard(), device)
            .map_err(|e| e.to_string())?;
        solver.with_policy(policy).map_err(|e| e.to_string())
    } else {
        Ok(solver)
    }
}

fn run_arm(
    problems: &[ProblemRecord],
    model: Option<&Path>,
    simulations: usize,
    arm: &str,
) -> Result<ArmSummary, String> {
    let solver = make_solver(model, simulations)?;
    let start = Instant::now();
    let mut summary = ArmSummary {
        arm: arm.to_string(),
        total: problems.len(),
        ..Default::default()
    };
    for (index, problem) in problems.iter().enumerate() {
        let mut symbols = SymbolTable::new();
        let mut parser = Parser::new(&mut symbols);
        let input = parser.parse(&problem.input).map_err(|e| e.to_string())?;
        let expected = parser.parse(&problem.expected).map_err(|e| e.to_string())?;
        seed_sampling_rng(GENERATOR_SEED ^ index as u64);
        let solution = solver.search(input, |candidate| *candidate == expected);
        let solved = solution
            .as_ref()
            .is_some_and(|s| s.result == expected && s.status.replays());
        let cell = summary
            .by_split_depth
            .entry(format!("{}-d{}", problem.split, problem.construction_depth))
            .or_default();
        cell.total += 1;
        if solved {
            summary.solved += 1;
            cell.solved += 1;
        }
        if let Some(solution) = solution {
            summary.total_steps += solution.steps.len();
        }
        if (index + 1) % 25 == 0 || index + 1 == problems.len() {
            println!(
                "{arm}: {}/{} complete, solved {}",
                index + 1,
                problems.len(),
                summary.solved
            );
        }
    }
    summary.elapsed_ms = start.elapsed().as_millis();
    Ok(summary)
}

fn write_validation_report(path: &Path, summary: &ValidationSummary) -> Result<(), String> {
    let mean = summary.branch_sum as f64 / summary.reference_steps.max(1) as f64;
    let text = format!(
        "# Corpus validation report\n\n- Problems: {}\n- Reference steps: {}\n- Verifier-accepted reference steps: {}\n- Cells: 2 splits x 5 depths x 40 problems\n- Branching over reference states: min {}, mean {:.2}, max {}\n- Parser round-trip failures: 0\n- Duplicate IDs or exact pairs: 0\n- Split/rule-label violations: 0\n- Reference paths failing to reach expected output: 0\n\nThe corpus is a controlled synthetic search benchmark, not an external benchmark. `construction_depth` is the length of the recorded valid construction path; it is not claimed to be globally minimal.\n",
        summary.problems, summary.reference_steps, summary.verified_steps,
        summary.branch_min, mean, summary.branch_max,
    );
    fs::write(path, text).map_err(|e| e.to_string())
}

fn real_main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let command = args.get(1).map(String::as_str).unwrap_or("validate");
    let corpus = Path::new(args.get(2).map(String::as_str).unwrap_or(DEFAULT_CORPUS));
    match command {
        "generate" => {
            let problems = generate();
            write_corpus(corpus, &problems)?;
            let summary = validate(&problems)?;
            write_validation_report(Path::new("experiments/corpus/VALIDATION.md"), &summary)?;
            println!(
                "generated and validated {} problems at {}",
                problems.len(),
                corpus.display()
            );
        }
        "validate" => {
            let problems = read_corpus(corpus)?;
            let summary = validate(&problems)?;
            println!(
                "validated {} problems, {} reference steps, branching min/mean/max = {}/{:.2}/{}",
                summary.problems,
                summary.reference_steps,
                summary.branch_min,
                summary.branch_sum as f64 / summary.reference_steps.max(1) as f64,
                summary.branch_max
            );
        }
        "run" => {
            let model = Path::new(args.get(3).map(String::as_str).unwrap_or(DEFAULT_MODEL));
            let simulations = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(150usize);
            let problems = read_corpus(corpus)?;
            validate(&problems)?;
            let uniform = run_arm(&problems, None, simulations, "uniform")?;
            let trained = run_arm(&problems, Some(model), simulations, "trained")?;
            fs::create_dir_all("experiments/results").map_err(|e| e.to_string())?;
            let result = serde_json::json!({"simulations": simulations, "uniform": uniform, "trained": trained});
            fs::write(
                "experiments/results/trained_vs_uniform.json",
                serde_json::to_string_pretty(&result).unwrap(),
            )
            .map_err(|e| e.to_string())?;
            println!("{}", serde_json::to_string_pretty(&result).unwrap());
        }
        _ => {
            return Err(format!(
                "unknown command {command}; use generate, validate, or run"
            ))
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match real_main() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("FAIL: {error}");
            ExitCode::FAILURE
        }
    }
}
