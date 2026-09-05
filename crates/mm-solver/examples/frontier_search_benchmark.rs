//! LEMMA Frontier Search: a long-horizon adversarial benchmark for guided rewriting.
//!
//! Unlike the original corpus, this suite repeatedly places the reference move beside a
//! verified distractor and uses exact goals at construction depths 8--32. It is inspired by
//! frontier-benchmark principles (hard, unambiguous, outcome-verified), but it is a controlled
//! synthetic rewriting benchmark, not Humanity's Last Exam or FrontierMath.
//!
//! Usage:
//!   frontier_search_benchmark generate [corpus.jsonl]
//!   frontier_search_benchmark validate [corpus.jsonl]
//!   frontier_search_benchmark run [corpus.jsonl] [budget]

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
use mm_core::{Expr, SymbolTable};
use mm_rules::{standard_rules, ActionVocabulary, RuleContext, RuleSet};
use mm_search::{MCTSConfig, NeuralMCTS};
use mm_verifier::Verifier;
use serde::{Deserialize, Serialize};

const DEFAULT_CORPUS: &str = "experiments/frontier/corpus.jsonl";
const SOURCE_CORPUS: &str = "experiments/corpus/problems.jsonl";
const SHALLOW: &str = "experiments/models/policy.safetensors";
const COMPOSITIONAL: &str = "experiments/models/policy_compositional.safetensors";
const SEED: u64 = 0x4c45_4d4d_415f_f001;
const DEPTHS: [usize; 5] = [8, 12, 16, 24, 32];
const PER_TIER: usize = 60;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Problem {
    id: String,
    track: String,
    family: String,
    source_id: String,
    construction_depth: usize,
    input: String,
    expected: String,
    reference_rules: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct SourceProblem {
    id: String,
    family: String,
    split: String,
    construction_depth: usize,
    input: String,
    expected: String,
    reference_rules: Vec<String>,
}

#[derive(Default, Serialize)]
struct Cell {
    solved: usize,
    total: usize,
    steps: usize,
}

#[derive(Serialize)]
struct Arm {
    name: String,
    budget: usize,
    solved: usize,
    total: usize,
    elapsed_ms: u128,
    cells: BTreeMap<String, Cell>,
}

fn b(x: Expr) -> Box<Expr> {
    Box::new(x)
}
fn add(x: Expr, y: Expr) -> Expr {
    Expr::Add(b(x), b(y))
}
fn mul(x: Expr, y: Expr) -> Expr {
    Expr::Mul(b(x), b(y))
}
/// Two-step collision layer. At the multiplication state both identity-multiplication and
/// distribution are legal and verifier-accepted. The recorded identity-first route provides
/// a checked witness to the exact target; competing routes remain available to search.
fn collision_layer(inner: Expr, variant: usize) -> Expr {
    if variant % 2 == 0 {
        mul(Expr::int(1), add(inner, Expr::int(0)))
    } else {
        mul(add(inner, Expr::int(0)), Expr::int(1))
    }
}

fn read_source() -> Result<Vec<SourceProblem>, String> {
    BufReader::new(File::open(SOURCE_CORPUS).map_err(|e| e.to_string())?)
        .lines()
        .enumerate()
        .map(|(i, line)| {
            serde_json::from_str(&line.map_err(|e| e.to_string())?)
                .map_err(|e| format!("source line {}: {e}", i + 1))
        })
        .collect()
}

fn select_cell<'a>(
    source: &'a [SourceProblem],
    split: &str,
    target_depth: usize,
    used: &mut HashSet<String>,
) -> Result<Vec<&'a SourceProblem>, String> {
    let mut by_family: BTreeMap<&str, Vec<&SourceProblem>> = BTreeMap::new();
    for p in source.iter().filter(|p| {
        p.split == split
            && p.construction_depth <= target_depth
            && (target_depth - p.construction_depth) % 2 == 0
            && !used.contains(&p.id)
    }) {
        by_family.entry(&p.family).or_default().push(p);
    }
    for candidates in by_family.values_mut() {
        candidates.sort_by_key(|p| &p.id);
    }
    let families: Vec<_> = by_family.keys().copied().collect();
    let mut offsets = BTreeMap::<&str, usize>::new();
    let mut selected = Vec::with_capacity(PER_TIER / 2);
    while selected.len() < PER_TIER / 2 {
        let mut progressed = false;
        for family in &families {
            let offset = offsets.entry(family).or_default();
            if let Some(problem) = by_family[*family].get(*offset) {
                selected.push(*problem);
                used.insert(problem.id.clone());
                *offset += 1;
                progressed = true;
                if selected.len() == PER_TIER / 2 {
                    break;
                }
            }
        }
        if !progressed {
            return Err(format!(
                "not enough unused {split} sources for target depth {target_depth}"
            ));
        }
    }
    Ok(selected)
}

fn generate() -> Result<Vec<Problem>, String> {
    let source = read_source()?;
    let mut out = Vec::with_capacity(DEPTHS.len() * PER_TIER);
    let mut used = HashSet::new();
    for target_depth in DEPTHS {
        for split in ["ID", "OOD"] {
            let selected = select_cell(&source, split, target_depth, &mut used)?;
            for (index, base) in selected.into_iter().enumerate() {
                let mut symbols = SymbolTable::new();
                let mut parser = Parser::new(&mut symbols);
                let mut input = parser
                    .parse(&base.input)
                    .map_err(|e| format!("source {} input: {e}", base.id))?;
                let expected = parser
                    .parse(&base.expected)
                    .map_err(|e| format!("source {} expected: {e}", base.id))?;
                let wrapper_layers = (target_depth - base.construction_depth) / 2;
                for layer in 0..wrapper_layers {
                    input = collision_layer(input, index + layer);
                }
                let mut reference_rules = Vec::with_capacity(target_depth);
                for _ in 0..wrapper_layers {
                    reference_rules.push("algebra::identity_mul_one".to_string());
                    reference_rules.push("algebra::identity_add_zero".to_string());
                }
                reference_rules.extend(base.reference_rules.iter().cloned());
                out.push(Problem {
                    id: format!(
                        "frontier-{}-d{target_depth}-{index:02}-{}",
                        split.to_lowercase(),
                        base.id
                    ),
                    track: format!("wrapped-{split}"),
                    family: base.family.clone(),
                    source_id: base.id.clone(),
                    construction_depth: target_depth,
                    input: format_expr(&input, &symbols),
                    expected: format_expr(&expected, &symbols),
                    reference_rules,
                });
            }
        }
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(out)
}

fn write(path: &Path, problems: &[Problem]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let mut w = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);
    for p in problems {
        serde_json::to_writer(&mut w, p).map_err(|e| e.to_string())?;
        writeln!(w).map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())
}

fn read(path: &Path) -> Result<Vec<Problem>, String> {
    BufReader::new(File::open(path).map_err(|e| e.to_string())?)
        .lines()
        .enumerate()
        .map(|(i, l)| {
            serde_json::from_str(&l.map_err(|e| e.to_string())?)
                .map_err(|e| format!("line {}: {e}", i + 1))
        })
        .collect()
}

fn rule<'a>(rules: &'a RuleSet, key: &str) -> Option<&'a mm_rules::Rule> {
    rules
        .keys()
        .iter()
        .zip(rules.all())
        .find(|(k, _)| k.to_string() == key)
        .map(|(_, r)| r)
}

fn branches(rules: &RuleSet, verifier: &Verifier, state: &Expr) -> usize {
    let ctx = RuleContext::default();
    let profile = mm_boink::analyze(state);
    mm_boink::filter_rules(rules.all(), &profile)
        .into_iter()
        .map(|r| {
            if !r.can_apply(state, &ctx) {
                return 0;
            }
            r.apply(state, &ctx)
                .into_iter()
                .filter(|a| {
                    a.result != *state && verifier.verify_step(state, &a.result, r, &ctx).is_valid()
                })
                .count()
        })
        .sum()
}

fn validate(problems: &[Problem]) -> Result<(usize, f64, usize), String> {
    if problems.len() != DEPTHS.len() * PER_TIER {
        return Err(format!("expected 300 problems, got {}", problems.len()));
    }
    let rules = standard_rules();
    let verifier = Verifier::new();
    let ctx = RuleContext::default();
    let mut ids = HashSet::new();
    let mut pairs = HashSet::new();
    let mut branch_min = usize::MAX;
    let mut branch_max = 0;
    let mut branch_sum = 0;
    let mut steps = 0;
    let mut cells = BTreeMap::new();
    for p in problems {
        if !ids.insert(&p.id) || !pairs.insert((&p.input, &p.expected)) {
            return Err(format!("duplicate {}", p.id));
        }
        if p.reference_rules.len() != p.construction_depth {
            return Err(format!("{} depth mismatch", p.id));
        }
        *cells
            .entry((p.track.as_str(), p.construction_depth))
            .or_insert(0usize) += 1;
        let mut syms = SymbolTable::new();
        let mut parser = Parser::new(&mut syms);
        let mut state = parser
            .parse(&p.input)
            .map_err(|e| format!("{} input: {e}", p.id))?;
        let expected = parser
            .parse(&p.expected)
            .map_err(|e| format!("{} expected: {e}", p.id))?;
        if format_expr(&state, &syms) != p.input || format_expr(&expected, &syms) != p.expected {
            return Err(format!("{} round-trip failure", p.id));
        }
        for key in &p.reference_rules {
            let n = branches(&rules, &verifier, &state);
            branch_min = branch_min.min(n);
            branch_max = branch_max.max(n);
            branch_sum += n;
            steps += 1;
            let r = rule(&rules, key).ok_or_else(|| format!("{} missing {key}", p.id))?;
            if !r.can_apply(&state, &ctx) {
                return Err(format!("{} cannot apply {key}", p.id));
            }
            let apps: Vec<_> = r
                .apply(&state, &ctx)
                .into_iter()
                .filter(|a| {
                    a.result != state && verifier.verify_step(&state, &a.result, r, &ctx).is_valid()
                })
                .collect();
            if apps.len() != 1 {
                return Err(format!("{} {key} has {} valid outputs", p.id, apps.len()));
            }
            state = apps[0].result.clone();
        }
        if state != expected {
            return Err(format!("{} reference does not reach goal", p.id));
        }
    }
    for depth in DEPTHS {
        for track in ["wrapped-ID", "wrapped-OOD"] {
            if cells.get(&(track, depth)) != Some(&(PER_TIER / 2)) {
                return Err(format!("bad cell {track}/d{depth}"));
            }
        }
    }
    Ok((branch_min, branch_sum as f64 / steps as f64, branch_max))
}

fn device() -> Device {
    #[cfg(feature = "cuda")]
    {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Device::Cpu
    }
}

fn solver(model: Option<&Path>, budget: usize) -> Result<NeuralMCTS, String> {
    let s = NeuralMCTS::with_config(
        standard_rules(),
        Verifier::new(),
        MCTSConfig {
            simulations: budget,
            max_depth: 40,
            ..Default::default()
        },
    );
    match model {
        None => Ok(s),
        Some(path) => s
            .with_policy(
                PolicyNetwork::load(path, ActionVocabulary::standard(), device())
                    .map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string()),
    }
}

fn run_arm(
    problems: &[Problem],
    name: &str,
    model: Option<&Path>,
    budget: usize,
) -> Result<Arm, String> {
    let s = solver(model, budget)?;
    let begin = Instant::now();
    let mut arm = Arm {
        name: name.into(),
        budget,
        solved: 0,
        total: problems.len(),
        elapsed_ms: 0,
        cells: BTreeMap::new(),
    };
    for (i, p) in problems.iter().enumerate() {
        let mut syms = SymbolTable::new();
        let mut parser = Parser::new(&mut syms);
        let input = parser.parse(&p.input).map_err(|e| e.to_string())?;
        let goal = parser.parse(&p.expected).map_err(|e| e.to_string())?;
        seed_sampling_rng(SEED ^ i as u64);
        let result = s.search(input, |x| *x == goal);
        let ok = result
            .as_ref()
            .is_some_and(|x| x.result == goal && x.status.replays());
        let cell = arm
            .cells
            .entry(format!("{}-d{}", p.track, p.construction_depth))
            .or_default();
        cell.total += 1;
        if ok {
            arm.solved += 1;
            cell.solved += 1;
            cell.steps += result.unwrap().steps.len();
        }
        if (i + 1) % 50 == 0 {
            println!("{name}: {}/{} solved {}", i + 1, problems.len(), arm.solved);
        }
    }
    arm.elapsed_ms = begin.elapsed().as_millis();
    Ok(arm)
}

fn main_result() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("validate");
    let path = Path::new(args.get(2).map(String::as_str).unwrap_or(DEFAULT_CORPUS));
    match cmd {
        "generate" => {
            let p = generate()?;
            let b = validate(&p)?;
            write(path, &p)?;
            println!(
                "generated {} problems; branching {}/{:.2}/{}",
                p.len(),
                b.0,
                b.1,
                b.2
            );
        }
        "validate" => {
            let p = read(path)?;
            let b = validate(&p)?;
            println!(
                "validated {} problems; branching {}/{:.2}/{}",
                p.len(),
                b.0,
                b.1,
                b.2
            );
        }
        "run" => {
            let budget = args.get(3).and_then(|x| x.parse().ok()).unwrap_or(400);
            let p = read(path)?;
            validate(&p)?;
            let uniform = run_arm(&p, "uniform", None, budget)?;
            let shallow = run_arm(&p, "shallow", Some(Path::new(SHALLOW)), budget)?;
            let compositional =
                run_arm(&p, "compositional", Some(Path::new(COMPOSITIONAL)), budget)?;
            let result = serde_json::json!({"seed": SEED, "uniform": uniform, "shallow": shallow, "compositional": compositional});
            fs::create_dir_all("experiments/frontier").map_err(|e| e.to_string())?;
            fs::write(
                "experiments/frontier/results.json",
                serde_json::to_string_pretty(&result).unwrap(),
            )
            .map_err(|e| e.to_string())?;
            println!("{}", serde_json::to_string_pretty(&result).unwrap());
        }
        _ => return Err("use generate, validate, or run".into()),
    }
    Ok(())
}

fn main() -> ExitCode {
    match main_result() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("FAIL: {e}");
            ExitCode::FAILURE
        }
    }
}
