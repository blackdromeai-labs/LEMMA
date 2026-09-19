//! Three-arm comparison on the locked evaluation corpus: uniform priors, the shallow-template
//! policy (`experiments/models/policy.safetensors`), and the compositional policy
//! (`experiments/models/policy_compositional.safetensors`).
//!
//! This tool only READS `experiments/corpus/problems.jsonl` to run search and to replay
//! reference paths for the prior-rank diagnostic below -- exactly the use the corpus is locked
//! FOR (evaluation). It does not derive training data from it and does not feed corpus contents
//! back into either model.
//!
//! Reports, per arm:
//! - a fixed simulation-budget curve {1, 2, 5, 10, 25, 50, 150}: solve rate, ID/OOD solve rate,
//!   GPU wall-clock, mean nodes-expanded (search-tree size at the point the budget ran out);
//! - for the two trained arms only, a static diagnostic independent of search budget: at every
//!   state along every problem's recorded reference path, the rank (1 = best) that the policy's
//!   own prior assigns to the historically-correct next rule among all rules that are legal
//!   (guardrail + can_apply + verified) at that state. This isolates policy quality from search
//!   variance -- a low solve rate could come from a good policy that search fails to exploit, or
//!   from priors that are simply wrong; this metric distinguishes the two.
//!
//! Usage:
//!   compare_policies [corpus.jsonl]

use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use candle_core::Device;
use mm_brain::PolicyNetwork;
use mm_core::parse::Parser;
use mm_core::sampling::seed_sampling_rng;
use mm_core::{Expr, SymbolTable};
use mm_rules::rule::RuleId;
use mm_rules::{standard_rules, ActionVocabulary, RuleContext, RuleSet};
use mm_search::{MCTSConfig, NeuralMCTS};
use mm_verifier::Verifier;
use serde::{Deserialize, Serialize};

const DEFAULT_CORPUS: &str = "experiments/corpus/problems.jsonl";
const SHALLOW_MODEL: &str = "experiments/models/policy.safetensors";
const COMPOSITIONAL_MODEL: &str = "experiments/models/policy_compositional.safetensors";
const BUDGETS: [usize; 7] = [1, 2, 5, 10, 25, 50, 150];
const SEED: u64 = 0x4c45_4d4d_415f_0001;

#[derive(Debug, Clone, Deserialize)]
struct ProblemRecord {
    id: String,
    split: String,
    // Part of the corpus record schema (see `neurosymbolic_benchmark.rs`); not read by this
    // tool, which reports by split only, not by depth.
    #[allow(dead_code)]
    construction_depth: usize,
    input: String,
    expected: String,
    reference_rules: Vec<String>,
}

fn read_corpus(path: &Path) -> Result<Vec<ProblemRecord>, String> {
    let file = fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
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

fn eval_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Device::Cpu
    }
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
        let policy = PolicyNetwork::load(path, ActionVocabulary::standard(), eval_device())
            .map_err(|e| e.to_string())?;
        solver.with_policy(policy).map_err(|e| e.to_string())
    } else {
        Ok(solver)
    }
}

#[derive(Debug, Default, Serialize, Clone)]
struct SplitCounts {
    solved: usize,
    total: usize,
    nodes_sum_solved: u64,
    nodes_sum_all: u64,
    steps_sum_solved: u64,
}

#[derive(Debug, Default, Serialize, Clone)]
struct BudgetPoint {
    arm: String,
    budget: usize,
    elapsed_ms: u128,
    id: SplitCounts,
    ood: SplitCounts,
}

fn run_budget_point(
    problems: &[ProblemRecord],
    model: Option<&Path>,
    budget: usize,
    arm: &str,
) -> Result<BudgetPoint, String> {
    let solver = make_solver(model, budget)?;
    let start = Instant::now();
    let mut point = BudgetPoint {
        arm: arm.to_string(),
        budget,
        ..Default::default()
    };
    for (index, problem) in problems.iter().enumerate() {
        let mut symbols = SymbolTable::new();
        let mut parser = Parser::new(&mut symbols);
        let input = parser.parse(&problem.input).map_err(|e| e.to_string())?;
        let expected = parser.parse(&problem.expected).map_err(|e| e.to_string())?;
        seed_sampling_rng(SEED ^ index as u64 ^ (budget as u64) << 32);
        let outcome = solver.search_best_effort(input, |candidate| *candidate == expected);
        let solved = outcome.reached_goal
            && outcome.solution.result == expected
            && outcome.solution.status.replays();
        let split = if problem.split == "ID" {
            &mut point.id
        } else {
            &mut point.ood
        };
        split.total += 1;
        split.nodes_sum_all += outcome.nodes_expanded as u64;
        if solved {
            split.solved += 1;
            split.nodes_sum_solved += outcome.nodes_expanded as u64;
            split.steps_sum_solved += outcome.solution.steps.len() as u64;
        }
    }
    point.elapsed_ms = start.elapsed().as_millis();
    println!(
        "{arm:<14} budget={budget:>4}  ID {}/{}  OOD {}/{}  {:>8}ms",
        point.id.solved, point.id.total, point.ood.solved, point.ood.total, point.elapsed_ms
    );
    Ok(point)
}

fn rule_key_map(rules: &RuleSet) -> HashMap<RuleId, String> {
    rules
        .keys()
        .iter()
        .zip(rules.all())
        .map(|(k, r)| (r.id, k.to_string()))
        .collect()
}

/// Rank (1 = best) of `correct_key` among the priors of every rule that is legal
/// (guardrail-passed, `can_apply`, produces a change, verifier-accepted) at `state`, replaying
/// exactly the filter `NeuralMCTS::expand` applies. `None` if the policy forward pass fails or
/// the correct rule is not among the legal set at this state (a corpus/verifier disagreement,
/// not expected but not assumed away).
fn prior_rank_at_state(
    policy: &PolicyNetwork,
    vocabulary: &ActionVocabulary,
    rules: &RuleSet,
    key_of: &HashMap<RuleId, String>,
    verifier: &Verifier,
    state: &Expr,
    correct_key: &str,
) -> Option<(usize, usize)> {
    let ctx = RuleContext::default();
    let raw_priors = policy.rule_priors(state).ok()?;
    let profile = mm_boink::analyze(state);
    let valid_rules = mm_boink::filter_rules(rules.all(), &profile);

    let mut legal: Vec<(String, f32)> = Vec::new();
    for rule in valid_rules {
        if !rule.can_apply(state, &ctx) {
            continue;
        }
        for app in rule.apply(state, &ctx) {
            if app.result == *state {
                continue;
            }
            if !verifier
                .verify_step(state, &app.result, rule, &ctx)
                .is_valid()
            {
                continue;
            }
            let prior = vocabulary.prior_for_rule(&raw_priors, rule.id).ok()?;
            let key = key_of.get(&rule.id).cloned().unwrap_or_default();
            legal.push((key, prior));
            break; // one entry per rule at this state, matching NeuralMCTS::expand's dedup.
        }
    }
    legal.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let total = legal.len();
    let rank = legal.iter().position(|(k, _)| k == correct_key)? + 1;
    Some((rank, total))
}

#[derive(Debug, Default, Serialize, Clone)]
struct PriorRankSplit {
    decision_points: usize,
    rank_sum: u64,
    top1: usize,
    top3: usize,
    ranks: Vec<usize>,
}

impl PriorRankSplit {
    fn record(&mut self, rank: usize) {
        self.decision_points += 1;
        self.rank_sum += rank as u64;
        if rank == 1 {
            self.top1 += 1;
        }
        if rank <= 3 {
            self.top3 += 1;
        }
        self.ranks.push(rank);
    }

    fn median(&self) -> f64 {
        if self.ranks.is_empty() {
            return f64::NAN;
        }
        let mut sorted = self.ranks.clone();
        sorted.sort_unstable();
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) as f64 / 2.0
        } else {
            sorted[mid] as f64
        }
    }

    fn mean(&self) -> f64 {
        self.rank_sum as f64 / self.decision_points.max(1) as f64
    }
}

fn prior_rank_report(
    problems: &[ProblemRecord],
    model_path: &Path,
    arm: &str,
) -> Result<serde_json::Value, String> {
    let rules = standard_rules();
    let verifier = Verifier::new();
    let key_of = rule_key_map(&rules);
    let policy = PolicyNetwork::load(model_path, ActionVocabulary::standard(), eval_device())
        .map_err(|e| e.to_string())?;
    let vocabulary = ActionVocabulary::standard();
    let ctx = RuleContext::default();

    let mut id_split = PriorRankSplit::default();
    let mut ood_split = PriorRankSplit::default();
    let mut skipped = 0usize;

    for problem in problems {
        let mut symbols = SymbolTable::new();
        let mut parser = Parser::new(&mut symbols);
        let mut state = parser.parse(&problem.input).map_err(|e| e.to_string())?;
        for key in &problem.reference_rules {
            match prior_rank_at_state(
                &policy,
                &vocabulary,
                &rules,
                &key_of,
                &verifier,
                &state,
                key,
            ) {
                Some((rank, _total)) => {
                    if problem.split == "ID" {
                        id_split.record(rank);
                    } else {
                        ood_split.record(rank);
                    }
                }
                None => skipped += 1,
            }
            // Advance the replay along the recorded reference path, exactly as validate() does.
            let rule = rules
                .keys()
                .iter()
                .zip(rules.all())
                .find(|(k, _)| k.to_string() == *key)
                .map(|(_, r)| r)
                .ok_or_else(|| format!("{} missing rule {key}", problem.id))?;
            let applications = rule.apply(&state, &ctx);
            let next = applications
                .into_iter()
                .find(|app| app.result != state)
                .ok_or_else(|| format!("{} rule {key} produced no change", problem.id))?
                .result;
            state = next;
        }
    }

    println!(
        "{arm:<14} prior-rank  ID: mean={:.2} median={:.1} top1={}/{}  OOD: mean={:.2} median={:.1} top1={}/{}  (skipped {skipped})",
        id_split.mean(), id_split.median(), id_split.top1, id_split.decision_points,
        ood_split.mean(), ood_split.median(), ood_split.top1, ood_split.decision_points,
    );

    Ok(serde_json::json!({
        "arm": arm,
        "id": {
            "decision_points": id_split.decision_points,
            "mean_rank": id_split.mean(),
            "median_rank": id_split.median(),
            "top1_rate": id_split.top1 as f64 / id_split.decision_points.max(1) as f64,
            "top3_rate": id_split.top3 as f64 / id_split.decision_points.max(1) as f64,
        },
        "ood": {
            "decision_points": ood_split.decision_points,
            "mean_rank": ood_split.mean(),
            "median_rank": ood_split.median(),
            "top1_rate": ood_split.top1 as f64 / ood_split.decision_points.max(1) as f64,
            "top3_rate": ood_split.top3 as f64 / ood_split.decision_points.max(1) as f64,
        },
        "skipped_decision_points": skipped,
    }))
}

fn real_main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let corpus = Path::new(args.get(1).map(String::as_str).unwrap_or(DEFAULT_CORPUS));
    let problems = read_corpus(corpus)?;
    println!(
        "loaded {} problems from {}\n",
        problems.len(),
        corpus.display()
    );

    println!("== budget curve ==");
    let mut budget_points = Vec::new();
    for &budget in &BUDGETS {
        budget_points.push(run_budget_point(&problems, None, budget, "uniform")?);
        budget_points.push(run_budget_point(
            &problems,
            Some(Path::new(SHALLOW_MODEL)),
            budget,
            "shallow",
        )?);
        budget_points.push(run_budget_point(
            &problems,
            Some(Path::new(COMPOSITIONAL_MODEL)),
            budget,
            "compositional",
        )?);
    }

    println!("\n== prior-rank diagnostic (trained arms only, budget-independent) ==");
    let shallow_rank = prior_rank_report(&problems, Path::new(SHALLOW_MODEL), "shallow")?;
    let compositional_rank =
        prior_rank_report(&problems, Path::new(COMPOSITIONAL_MODEL), "compositional")?;

    fs::create_dir_all("experiments/results").map_err(|e| e.to_string())?;
    let result = serde_json::json!({
        "budgets": BUDGETS,
        "budget_curve": budget_points,
        "prior_rank": {
            "shallow": shallow_rank,
            "compositional": compositional_rank,
        },
    });
    fs::write(
        "experiments/results/three_arm_comparison.json",
        serde_json::to_string_pretty(&result).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    println!("\nwrote experiments/results/three_arm_comparison.json");
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
