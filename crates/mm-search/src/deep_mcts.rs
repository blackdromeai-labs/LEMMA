//! Parallel tree search with a large node budget.
//!
//! What this is: a multi-threaded best-first tree search over rule applications, with
//! constant action priors (0.15 at the top level, 0.1 for sub-expression rewrites), no
//! learned leaf evaluation, and a binary success/failure backup.
//!
//! What it is not: AlphaZero-style neural MCTS. It does not consult a policy or value
//! network. [`crate::mcts::NeuralMCTS`] is the module that does.
//!
//! Statistics reported by [`DeepMCTS::search`] are measured, not estimated. `nodes_expanded`
//! and `max_depth_reached` were previously derived from the configuration
//! (`nodes_explored / 10` and the configured depth limit) rather than observed.

use mm_core::Expr;
use mm_rules::{RuleContext, RuleId, RuleSet};
use mm_verifier::Verifier;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Configuration for deep MCTS search
#[derive(Debug, Clone)]
pub struct DeepMCTSConfig {
    /// Maximum nodes to explore (10M+ for real problems)
    pub max_nodes: u64,
    /// Time limit in seconds (3600 = 1 hour)
    pub time_limit_secs: u64,
    /// Number of parallel workers
    pub num_workers: usize,
    /// Maximum search depth per simulation
    pub max_depth: usize,
    /// Exploration constant (c_puct)
    pub exploration_weight: f64,
    /// Virtual loss for parallel MCTS
    pub virtual_loss: f64,
    /// Print progress every N nodes
    pub progress_interval: u64,
}

impl Default for DeepMCTSConfig {
    fn default() -> Self {
        Self {
            max_nodes: 10_000_000, // 10M nodes
            time_limit_secs: 3600, // 1 hour
            num_workers: num_cpus::get(),
            max_depth: 100,
            exploration_weight: 1.41,
            virtual_loss: 3.0,
            progress_interval: 100_000,
        }
    }
}

impl DeepMCTSConfig {
    /// Quick search: 100K nodes, 1 minute
    pub fn quick() -> Self {
        Self {
            max_nodes: 100_000,
            time_limit_secs: 60,
            ..Default::default()
        }
    }

    /// Medium search: 1M nodes, 10 minutes
    pub fn medium() -> Self {
        Self {
            max_nodes: 1_000_000,
            time_limit_secs: 600,
            ..Default::default()
        }
    }

    /// Deep search: 100M nodes, 1 hour
    pub fn deep() -> Self {
        Self {
            max_nodes: 100_000_000,
            time_limit_secs: 3600,
            ..Default::default()
        }
    }

    /// Maximum search: 1B nodes, 24 hours
    pub fn maximum() -> Self {
        Self {
            max_nodes: 1_000_000_000,
            time_limit_secs: 86400,
            ..Default::default()
        }
    }
}

/// A node in the deep MCTS tree (memory-optimized)
#[derive(Debug)]
pub struct DeepNode {
    /// Current state (expression)
    pub state: Expr,
    /// Visit count
    pub visits: AtomicU64,
    /// Sum of values from simulations
    pub value_sum: AtomicU64, // Stored as fixed-point for atomics
    /// Prior probability from policy network
    pub prior: f32,
    /// Rule that produced this node
    pub rule_id: Option<RuleId>,
    /// Children (lazily expanded)
    pub children: RwLock<Vec<Arc<DeepNode>>>,
    /// Is this node expanded?
    pub expanded: AtomicBool,
}

impl DeepNode {
    pub fn new(state: Expr, prior: f32) -> Self {
        Self {
            state,
            visits: AtomicU64::new(0),
            value_sum: AtomicU64::new(0),
            prior,
            rule_id: None,
            children: RwLock::new(Vec::new()),
            expanded: AtomicBool::new(false),
        }
    }

    pub fn with_rule(state: Expr, prior: f32, rule_id: RuleId) -> Self {
        Self {
            state,
            visits: AtomicU64::new(0),
            value_sum: AtomicU64::new(0),
            prior,
            rule_id: Some(rule_id),
            children: RwLock::new(Vec::new()),
            expanded: AtomicBool::new(false),
        }
    }

    /// Get average value (0.0 to 1.0)
    pub fn value(&self) -> f64 {
        let visits = self.visits.load(Ordering::Relaxed);
        if visits == 0 {
            return 0.5;
        }
        let sum = self.value_sum.load(Ordering::Relaxed);
        // Fixed-point: stored as value * 1000000
        (sum as f64) / (visits as f64 * 1_000_000.0)
    }

    /// Calculate UCB score (PUCT formula)
    pub fn ucb_score(&self, parent_visits: u64, c_puct: f64) -> f64 {
        let visits = self.visits.load(Ordering::Relaxed);
        let value = self.value();

        let exploration =
            c_puct * (self.prior as f64) * ((parent_visits as f64).sqrt()) / (1.0 + visits as f64);

        value + exploration
    }
}

/// Measured statistics for a search.
#[derive(Debug, Clone, Default)]
pub struct SearchStats {
    /// Nodes visited by a simulation.
    pub nodes_explored: u64,
    /// Nodes whose children were generated. Counted, not estimated.
    pub nodes_expanded: u64,
    /// Deepest simulation depth actually reached. Observed, not the configured limit.
    pub max_depth_reached: usize,
    /// Number of goal paths found.
    pub solutions_found: u64,
    /// Wall-clock duration.
    pub elapsed_seconds: f64,
    /// Nodes explored per second.
    pub nodes_per_second: f64,
}

/// Parallel tree search solver.
pub struct DeepMCTS {
    pub rules: RuleSet,
    pub verifier: Verifier,
    pub config: DeepMCTSConfig,
}

impl DeepMCTS {
    pub fn new(rules: RuleSet, verifier: Verifier) -> Self {
        Self {
            rules,
            verifier,
            config: DeepMCTSConfig::default(),
        }
    }

    pub fn with_config(rules: RuleSet, verifier: Verifier, config: DeepMCTSConfig) -> Self {
        Self {
            rules,
            verifier,
            config,
        }
    }

    /// Search for a solution with deep MCTS
    pub fn search<F>(&self, start: Expr, goal: F) -> (Option<Vec<Expr>>, SearchStats)
    where
        F: Fn(&Expr) -> bool + Sync,
    {
        let start_time = Instant::now();
        let deadline = start_time + Duration::from_secs(self.config.time_limit_secs);

        // Check if already at goal
        if goal(&start) {
            return (Some(vec![start.clone()]), SearchStats::default());
        }

        // Create root node
        let root = Arc::new(DeepNode::new(start.clone(), 1.0));

        // Shared state for parallel search
        let nodes_explored = AtomicU64::new(0);
        let nodes_expanded = AtomicU64::new(0);
        let max_depth_reached = AtomicUsize::new(0);
        let solutions_found = AtomicU64::new(0);
        let found_solution: RwLock<Option<Vec<Expr>>> = RwLock::new(None);
        let should_stop = AtomicBool::new(false);

        // Run parallel simulations
        (0..self.config.num_workers)
            .into_par_iter()
            .for_each(|_worker_id| {
                while !should_stop.load(Ordering::Relaxed) {
                    // Check termination conditions
                    let explored = nodes_explored.load(Ordering::Relaxed);
                    if explored >= self.config.max_nodes {
                        should_stop.store(true, Ordering::Relaxed);
                        break;
                    }

                    if Instant::now() > deadline {
                        should_stop.store(true, Ordering::Relaxed);
                        break;
                    }

                    // Run one simulation
                    let counters = Counters {
                        nodes_explored: &nodes_explored,
                        nodes_expanded: &nodes_expanded,
                        max_depth_reached: &max_depth_reached,
                    };
                    if let Some(path) = self.simulate(Arc::clone(&root), &goal, 0, &counters) {
                        // Found a solution!
                        solutions_found.fetch_add(1, Ordering::Relaxed);
                        let mut sol = found_solution.write().unwrap();
                        if sol.is_none() {
                            *sol = Some(path);
                        }
                        should_stop.store(true, Ordering::Relaxed);
                    }

                    // Progress reporting
                    let current = nodes_explored.load(Ordering::Relaxed);
                    if current % self.config.progress_interval == 0 && current > 0 {
                        let elapsed = start_time.elapsed().as_secs_f64();
                        let rate = current as f64 / elapsed;
                        eprintln!(
                            "  Explored {} nodes ({:.0} nodes/sec, {:.1}s elapsed)",
                            current, rate, elapsed
                        );
                    }
                }
            });

        let elapsed = start_time.elapsed();
        let total_nodes = nodes_explored.load(Ordering::Relaxed);

        let stats = SearchStats {
            nodes_explored: total_nodes,
            nodes_expanded: nodes_expanded.load(Ordering::Relaxed),
            max_depth_reached: max_depth_reached.load(Ordering::Relaxed),
            solutions_found: solutions_found.load(Ordering::Relaxed),
            elapsed_seconds: elapsed.as_secs_f64(),
            nodes_per_second: total_nodes as f64 / elapsed.as_secs_f64(),
        };

        let solution = found_solution.read().unwrap().clone();
        (solution, stats)
    }

    /// Run one simulation (SELECT, EXPAND, BACKUP).
    fn simulate<F>(
        &self,
        node: Arc<DeepNode>,
        goal: &F,
        depth: usize,
        counters: &Counters<'_>,
    ) -> Option<Vec<Expr>>
    where
        F: Fn(&Expr) -> bool + Sync,
    {
        counters.nodes_explored.fetch_add(1, Ordering::Relaxed);
        counters.record_depth(depth);

        // Check if at goal
        if goal(&node.state) {
            return Some(vec![node.state.clone()]);
        }

        // Check depth limit
        if depth >= self.config.max_depth {
            return None;
        }

        // EXPAND if needed
        if !node.expanded.load(Ordering::Relaxed)
            && node
                .expanded
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
        {
            self.expand(&node);
            counters.nodes_expanded.fetch_add(1, Ordering::Relaxed);
        }

        // SELECT best child
        let children = node.children.read().unwrap();
        if children.is_empty() {
            return None;
        }

        let parent_visits = node.visits.load(Ordering::Relaxed);
        let mut best_score = f64::NEG_INFINITY;
        let mut best_child = None;

        for child in children.iter() {
            let score = child.ucb_score(parent_visits, self.config.exploration_weight);
            if score > best_score {
                best_score = score;
                best_child = Some(Arc::clone(child));
            }
        }
        drop(children);

        // RECURSE
        if let Some(child) = best_child {
            // Apply virtual loss
            child.visits.fetch_add(1, Ordering::Relaxed);

            let result = self.simulate(child.clone(), goal, depth + 1, counters);

            // BACKUP
            let value = if result.is_some() { 1.0 } else { 0.0 };
            let value_fixed = (value * 1_000_000.0) as u64;
            node.visits.fetch_add(1, Ordering::Relaxed);
            node.value_sum.fetch_add(value_fixed, Ordering::Relaxed);

            if let Some(mut path) = result {
                path.insert(0, node.state.clone());
                return Some(path);
            }
        }

        None
    }

    /// Expand a node by generating children
    /// Applies rules both at top level AND to sub-expressions
    fn expand(&self, node: &DeepNode) {
        let ctx = RuleContext::default();
        let mut new_children = Vec::new();

        // Apply rules at TOP LEVEL
        for rule in self.rules.applicable(&node.state, &ctx) {
            let applications = rule.apply(&node.state, &ctx);

            for app in applications.into_iter().take(5) {
                let verify_result = self
                    .verifier
                    .verify_step(&node.state, &app.result, rule, &ctx);

                if verify_result.is_valid() {
                    let child = Arc::new(DeepNode::with_rule(
                        app.result, 0.15, // Higher prior for top-level
                        rule.id,
                    ));
                    new_children.push(child);
                }
            }
        }

        // Apply rules to SUB-EXPRESSIONS (one level deep)
        let sub_results = self.apply_to_subexpressions(&node.state, &ctx);
        for (result, rule_id) in sub_results.into_iter().take(10) {
            let child = Arc::new(DeepNode::with_rule(
                result, 0.1, // Lower prior for sub-expression changes
                rule_id,
            ));
            new_children.push(child);
        }

        let mut children = node.children.write().unwrap();
        children.extend(new_children);
    }

    /// Apply rules to immediate sub-expressions and return transformed results
    fn apply_to_subexpressions(&self, expr: &Expr, ctx: &RuleContext) -> Vec<(Expr, RuleId)> {
        let mut results = Vec::new();

        match expr {
            Expr::Add(a, b) => {
                // Try applying rules to 'a'
                for rule in self.rules.applicable(a, ctx) {
                    for app in rule.apply(a, ctx).into_iter().take(2) {
                        let new_expr = Expr::Add(Box::new(app.result), b.clone());
                        results.push((new_expr, rule.id));
                    }
                }
                // Try applying rules to 'b'
                for rule in self.rules.applicable(b, ctx) {
                    for app in rule.apply(b, ctx).into_iter().take(2) {
                        let new_expr = Expr::Add(a.clone(), Box::new(app.result));
                        results.push((new_expr, rule.id));
                    }
                }
            }
            Expr::Sub(a, b) => {
                for rule in self.rules.applicable(a, ctx) {
                    for app in rule.apply(a, ctx).into_iter().take(2) {
                        let new_expr = Expr::Sub(Box::new(app.result), b.clone());
                        results.push((new_expr, rule.id));
                    }
                }
                for rule in self.rules.applicable(b, ctx) {
                    for app in rule.apply(b, ctx).into_iter().take(2) {
                        let new_expr = Expr::Sub(a.clone(), Box::new(app.result));
                        results.push((new_expr, rule.id));
                    }
                }
            }
            Expr::Mul(a, b) => {
                for rule in self.rules.applicable(a, ctx) {
                    for app in rule.apply(a, ctx).into_iter().take(2) {
                        let new_expr = Expr::Mul(Box::new(app.result), b.clone());
                        results.push((new_expr, rule.id));
                    }
                }
                for rule in self.rules.applicable(b, ctx) {
                    for app in rule.apply(b, ctx).into_iter().take(2) {
                        let new_expr = Expr::Mul(a.clone(), Box::new(app.result));
                        results.push((new_expr, rule.id));
                    }
                }
            }
            Expr::Div(a, b) => {
                for rule in self.rules.applicable(a, ctx) {
                    for app in rule.apply(a, ctx).into_iter().take(2) {
                        let new_expr = Expr::Div(Box::new(app.result), b.clone());
                        results.push((new_expr, rule.id));
                    }
                }
                for rule in self.rules.applicable(b, ctx) {
                    for app in rule.apply(b, ctx).into_iter().take(2) {
                        let new_expr = Expr::Div(a.clone(), Box::new(app.result));
                        results.push((new_expr, rule.id));
                    }
                }
            }
            Expr::Pow(base, exp) => {
                for rule in self.rules.applicable(base, ctx) {
                    for app in rule.apply(base, ctx).into_iter().take(2) {
                        let new_expr = Expr::Pow(Box::new(app.result), exp.clone());
                        results.push((new_expr, rule.id));
                    }
                }
            }
            Expr::Neg(inner) => {
                for rule in self.rules.applicable(inner, ctx) {
                    for app in rule.apply(inner, ctx).into_iter().take(2) {
                        let new_expr = Expr::Neg(Box::new(app.result));
                        results.push((new_expr, rule.id));
                    }
                }
            }
            _ => {}
        }

        results
    }
}

/// Shared measurement counters for one search.
struct Counters<'a> {
    nodes_explored: &'a AtomicU64,
    nodes_expanded: &'a AtomicU64,
    max_depth_reached: &'a AtomicUsize,
}

impl Counters<'_> {
    /// Record the deepest simulation depth seen so far.
    fn record_depth(&self, depth: usize) {
        self.max_depth_reached.fetch_max(depth, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_rules::rule::{standard_rules, Rule, RuleApplication, RuleSet};
    use mm_rules::{Domain, Feature, RuleCategory};

    /// A rule that only ever matches a bare variable, so a constant root has no children.
    fn variable_only_rule() -> Rule {
        Rule {
            id: RuleId(1),
            name: "wrap_variable",
            category: RuleCategory::Simplification,
            description: "x -> x + 0",
            domains: &[] as &[Domain],
            requires: &[] as &[Feature],
            is_applicable: |e, _| matches!(e, Expr::Var(_)),
            apply: |e, _| {
                vec![RuleApplication {
                    result: Expr::Add(Box::new(e.clone()), Box::new(Expr::int(0))),
                    justification: "x -> x + 0".to_string(),
                }]
            },
            reversible: false,
            cost: 1,
        }
    }

    #[test]
    fn test_config_presets() {
        let quick = DeepMCTSConfig::quick();
        assert_eq!(quick.max_nodes, 100_000);

        let deep = DeepMCTSConfig::deep();
        assert_eq!(deep.max_nodes, 100_000_000);
    }

    #[test]
    fn statistics_are_measured_not_derived_from_the_configuration() {
        let mut rules = RuleSet::new();
        rules.try_add("test", variable_only_rule()).unwrap();

        let configured_depth = 32;
        let search = DeepMCTS::with_config(
            rules,
            Verifier::new(),
            DeepMCTSConfig {
                max_nodes: 200,
                time_limit_secs: 10,
                max_depth: configured_depth,
                num_workers: 1,
                ..DeepMCTSConfig::default()
            },
        );

        // The root is a constant, so no rule applies and the tree has no children at all.
        let (solution, stats) = search.search(Expr::int(5), |e: &Expr| *e == Expr::int(4242));

        assert!(solution.is_none());
        assert!(stats.nodes_explored > 0);
        assert_eq!(
            stats.max_depth_reached, 0,
            "max_depth_reached must be the depth actually reached, not the configured limit              of {configured_depth}"
        );
        assert_eq!(
            stats.nodes_expanded, 1,
            "exactly one node (the root) is expanded here; the count must not be estimated              as nodes_explored / 10"
        );
        assert_ne!(stats.nodes_expanded, stats.nodes_explored / 10);
    }

    #[test]
    fn depth_is_recorded_when_the_search_descends() {
        let mut symbols = mm_core::SymbolTable::new();
        let x = symbols.intern("x");
        let mut rules = RuleSet::new();
        rules.try_add("test", variable_only_rule()).unwrap();

        let search = DeepMCTS::with_config(
            rules,
            Verifier::new(),
            DeepMCTSConfig {
                max_nodes: 100,
                time_limit_secs: 10,
                max_depth: 6,
                num_workers: 1,
                ..DeepMCTSConfig::default()
            },
        );

        let (_solution, stats) = search.search(Expr::Var(x), |e: &Expr| *e == Expr::int(4242));
        assert!(
            stats.max_depth_reached > 0,
            "the search should have descended"
        );
        assert!(stats.max_depth_reached <= 6);
    }

    #[test]
    fn a_goal_at_the_root_is_reported_immediately() {
        let search = DeepMCTS::with_config(
            standard_rules(),
            Verifier::new(),
            DeepMCTSConfig {
                max_nodes: 10,
                time_limit_secs: 1,
                num_workers: 1,
                ..DeepMCTSConfig::default()
            },
        );

        let (solution, _stats) = search.search(Expr::int(5), |e: &Expr| *e == Expr::int(5));
        assert_eq!(solution, Some(vec![Expr::int(5)]));
    }
}
