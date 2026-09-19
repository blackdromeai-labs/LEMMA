// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Author: Pushp Kharat

//! Monte Carlo Tree Search over rule applications.
//!
//! Structure follows AlphaZero-style MCTS: a policy supplies action priors, a value head
//! scores leaves, and PUCT balances exploration against exploitation. What the search
//! guarantees, and what it does not:
//!
//! - Children are keyed by [`ActionKey`], which is a rule *and* which of that rule's outputs
//!   was taken. Keying by rule identifier alone silently discarded every application after
//!   the first.
//! - Reaching the goal backs up through the terminal node, so a goal node cannot stay at zero
//!   visits and lose the final extraction to an unrelated sibling.
//! - [`NeuralMCTS::search`] returns a solution only when a goal node was actually reached;
//!   [`NeuralMCTS::search_best_effort`] returns the partial path separately and labels it.
//! - Every expansion is checked by the verifier, and each child records the evidence, so a
//!   returned trace can be replayed by [`crate::assess_trace`].
//! - Ties in selection and extraction break towards the lower child index, and children are
//!   held in an ordered vector, so two runs of the same search visit the same nodes.
//!
//! The default policy network is untrained. [`NeuralMCTS::provenance`] reports that, and the
//! search does not consult it: randomly initialised weights carry no signal, and reading them
//! would make results differ from process to process for no gain. With an untrained model the
//! search uses uniform priors and a constant leaf value, which is what it actually has.

use crate::{Solution, Step};
use mm_brain::{ModelProvenance, PolicyNetwork};
use mm_core::{Expr, Rational};
use mm_rules::{ActionVocabulary, RuleCategory, RuleContext, RuleId, RuleSet};
use mm_verifier::{StepEvidence, VerificationMethod, VerificationStatus, Verifier};
use std::collections::HashSet;

/// The weakest evidence in a set, or symbolic equivalence if the set is empty.
///
/// A phase is only as well established as its least well established transition.
fn weakest_evidence(evidence: &[StepEvidence]) -> StepEvidence {
    let mut weakest = VerificationMethod::SymbolicEquivalence;
    for e in evidence {
        match e.method() {
            Some(m) => {
                if m > weakest {
                    weakest = m;
                }
            }
            None => return StepEvidence::Unchecked,
        }
    }
    StepEvidence::Checked(weakest)
}

/// Compute GCD using Euclidean algorithm.
fn gcd(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Compute factorial.
fn factorial(n: u64) -> u64 {
    (1..=n).product()
}

/// Identity of an edge in the search tree.
///
/// A rule can produce several distinct applications for one expression; each is its own
/// action. Keying children by `rule_id` alone made later applications overwrite earlier ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ActionKey {
    /// The rule that was applied.
    pub rule_id: RuleId,
    /// Index of the application within that rule's output for the parent state.
    pub application: usize,
}

/// A node in the MCTS tree.
pub struct MCTSNode {
    /// The expression at this node.
    pub state: Expr,
    /// Number of times this node has been visited.
    pub visits: u32,
    /// Sum of values from rollouts through this node.
    pub value_sum: f64,
    /// Prior probability from the policy network.
    pub prior: f64,
    /// Action that led to this state (None for root).
    pub action: Option<ActionKey>,
    /// Rule name for step recording.
    pub rule_name: Option<&'static str>,
    /// Justification recorded by the rule application.
    pub justification: String,
    /// Evidence the verifier produced for the transition into this node.
    pub evidence: StepEvidence,
    /// Child nodes, in insertion order.
    pub children: Vec<Box<MCTSNode>>,
    /// Whether this node has been expanded.
    pub expanded: bool,
}

impl MCTSNode {
    /// Create a new root node.
    pub fn new(state: Expr, prior: f64) -> Self {
        Self {
            state,
            visits: 0,
            value_sum: 0.0,
            prior,
            action: None,
            rule_name: None,
            justification: String::new(),
            evidence: StepEvidence::Unchecked,
            children: Vec::new(),
            expanded: false,
        }
    }

    /// Create a child node reached by an action.
    pub fn with_action(
        state: Expr,
        prior: f64,
        action: ActionKey,
        rule_name: &'static str,
        justification: String,
        evidence: StepEvidence,
    ) -> Self {
        Self {
            state,
            visits: 0,
            value_sum: 0.0,
            prior,
            action: Some(action),
            rule_name: Some(rule_name),
            justification,
            evidence,
            children: Vec::new(),
            expanded: false,
        }
    }

    /// Get the average value of this node.
    pub fn value(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f64
        }
    }

    /// Calculate UCB score for selection (PUCT formula from AlphaZero).
    pub fn ucb_score(&self, parent_visits: u32, exploration_weight: f64) -> f64 {
        if self.visits == 0 {
            // Prefer unexplored nodes with high prior
            exploration_weight * self.prior * (parent_visits as f64).sqrt()
        } else {
            // Q + U formula
            self.value()
                + exploration_weight
                    * self.prior
                    * ((parent_visits as f64).sqrt() / (1.0 + self.visits as f64))
        }
    }

    /// Record a rollout result at this node.
    fn back_up(&mut self, value: f64) {
        self.visits += 1;
        self.value_sum += value;
    }

    /// Step describing the transition into this node from `before`.
    fn to_step(&self, before: Expr) -> Option<Step> {
        let action = self.action?;
        let name = self.rule_name?;
        Some(Step::rule(
            before,
            self.state.clone(),
            action.rule_id,
            name,
            self.justification.clone(),
            self.evidence,
        ))
    }
}

/// Neural-guided Monte Carlo Tree Search solver.
pub struct NeuralMCTS {
    rules: RuleSet,
    verifier: Verifier,
    policy: PolicyNetwork,
    vocabulary: ActionVocabulary,
    config: MCTSConfig,
}

/// MCTS configuration.
#[derive(Debug, Clone)]
pub struct MCTSConfig {
    /// Number of MCTS simulations per search.
    pub simulations: usize,
    /// Exploration weight (c_puct in AlphaZero).
    pub exploration_weight: f64,
    /// Maximum search depth.
    pub max_depth: usize,
    /// Maximum number of iterations [`NeuralMCTS::simplify`] will chain.
    pub max_simplify_iterations: usize,
}

impl Default for MCTSConfig {
    fn default() -> Self {
        Self {
            simulations: 100,
            exploration_weight: 1.41,
            max_depth: 20,
            max_simplify_iterations: 50,
        }
    }
}

/// Outcome of a search, distinguishing a reached goal from a partial path.
#[derive(Debug, Clone)]
pub struct SearchOutcome {
    /// The best path found, as a solution with its derived status.
    pub solution: Solution,
    /// Whether the final state satisfies the goal predicate.
    pub reached_goal: bool,
    /// Total nodes in the search tree when the budget was exhausted (or 1, if the start state
    /// already satisfied the goal and no simulation ran at all).
    ///
    /// Pure post-hoc counting over the finished tree -- it does not read anything `simulate`,
    /// `expand`, or `select_child` do not already produce, and adding it changes no selection
    /// or expansion decision. This is a size metric, not a claim about which nodes mattered;
    /// two searches that visit the same nodes a different number of times report the same
    /// count here.
    pub nodes_expanded: usize,
}

/// Count of nodes in a tree, root included.
fn count_nodes(node: &MCTSNode) -> usize {
    1 + node.children.iter().map(|c| count_nodes(c)).sum::<usize>()
}

impl NeuralMCTS {
    /// Create a solver backed by an untrained policy network.
    ///
    /// The priors are random. [`Self::provenance`] says so, and callers must not present
    /// results from this constructor as neural guidance.
    pub fn new(rules: RuleSet, verifier: Verifier) -> Self {
        Self::with_config(rules, verifier, MCTSConfig::default())
    }

    /// Create with custom configuration and an untrained policy network.
    pub fn with_config(rules: RuleSet, verifier: Verifier, config: MCTSConfig) -> Self {
        let vocabulary = ActionVocabulary::from_rule_set(&rules);
        let policy = PolicyNetwork::untrained_for(vocabulary.clone(), candle_core::Device::Cpu)
            .expect("failed to create policy network");
        Self {
            rules,
            verifier,
            policy,
            vocabulary,
            config,
        }
    }

    /// Use a specific policy network.
    ///
    /// The network's vocabulary must match this solver's rule set, otherwise its columns do
    /// not stand for these rules.
    pub fn with_policy(mut self, policy: PolicyNetwork) -> Result<Self, mm_rules::ActionError> {
        self.vocabulary.check_digest(policy.vocabulary().digest())?;
        self.policy = policy;
        Ok(self)
    }

    /// Where the policy weights came from.
    pub fn provenance(&self) -> &ModelProvenance {
        self.policy.provenance()
    }

    /// The action vocabulary this search reads priors through.
    pub fn vocabulary(&self) -> &ActionVocabulary {
        &self.vocabulary
    }

    /// Search for a solution, returning `Some` only if the goal was reached.
    pub fn search<F>(&self, start: Expr, goal: F) -> Option<Solution>
    where
        F: Fn(&Expr) -> bool,
    {
        let outcome = self.search_best_effort(start, goal);
        outcome.reached_goal.then_some(outcome.solution)
    }

    /// Search and return the best path found, whether or not it reaches the goal.
    pub fn search_best_effort<F>(&self, start: Expr, goal: F) -> SearchOutcome
    where
        F: Fn(&Expr) -> bool,
    {
        if goal(&start) {
            return SearchOutcome {
                solution: Solution::assess(start.clone(), start, vec![]),
                reached_goal: true,
                nodes_expanded: 1,
            };
        }

        let mut root = MCTSNode::new(start.clone(), 1.0);
        for _ in 0..self.config.simulations {
            self.simulate(&mut root, &goal, 0);
        }

        let total_nodes = count_nodes(&root);
        let mut outcome = self.extract(&root, &start, &goal);
        outcome.nodes_expanded = total_nodes;
        outcome
    }

    /// Run one MCTS simulation (SELECT, EXPAND, EVALUATE, BACKUP).
    fn simulate<F>(&self, node: &mut MCTSNode, goal: &F, depth: usize) -> f64
    where
        F: Fn(&Expr) -> bool,
    {
        // Terminal: record the visit here too, otherwise a goal node stays at zero visits and
        // loses the most-visited comparison during extraction.
        if goal(&node.state) {
            node.back_up(1.0);
            return 1.0;
        }

        if depth >= self.config.max_depth {
            let value = self.evaluate(&node.state);
            node.back_up(value);
            return value;
        }

        if !node.expanded {
            self.expand(node);
            node.expanded = true;

            let value = self.evaluate(&node.state);
            node.back_up(value);
            return value;
        }

        if node.children.is_empty() {
            // No valid moves: this is a leaf of the reachable space.
            let value = self.evaluate(&node.state);
            node.back_up(value);
            return value;
        }

        let best = self.select_child(node);
        let value = self.simulate(&mut node.children[best], goal, depth + 1);
        node.back_up(value);
        value
    }

    /// Renormalize child priors over the legal action set.
    ///
    /// The policy head is a softmax over the whole action vocabulary (573 classes), but only a
    /// few of those actions are legal at any given state -- the rule must be offered by the
    /// guardrail, apply to this expression, change it, and survive verification. PUCT requires
    /// the priors it consumes to be a distribution over *legal* actions, which is why
    /// AlphaZero-style implementations mask illegal moves and renormalize.
    ///
    /// Without this step the exploration term's absolute scale is arbitrary: if the network
    /// puts most of its mass on actions that are not legal here, every legal child's raw
    /// probability sits far below the nominal `1/573`, which changes how the exploration term
    /// trades off against the value term as visits accumulate. This is worth doing on general
    /// PUCT-hygiene grounds.
    ///
    /// It is NOT, on its own, a fix for a measured trained-vs-uniform regression this project
    /// hit (a trained policy scoring 72/200 on its own training families while matching uniform
    /// at 200/200 on families it had no signal for). That was checked directly, not assumed:
    /// renormalizing is a positive linear rescaling of every legal child's prior by the same
    /// constant, which cannot change which child has the highest prior -- confirmed by rerunning
    /// the exact benchmark before and after this function existed and getting byte-identical
    /// results (272/400 both times). The actual cause, found by printing the policy's raw
    /// output at a failing root state, is a genuine miscalibration: on
    /// `1 * (1 * (3*x + 6*x) + 0)`, the two legal moves are `identity_mul_one` (the correct
    /// first step) and `distribute`; the network assigns `distribute` roughly 5,400x more raw
    /// probability, and the same mis-ranking reproduces at deeper instances of the same family
    /// (`identity_mul_one` still loses by ~2,500x at depth 6 and ~500x at depth 8). The
    /// network's training data (`mm_brain::data`) teaches this rule only on flat, single-level
    /// expressions (`x * 1`, `1 * x`); it was never shown the rule firing on a `1 * (...)`
    /// wrapping a compound sub-expression, and confidently prefers a different, also-trained
    /// rule that superficially matches the nested shape instead. A confidently wrong prior
    /// starves the correct branch of simulation budget regardless of normalization -- MCTS
    /// spends its budget on the branch the policy insists is best, not on the one UCB's
    /// exploration term would eventually reach if given enough visits.
    ///
    /// Degenerate case this function does still handle: if every legal child has a vanishing
    /// prior (the policy's mass is entirely on illegal actions), fall back to a uniform
    /// distribution over the children rather than dividing by ~0.
    fn renormalize_priors(children: &mut [Box<MCTSNode>]) {
        if children.is_empty() {
            return;
        }
        let total: f64 = children.iter().map(|c| c.prior).sum();
        let uniform = 1.0 / children.len() as f64;
        if total.is_finite() && total > 1e-12 {
            for child in children.iter_mut() {
                child.prior /= total;
            }
        } else {
            for child in children.iter_mut() {
                child.prior = uniform;
            }
        }
    }

    /// Expand a node by adding one child per verified rule application.
    ///
    /// Uses the BOINK guardrail to filter rules by domain and features first.
    fn expand(&self, node: &mut MCTSNode) {
        let ctx = RuleContext::default();
        let profile = mm_boink::analyze(&node.state);
        let valid_rules = mm_boink::filter_rules(self.rules.all(), &profile);

        // Priors are read through the action vocabulary. A rule the vocabulary does not know
        // cannot be given a meaningful prior, so it falls back to a uniform value; that is
        // recorded here rather than silently applied to almost every rule as it once was.
        let uniform = 1.0 / self.vocabulary.len().max(1) as f32;
        let priors = self.learned_priors(&node.state);

        for rule in valid_rules {
            // `verify_step` rejects anything the rule does not claim to handle, so skipping
            // it here changes nothing except the work done.
            if !rule.can_apply(&node.state, &ctx) {
                continue;
            }

            let applications = rule.apply(&node.state, &ctx);

            for (application, app) in applications.into_iter().enumerate() {
                // A rule that returns its input is a self-loop, not an action.
                if app.result == node.state {
                    continue;
                }

                let verify = self
                    .verifier
                    .verify_step(&node.state, &app.result, rule, &ctx);
                if !verify.is_valid() {
                    continue;
                }

                let prior = priors
                    .as_ref()
                    .and_then(|p| self.vocabulary.prior_for_rule(p, rule.id).ok())
                    .unwrap_or(uniform);

                node.children.push(Box::new(MCTSNode::with_action(
                    app.result,
                    prior as f64,
                    ActionKey {
                        rule_id: rule.id,
                        application,
                    },
                    rule.name,
                    app.justification,
                    verify.evidence(),
                )));
            }
        }

        Self::renormalize_priors(&mut node.children);
    }

    /// Select the index of the best child by UCB, breaking ties towards the lower index.
    fn select_child(&self, node: &MCTSNode) -> usize {
        let mut best_score = f64::NEG_INFINITY;
        let mut best = 0;

        for (index, child) in node.children.iter().enumerate() {
            let score = child.ucb_score(node.visits, self.config.exploration_weight);
            if score > best_score {
                best_score = score;
                best = index;
            }
        }

        best
    }

    /// Policy priors, or `None` when there is no trained model to ask.
    ///
    /// An untrained head holds randomly initialised weights, so its output is noise that
    /// differs between processes. Consulting it would make the search both unreproducible and
    /// falsely "neural-guided", so the untrained case is answered with uniform priors instead.
    fn learned_priors(&self, state: &Expr) -> Option<Vec<f32>> {
        if !self.policy.provenance().is_trained() {
            return None;
        }
        self.policy.rule_priors(state).ok()
    }

    /// Evaluate a state using the value head.
    ///
    /// Untrained weights give no information about a state, so they are not consulted; every
    /// leaf scores 0.0 and selection is driven by priors and visit counts alone.
    fn evaluate(&self, state: &Expr) -> f64 {
        if !self.policy.provenance().is_trained() {
            return 0.0;
        }
        self.policy.get_value(state).unwrap_or(0.0) as f64
    }

    /// Extract the best path from the tree.
    ///
    /// A goal path always wins over a non-goal path, whatever the visit counts say. Among
    /// goal paths the shortest is taken; among non-goal paths the most-visited child is
    /// followed, with ties broken towards the lower index.
    fn extract<F>(&self, root: &MCTSNode, start: &Expr, goal: &F) -> SearchOutcome
    where
        F: Fn(&Expr) -> bool,
    {
        if let Some(steps) = self.find_goal_path(root, goal) {
            let result = steps
                .last()
                .map(|s: &Step| s.after.clone())
                .unwrap_or_else(|| start.clone());
            return SearchOutcome {
                solution: Solution::assess(start.clone(), result, steps),
                reached_goal: true,
                // Overwritten by the caller (`search_best_effort`), which knows the actual
                // tree size; `extract` only sees `root` by reference and has no reason to
                // recompute a count it did not need for path extraction itself.
                nodes_expanded: 0,
            };
        }

        // No goal anywhere in the tree: report the most-visited path as an explicitly
        // incomplete result rather than as a solution.
        let mut steps: Vec<Step> = Vec::new();
        let mut current = root;
        let mut prev = start.clone();

        while !current.children.is_empty() {
            let mut best = 0;
            for (index, child) in current.children.iter().enumerate() {
                if child.visits > current.children[best].visits {
                    best = index;
                }
            }
            let child = &current.children[best];
            if child.visits == 0 {
                break;
            }
            if let Some(step) = child.to_step(prev.clone()) {
                steps.push(step);
            }
            prev = child.state.clone();
            current = child;
        }

        let solution = Solution::assess_at_most(
            start.clone(),
            prev,
            steps,
            VerificationStatus::Partial {
                reason: "search did not reach the goal; this is a partial path".to_string(),
            },
        );
        SearchOutcome {
            solution,
            reached_goal: false,
            nodes_expanded: 0, // overwritten by the caller; see the note on the goal-path arm
        }
    }

    /// Depth-first search for the shortest recorded path to a goal node.
    fn find_goal_path<F>(&self, root: &MCTSNode, goal: &F) -> Option<Vec<Step>>
    where
        F: Fn(&Expr) -> bool,
    {
        // Breadth-first over the tree so the first goal found is the shortest path to one.
        let mut frontier: Vec<(&MCTSNode, Vec<Step>)> = vec![(root, Vec::new())];

        while !frontier.is_empty() {
            let mut next: Vec<(&MCTSNode, Vec<Step>)> = Vec::new();

            for (node, path) in frontier {
                for child in &node.children {
                    let mut child_path = path.clone();
                    if let Some(step) = child.to_step(node.state.clone()) {
                        child_path.push(step);
                    }
                    if goal(&child.state) {
                        return Some(child_path);
                    }
                    next.push((child, child_path));
                }
            }

            frontier = next;
        }

        None
    }

    /// Simplify an expression, chaining searches until no further progress is possible.
    ///
    /// Every recorded step carries the verifier's evidence, and the post-processing phases
    /// record their own steps, so the returned [`Solution`] replays from `expr` to the
    /// reported result and its status reflects what was actually checked.
    pub fn simplify(&self, expr: Expr) -> Solution {
        let ctx = RuleContext::default();
        let mut current = expr.clone();
        let mut steps: Vec<Step> = Vec::new();

        // Track visited states so a rule pair such as distribute/factor_common cannot loop.
        let mut seen: HashSet<Expr> = HashSet::new();
        seen.insert(current.clone());

        for _ in 0..self.config.max_simplify_iterations {
            // Nothing applies here, so neither the search nor the fallback can move. Checking
            // first avoids running a full simulation budget against a dead end.
            if !self.has_applicable_rule(&current, &ctx) {
                break;
            }

            let complexity = current.complexity();
            let goal = |candidate: &Expr| self.is_iteration_goal(candidate, complexity);

            let progressed = match self.search(current.clone(), goal) {
                Some(solution) if !solution.steps.is_empty() => {
                    let next = solution.result.clone();
                    if seen.contains(&next) {
                        break;
                    }
                    seen.insert(next.clone());
                    steps.extend(solution.steps);
                    current = next;
                    true
                }
                // The search found nothing, or found the goal without moving. Fall back to the
                // first verified rule application that reaches an unseen state.
                _ => match self.first_verified_step(&current, &ctx, &seen) {
                    Some(step) => {
                        let next = step.after.clone();
                        seen.insert(next.clone());
                        steps.push(step);
                        current = next;
                        true
                    }
                    None => false,
                },
            };

            if !progressed {
                break;
            }
        }

        // Post-processing. Each phase records a step describing what it changed, so a change
        // it cannot justify shows up in the status instead of vanishing.
        self.apply_subterm_rules(&mut current, &mut steps);
        self.apply_simplification_rules(&mut current, &mut steps);
        self.apply_constant_folding(&mut current, &mut steps);

        Solution::assess(expr, current, steps)
    }

    /// Whether a candidate counts as progress for one `simplify` iteration.
    ///
    /// Deliberately not "any structural change": that made every single rule application a
    /// goal, so the tree search never looked more than one move ahead and the procedural
    /// fallbacks did most of the work.
    ///
    /// A state with no applicable rules is not treated as a goal here. It is a dead end, and
    /// the search will simply fail to find a goal, which the caller already handles. Testing
    /// for it would mean scanning the whole registry at every visited node.
    fn is_iteration_goal(&self, candidate: &Expr, start_complexity: usize) -> bool {
        if let Expr::Equation { lhs, .. } = candidate {
            if matches!(lhs.as_ref(), Expr::Var(_)) {
                return true;
            }
        }

        candidate.complexity() < start_complexity
    }

    /// Whether any guardrail-permitted rule applies to this expression.
    fn has_applicable_rule(&self, expr: &Expr, ctx: &RuleContext) -> bool {
        let profile = mm_boink::analyze(expr);
        mm_boink::filter_rules(self.rules.all(), &profile)
            .into_iter()
            .any(|rule| rule.can_apply(expr, ctx))
    }

    /// First verified rule application that leads to an unseen state.
    fn first_verified_step(
        &self,
        current: &Expr,
        ctx: &RuleContext,
        seen: &HashSet<Expr>,
    ) -> Option<Step> {
        let profile = mm_boink::analyze(current);
        for rule in mm_boink::filter_rules(self.rules.all(), &profile) {
            if !rule.can_apply(current, ctx) {
                continue;
            }
            for app in rule.apply(current, ctx) {
                if app.result == *current || seen.contains(&app.result) {
                    continue;
                }
                let verify = self.verifier.verify_step(current, &app.result, rule, ctx);
                if !verify.is_valid() {
                    continue;
                }
                return Some(Step::rule(
                    current.clone(),
                    app.result,
                    rule.id,
                    rule.name,
                    app.justification,
                    verify.evidence(),
                ));
            }
        }
        None
    }

    /// Apply rules to sub-expressions, recording one step for the whole phase.
    ///
    /// Each individual sub-rewrite is verified against the sub-expression it touches; the
    /// step for the whole expression is justified by replacing a sub-term with a verified
    /// equal, and carries the weakest evidence any of those rewrites produced.
    fn apply_subterm_rules(&self, current: &mut Expr, steps: &mut Vec<Step>) {
        let mut evidence: Vec<StepEvidence> = Vec::new();
        let rewritten = self.rewrite_subterms(current, &mut evidence);

        if rewritten == *current {
            return;
        }

        steps.push(Step::normalization(
            current.clone(),
            rewritten.clone(),
            "subterm_rules",
            format!(
                "Applied {} verified rule(s) to sub-expressions",
                evidence.len()
            ),
            weakest_evidence(&evidence),
        ));
        *current = rewritten;
    }

    /// Repeatedly apply non-expanding simplification rules at the root until stable.
    fn apply_simplification_rules(&self, current: &mut Expr, steps: &mut Vec<Step>) {
        let ctx = RuleContext::default();
        let before = current.clone();
        let mut evidence: Vec<StepEvidence> = Vec::new();

        for _ in 0..10 {
            let applicable = self.rules.applicable(current, &ctx);
            // Skip expansion rules so distribute cannot undo collect_like_terms.
            let Some(rule) = applicable
                .iter()
                .find(|r| r.category != RuleCategory::Expansion)
            else {
                break;
            };

            let Some(app) = rule.apply(current, &ctx).into_iter().next() else {
                break;
            };
            if app.result.complexity() > current.complexity() || app.result == *current {
                break;
            }

            let verify = self.verifier.verify_step(current, &app.result, rule, &ctx);
            if !verify.is_valid() {
                break;
            }

            evidence.push(verify.evidence());
            *current = app.result;
        }

        if evidence.is_empty() {
            return;
        }

        steps.push(Step::normalization(
            before,
            current.clone(),
            "simplification_rules",
            format!("Applied {} verified simplification rule(s)", evidence.len()),
            weakest_evidence(&evidence),
        ));
    }

    /// Fold constants in the final result, recording the change as a checked step.
    ///
    /// The fold is an arithmetic normalisation rather than a registry rule, so it is checked
    /// by an independent equivalence check. If that check cannot establish equivalence, the
    /// step is still recorded — as unchecked — and the solution's status says so rather than
    /// the change happening invisibly.
    fn apply_constant_folding(&self, current: &mut Expr, steps: &mut Vec<Step>) {
        let folded = self.try_const_fold(current);
        if folded == *current {
            return;
        }

        let verify = self.verifier.verify_equivalence(current, &folded);
        steps.push(Step::normalization(
            current.clone(),
            folded.clone(),
            "const_fold",
            "Evaluated constant sub-expressions".to_string(),
            verify.evidence(),
        ));
        *current = folded;
    }

    /// Rewrite sub-terms with verified rule applications, innermost first.
    fn rewrite_subterms(&self, expr: &Expr, evidence: &mut Vec<StepEvidence>) -> Expr {
        let ctx = RuleContext::default();

        let rebuilt = match expr {
            Expr::Add(a, b) => Expr::Add(
                Box::new(self.rewrite_subterms(a, evidence)),
                Box::new(self.rewrite_subterms(b, evidence)),
            ),
            Expr::Sub(a, b) => Expr::Sub(
                Box::new(self.rewrite_subterms(a, evidence)),
                Box::new(self.rewrite_subterms(b, evidence)),
            ),
            Expr::Mul(a, b) => Expr::Mul(
                Box::new(self.rewrite_subterms(a, evidence)),
                Box::new(self.rewrite_subterms(b, evidence)),
            ),
            Expr::Div(a, b) => Expr::Div(
                Box::new(self.rewrite_subterms(a, evidence)),
                Box::new(self.rewrite_subterms(b, evidence)),
            ),
            Expr::Neg(a) => Expr::Neg(Box::new(self.rewrite_subterms(a, evidence))),
            other => other.clone(),
        };

        // Then try to rewrite this node itself. Derivative and integral nodes are the reason
        // this phase exists: the root-level search leaves them untouched when they sit inside
        // an addition or product.
        if matches!(
            rebuilt,
            Expr::Derivative { .. } | Expr::Integral { .. } | Expr::Add(_, _) | Expr::Sub(_, _)
        ) {
            if let Some((result, ev)) = self.first_verified_application(&rebuilt, &ctx) {
                evidence.push(ev);
                // The result may itself contain a reducible sub-term.
                return self.rewrite_subterms(&result, evidence);
            }
        }

        rebuilt
    }

    /// First verified application of any applicable rule at this node.
    fn first_verified_application(
        &self,
        expr: &Expr,
        ctx: &RuleContext,
    ) -> Option<(Expr, StepEvidence)> {
        for rule in self.rules.applicable(expr, ctx) {
            for app in rule.apply(expr, ctx) {
                if app.result == *expr {
                    continue;
                }
                let verify = self.verifier.verify_step(expr, &app.result, rule, ctx);
                if verify.is_valid() {
                    return Some((app.result, verify.evidence()));
                }
            }
        }
        None
    }

    /// Decompose an integral of a sum, solve each term, and recombine.
    ///
    /// The recombination joins traces that were produced for different expressions, so the
    /// result does not replay from the original problem and [`Solution::assess`] reports it as
    /// unverified. That is accurate: nothing here checks the recombined answer.
    pub fn progressive_solve(&self, expr: Expr) -> Solution {
        if let Expr::Integral { expr: inner, var } = &expr {
            if matches!(inner.as_ref(), Expr::Add(_, _) | Expr::Sub(_, _)) {
                let terms = mm_rules::decompose_additive(inner);

                if terms.len() > 1 {
                    // Solve the easiest terms first.
                    let mut scored: Vec<_> = terms
                        .iter()
                        .map(|t| (t.clone(), mm_rules::solvability_score(t)))
                        .collect();
                    scored
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                    let mut all_steps = Vec::new();
                    let mut partial_results = Vec::new();

                    for (term, _score) in &scored {
                        let term_integral = Expr::Integral {
                            expr: Box::new(term.clone()),
                            var: *var,
                        };
                        let term_solution = self.simplify(term_integral);
                        all_steps.extend(term_solution.steps);
                        partial_results.push(term_solution.result);
                    }

                    let combined = partial_results
                        .into_iter()
                        .reduce(|acc, r| Expr::Add(Box::new(acc), Box::new(r)))
                        .unwrap_or(Expr::int(0));

                    return Solution::assess(expr, combined, all_steps);
                }
            }
        }

        self.simplify(expr)
    }

    /// Try to constant fold an expression if all parts are constants.
    fn try_const_fold(&self, expr: &Expr) -> Expr {
        // Recursively try to fold sub-expressions
        match expr {
            Expr::Add(a, b) => {
                let a_folded = self.try_const_fold(a);
                let b_folded = self.try_const_fold(b);
                // Constant fold
                if let (Expr::Const(ra), Expr::Const(rb)) = (&a_folded, &b_folded) {
                    return Expr::Const(*ra + *rb);
                }
                // x + 0 = x
                if let Expr::Const(r) = &b_folded {
                    if r.is_zero() {
                        return a_folded;
                    }
                }
                if let Expr::Const(r) = &a_folded {
                    if r.is_zero() {
                        return b_folded;
                    }
                }
                Expr::Add(Box::new(a_folded), Box::new(b_folded))
            }
            Expr::Sub(a, b) => {
                let a_folded = self.try_const_fold(a);
                let b_folded = self.try_const_fold(b);
                if let (Expr::Const(ra), Expr::Const(rb)) = (&a_folded, &b_folded) {
                    return Expr::Const(*ra - *rb);
                }
                // x - 0 = x
                if let Expr::Const(r) = &b_folded {
                    if r.is_zero() {
                        return a_folded;
                    }
                }
                Expr::Sub(Box::new(a_folded), Box::new(b_folded))
            }
            Expr::Mul(a, b) => {
                let a_folded = self.try_const_fold(a);
                let b_folded = self.try_const_fold(b);
                if let (Expr::Const(ra), Expr::Const(rb)) = (&a_folded, &b_folded) {
                    return Expr::Const(*ra * *rb);
                }
                // x * 1 = x
                if let Expr::Const(r) = &b_folded {
                    if r.numer() == 1 && r.denom() == 1 {
                        return a_folded;
                    }
                    if r.is_zero() {
                        return Expr::int(0);
                    }
                }
                if let Expr::Const(r) = &a_folded {
                    if r.numer() == 1 && r.denom() == 1 {
                        return b_folded;
                    }
                    if r.is_zero() {
                        return Expr::int(0);
                    }
                }
                Expr::Mul(Box::new(a_folded), Box::new(b_folded))
            }
            Expr::Div(a, b) => {
                let a_folded = self.try_const_fold(a);
                let b_folded = self.try_const_fold(b);
                if let (Expr::Const(ra), Expr::Const(rb)) = (&a_folded, &b_folded) {
                    if !rb.is_zero() {
                        return Expr::Const(*ra / *rb);
                    }
                }
                Expr::Div(Box::new(a_folded), Box::new(b_folded))
            }
            Expr::Pow(base, exp) => {
                let base_folded = self.try_const_fold(base);
                let exp_folded = self.try_const_fold(exp);
                // x^1 = x
                if let Expr::Const(r) = &exp_folded {
                    if r.numer() == 1 && r.denom() == 1 {
                        return base_folded;
                    }
                    // x^0 = 1
                    if r.is_zero() {
                        return Expr::int(1);
                    }
                }
                // Constant folding: a^n when both are constants and n is small integer
                if let (Expr::Const(base_r), Expr::Const(exp_r)) = (&base_folded, &exp_folded) {
                    if exp_r.denom() == 1 && exp_r.numer() >= 0 && exp_r.numer() <= 30 {
                        let n = exp_r.numer() as u32;
                        // Compute base^n using repeated multiplication
                        let mut result = Rational::from_integer(1);
                        for _ in 0..n {
                            result = result * *base_r;
                        }
                        return Expr::Const(result);
                    }
                }
                Expr::Pow(Box::new(base_folded), Box::new(exp_folded))
            }
            Expr::Neg(inner) => {
                let folded = self.try_const_fold(inner);
                if let Expr::Const(r) = &folded {
                    return Expr::Const(-*r);
                }
                Expr::Neg(Box::new(folded))
            }
            Expr::Equation { lhs, rhs } => {
                // Fold both sides of equation
                let lhs_folded = self.try_const_fold(lhs);
                let rhs_folded = self.try_const_fold(rhs);
                Expr::Equation {
                    lhs: Box::new(lhs_folded),
                    rhs: Box::new(rhs_folded),
                }
            }

            // Phase 1: Add constant folding for number theory operations
            Expr::GCD(a, b) => {
                let a_folded = self.try_const_fold(a);
                let b_folded = self.try_const_fold(b);
                if let (Expr::Const(ra), Expr::Const(rb)) = (&a_folded, &b_folded) {
                    let a_val = ra.numer().abs();
                    let b_val = rb.numer().abs();
                    if ra.denom() == 1 && rb.denom() == 1 {
                        return Expr::Const(Rational::from_integer(gcd(a_val, b_val)));
                    }
                }
                Expr::GCD(Box::new(a_folded), Box::new(b_folded))
            }
            Expr::LCM(a, b) => {
                let a_folded = self.try_const_fold(a);
                let b_folded = self.try_const_fold(b);
                if let (Expr::Const(ra), Expr::Const(rb)) = (&a_folded, &b_folded) {
                    let a_val = ra.numer().abs();
                    let b_val = rb.numer().abs();
                    if ra.denom() == 1 && rb.denom() == 1 && a_val > 0 && b_val > 0 {
                        let g = gcd(a_val, b_val);
                        return Expr::Const(Rational::from_integer(a_val / g * b_val));
                    }
                }
                Expr::LCM(Box::new(a_folded), Box::new(b_folded))
            }
            Expr::Mod(a, b) => {
                let a_folded = self.try_const_fold(a);
                let b_folded = self.try_const_fold(b);
                if let (Expr::Const(ra), Expr::Const(rb)) = (&a_folded, &b_folded) {
                    if ra.denom() == 1 && rb.denom() == 1 && rb.numer() != 0 {
                        let a_val = ra.numer();
                        let b_val = rb.numer();
                        let result = ((a_val % b_val) + b_val) % b_val; // Ensure positive
                        return Expr::Const(Rational::from_integer(result));
                    }
                }
                Expr::Mod(Box::new(a_folded), Box::new(b_folded))
            }
            Expr::Factorial(n) => {
                let n_folded = self.try_const_fold(n);
                if let Expr::Const(r) = &n_folded {
                    if r.denom() == 1 && r.numer() >= 0 && r.numer() <= 20 {
                        let n_val = r.numer() as u64;
                        let result = factorial(n_val);
                        return Expr::Const(Rational::from_integer(result as i64));
                    }
                }
                Expr::Factorial(Box::new(n_folded))
            }
            Expr::Binomial(n, k) => {
                let n_folded = self.try_const_fold(n);
                let k_folded = self.try_const_fold(k);
                if let (Expr::Const(rn), Expr::Const(rk)) = (&n_folded, &k_folded) {
                    if rn.denom() == 1 && rk.denom() == 1 {
                        let n_val = rn.numer();
                        let k_val = rk.numer();
                        if n_val >= 0 && k_val >= 0 && k_val <= n_val && n_val <= 20 {
                            let n_u = n_val as u64;
                            let k_u = k_val as u64;
                            let result = factorial(n_u) / (factorial(k_u) * factorial(n_u - k_u));
                            return Expr::Const(Rational::from_integer(result as i64));
                        }
                    }
                }
                Expr::Binomial(Box::new(n_folded), Box::new(k_folded))
            }
            Expr::Floor(e) => {
                let folded = self.try_const_fold(e);
                if let Expr::Const(r) = &folded {
                    // Floor of a rational: numer / denom (integer division towards -∞)
                    let n = r.numer();
                    let d = r.denom();
                    let result = if n >= 0 { n / d } else { (n - d + 1) / d };
                    return Expr::Const(Rational::from_integer(result));
                }
                Expr::Floor(Box::new(folded))
            }
            Expr::Ceiling(e) => {
                let folded = self.try_const_fold(e);
                if let Expr::Const(r) = &folded {
                    let n = r.numer();
                    let d = r.denom();
                    let result = if n >= 0 { (n + d - 1) / d } else { n / d };
                    return Expr::Const(Rational::from_integer(result));
                }
                Expr::Ceiling(Box::new(folded))
            }
            Expr::Sqrt(e) => {
                let folded = self.try_const_fold(e);
                // Check for perfect squares
                if let Expr::Const(r) = &folded {
                    if r.denom() == 1 && r.numer() >= 0 {
                        let n = r.numer() as u64;
                        let sqrt_n = (n as f64).sqrt() as u64;
                        if sqrt_n * sqrt_n == n {
                            return Expr::Const(Rational::from_integer(sqrt_n as i64));
                        }
                    }
                }
                Expr::Sqrt(Box::new(folded))
            }
            Expr::Abs(e) => {
                let folded = self.try_const_fold(e);
                if let Expr::Const(r) = &folded {
                    return Expr::Const(r.abs());
                }
                Expr::Abs(Box::new(folded))
            }

            _ => expr.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_rules::rule::{standard_rules, Rule, RuleApplication, RuleSet};
    use mm_rules::{Domain, Feature, RuleCategory};

    fn small_config() -> MCTSConfig {
        MCTSConfig {
            simulations: 20,
            ..Default::default()
        }
    }

    fn var() -> (mm_core::SymbolTable, Expr) {
        let mut symbols = mm_core::SymbolTable::new();
        let x = symbols.intern("x");
        (symbols, Expr::Var(x))
    }

    fn add_zero(e: &Expr) -> Expr {
        Expr::Add(Box::new(e.clone()), Box::new(Expr::int(0)))
    }

    fn mul_one(e: &Expr) -> Expr {
        Expr::Mul(Box::new(e.clone()), Box::new(Expr::int(1)))
    }

    fn test_rule(
        id: u32,
        name: &'static str,
        is_applicable: fn(&Expr, &RuleContext) -> bool,
        apply: fn(&Expr, &RuleContext) -> Vec<RuleApplication>,
    ) -> Rule {
        Rule {
            id: RuleId(id),
            name,
            category: RuleCategory::Simplification,
            description: "test rule",
            domains: &[] as &[Domain],
            requires: &[] as &[Feature],
            is_applicable,
            apply,
            reversible: false,
            cost: 1,
        }
    }

    /// One rule with two distinct, equivalence-preserving applications on a bare variable.
    fn two_application_rule() -> Rule {
        test_rule(
            1,
            "expand_two_ways",
            |e, _| matches!(e, Expr::Var(_)),
            |e, _| {
                if !matches!(e, Expr::Var(_)) {
                    return vec![];
                }
                vec![
                    RuleApplication {
                        result: add_zero(e),
                        justification: "x -> x + 0".to_string(),
                    },
                    RuleApplication {
                        result: mul_one(e),
                        justification: "x -> x * 1".to_string(),
                    },
                ]
            },
        )
    }

    #[test]
    fn node_starts_unvisited() {
        let node = MCTSNode::new(Expr::int(0), 0.5);
        assert_eq!(node.visits, 0);
        assert_eq!(node.value(), 0.0);
    }

    #[test]
    fn ucb_score_is_positive_for_a_visited_node() {
        let mut node = MCTSNode::new(Expr::int(0), 0.5);
        node.visits = 10;
        node.value_sum = 5.0;
        assert!(node.ucb_score(100, 1.41) > 0.0);
    }

    #[test]
    fn one_rule_with_two_applications_produces_two_children() {
        let mut rules = RuleSet::new();
        rules.try_add("test", two_application_rule()).unwrap();
        let mcts = NeuralMCTS::with_config(rules, Verifier::new(), small_config());

        let (_symbols, x) = var();
        let mut root = MCTSNode::new(x.clone(), 1.0);
        mcts.expand(&mut root);

        assert_eq!(
            root.children.len(),
            2,
            "the second application must not overwrite the first"
        );
        let states: Vec<&Expr> = root.children.iter().map(|c| &c.state).collect();
        assert!(states.contains(&&add_zero(&x)));
        assert!(states.contains(&&mul_one(&x)));

        let actions: Vec<ActionKey> = root.children.iter().filter_map(|c| c.action).collect();
        assert_eq!(actions[0].rule_id, actions[1].rule_id);
        assert_ne!(
            actions[0].application, actions[1].application,
            "applications of one rule must be distinguishable"
        );
    }

    #[test]
    fn two_different_rules_both_keep_their_children() {
        // Before the registry enforced unique identifiers these two could have shared one,
        // and the child map keyed by identifier would have kept only the second.
        let mut rules = RuleSet::new();
        rules
            .try_add(
                "test",
                test_rule(
                    1,
                    "to_add_zero",
                    |e, _| matches!(e, Expr::Var(_)),
                    |e, _| {
                        vec![RuleApplication {
                            result: add_zero(e),
                            justification: "x -> x + 0".to_string(),
                        }]
                    },
                ),
            )
            .unwrap();
        rules
            .try_add(
                "test",
                test_rule(
                    2,
                    "to_mul_one",
                    |e, _| matches!(e, Expr::Var(_)),
                    |e, _| {
                        vec![RuleApplication {
                            result: mul_one(e),
                            justification: "x -> x * 1".to_string(),
                        }]
                    },
                ),
            )
            .unwrap();

        let mcts = NeuralMCTS::with_config(rules, Verifier::new(), small_config());
        let (_symbols, x) = var();
        let mut root = MCTSNode::new(x, 1.0);
        mcts.expand(&mut root);

        assert_eq!(root.children.len(), 2);
        let names: Vec<&str> = root.children.iter().filter_map(|c| c.rule_name).collect();
        assert!(names.contains(&"to_add_zero"));
        assert!(names.contains(&"to_mul_one"));
    }

    #[test]
    fn reaching_the_goal_backs_up_through_the_terminal_node() {
        let mut rules = RuleSet::new();
        rules.try_add("test", two_application_rule()).unwrap();
        let mcts = NeuralMCTS::with_config(rules, Verifier::new(), small_config());

        let (_symbols, x) = var();
        let target = mul_one(&x);
        let mut root = MCTSNode::new(x, 1.0);
        let goal = |e: &Expr| *e == target;

        mcts.expand(&mut root);
        root.expanded = true;

        // Drive a simulation straight at the goal child, rather than relying on selection to
        // pick it: the invariant under test is the terminal backup, not the choice.
        let index = root
            .children
            .iter()
            .position(|c| goal(&c.state))
            .expect("goal child must exist");
        let value = mcts.simulate(&mut root.children[index], &goal, 0);

        assert_eq!(value, 1.0, "reaching the goal must be worth 1.0");
        let goal_child = &root.children[index];
        assert_eq!(
            goal_child.visits, 1,
            "a goal node must record its own visit"
        );
        assert!(
            goal_child.value() > 0.0,
            "a goal node must back up its value"
        );
    }

    #[test]
    fn a_goal_is_extracted_even_when_it_is_not_the_most_visited_child() {
        let mut rules = RuleSet::new();
        rules.try_add("test", two_application_rule()).unwrap();
        let mcts = NeuralMCTS::with_config(rules, Verifier::new(), small_config());

        let (_symbols, x) = var();
        let target = mul_one(&x);
        let decoy = add_zero(&x);
        let mut root = MCTSNode::new(x.clone(), 1.0);
        mcts.expand(&mut root);
        root.expanded = true;
        root.visits = 5;

        // Give the non-goal child all the visits and leave the goal child at zero.
        for child in root.children.iter_mut() {
            if child.state == decoy {
                child.visits = 100;
                child.value_sum = 100.0;
            }
        }

        let goal = |e: &Expr| *e == target;
        let outcome = mcts.extract(&root, &x, &goal);

        assert!(outcome.reached_goal);
        assert_eq!(outcome.solution.result, target);
    }

    #[test]
    fn a_search_without_a_goal_returns_none_not_a_partial_solution() {
        let mut rules = RuleSet::new();
        rules.try_add("test", two_application_rule()).unwrap();
        let mcts = NeuralMCTS::with_config(rules, Verifier::new(), small_config());

        let (_symbols, x) = var();
        let unreachable = |e: &Expr| *e == Expr::int(4242);

        assert!(mcts.search(x.clone(), unreachable).is_none());

        let outcome = mcts.search_best_effort(x, unreachable);
        assert!(!outcome.reached_goal);
        assert!(
            !outcome.solution.is_fully_verified(),
            "a partial path must not be reported as verified"
        );
    }

    #[test]
    fn search_results_are_deterministic() {
        // Two independently constructed solvers must agree. They hold separately initialised
        // networks, so this fails if the search reads randomly initialised weights.
        let build = || {
            let mut rules = RuleSet::new();
            rules.try_add("test", two_application_rule()).unwrap();
            NeuralMCTS::with_config(rules, Verifier::new(), small_config())
        };

        let (_symbols, x) = var();
        let target = mul_one(&x);
        let goal = |e: &Expr| *e == target;

        let a = build().search(x.clone(), goal).expect("goal is reachable");
        let b = build().search(x, goal).expect("goal is reachable");

        assert_eq!(a.result, b.result);
        assert_eq!(a.num_steps(), b.num_steps());
        assert_eq!(
            a.steps.iter().map(|s| s.rule_id.0).collect::<Vec<_>>(),
            b.steps.iter().map(|s| s.rule_id.0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn simplify_is_deterministic_across_solver_instances() {
        let expr = Expr::Mul(
            Box::new(Expr::Add(Box::new(Expr::int(2)), Box::new(Expr::int(3)))),
            Box::new(Expr::Add(Box::new(Expr::int(4)), Box::new(Expr::int(5)))),
        );

        let a = NeuralMCTS::with_config(standard_rules(), Verifier::new(), small_config())
            .simplify(expr.clone());
        let b = NeuralMCTS::with_config(standard_rules(), Verifier::new(), small_config())
            .simplify(expr);

        assert_eq!(a.result, b.result);
        assert_eq!(a.num_steps(), b.num_steps());
        assert_eq!(a.status, b.status);
    }

    #[test]
    fn max_depth_is_respected() {
        // A rule that always produces a new, larger expression. Without the depth bound the
        // simulation would recurse until the stack ran out.
        let mut rules = RuleSet::new();
        rules
            .try_add(
                "test",
                test_rule(
                    1,
                    "grow",
                    |_, _| true,
                    |e, _| {
                        vec![RuleApplication {
                            result: add_zero(e),
                            justification: "wrap in + 0".to_string(),
                        }]
                    },
                ),
            )
            .unwrap();

        let mcts = NeuralMCTS::with_config(
            rules,
            Verifier::new(),
            MCTSConfig {
                simulations: 8,
                max_depth: 3,
                ..Default::default()
            },
        );

        let outcome = mcts.search_best_effort(Expr::int(1), |e: &Expr| *e == Expr::int(4242));
        assert!(!outcome.reached_goal);
        assert!(
            outcome.solution.num_steps() <= 3,
            "path length must respect max_depth, got {}",
            outcome.solution.num_steps()
        );
    }

    #[test]
    fn a_cycle_does_not_stall_simplify() {
        // Two rules that undo each other. `simplify` must terminate and must not drift.
        let mut rules = RuleSet::new();
        rules
            .try_add(
                "test",
                test_rule(
                    1,
                    "add_zero",
                    |e, _| matches!(e, Expr::Var(_)),
                    |e, _| {
                        vec![RuleApplication {
                            result: add_zero(e),
                            justification: "x -> x + 0".to_string(),
                        }]
                    },
                ),
            )
            .unwrap();
        rules
            .try_add(
                "test",
                test_rule(
                    2,
                    "drop_zero",
                    |e, _| {
                        matches!(e, Expr::Add(_, b)
                            if matches!(b.as_ref(), Expr::Const(c) if c.is_zero()))
                    },
                    |e, _| match e {
                        Expr::Add(a, _) => vec![RuleApplication {
                            result: (**a).clone(),
                            justification: "x + 0 -> x".to_string(),
                        }],
                        _ => vec![],
                    },
                ),
            )
            .unwrap();

        let mcts = NeuralMCTS::with_config(rules, Verifier::new(), small_config());
        let (_symbols, x) = var();
        let solution = mcts.simplify(x.clone());
        assert_eq!(solution.result, x, "the cycle must not change the answer");
    }

    // `renormalize_priors` is justified purely on PUCT semantics -- see its doc comment for
    // why it is NOT presented as a fix for any measured search-quality result. These tests
    // check only the three properties a legal-action prior distribution must have; none of
    // them touch policy weights, search outcomes, or the E1 benchmark.
    mod renormalize_priors_semantics {
        use super::*;

        fn child(prior: f64) -> Box<MCTSNode> {
            Box::new(MCTSNode::new(Expr::int(0), prior))
        }

        #[test]
        fn priors_sum_to_one_after_renormalizing() {
            let mut children = vec![child(0.003), child(0.0000006), child(0.001)];
            NeuralMCTS::renormalize_priors(&mut children);
            let sum: f64 = children.iter().map(|c| c.prior).sum();
            assert!(
                (sum - 1.0).abs() < 1e-9,
                "renormalized priors must sum to 1.0 over the legal action set, got {sum}"
            );
        }

        #[test]
        fn relative_order_is_unchanged_by_renormalizing() {
            // This is the property that matters for reading the E1 result correctly: renormalizing
            // is a positive linear rescaling of every child by the same constant, so it can change
            // magnitude but never which child has the largest (or smallest) prior. A test that
            // showed otherwise would mean this function does something other than renormalize.
            let raw = [0.00403985_f64, 0.00000075, 0.002];
            let mut children: Vec<Box<MCTSNode>> = raw.iter().map(|&p| child(p)).collect();
            NeuralMCTS::renormalize_priors(&mut children);

            let mut raw_order: Vec<usize> = (0..raw.len()).collect();
            raw_order.sort_by(|&a, &b| raw[b].partial_cmp(&raw[a]).unwrap());

            let mut renorm_order: Vec<usize> = (0..children.len()).collect();
            renorm_order
                .sort_by(|&a, &b| children[b].prior.partial_cmp(&children[a].prior).unwrap());

            assert_eq!(
                raw_order, renorm_order,
                "renormalizing must not change the ranking of children by prior"
            );
        }

        #[test]
        fn a_vanishing_total_falls_back_to_uniform_rather_than_dividing_by_zero() {
            let mut children = vec![child(1e-20), child(1e-20), child(1e-20), child(1e-20)];
            NeuralMCTS::renormalize_priors(&mut children);
            for c in &children {
                assert!(
                    (c.prior - 0.25).abs() < 1e-9,
                    "with a vanishing total, every child should fall back to 1/n, got {}",
                    c.prior
                );
                assert!(c.prior.is_finite());
            }
        }

        #[test]
        fn an_empty_child_list_is_a_no_op() {
            let mut children: Vec<Box<MCTSNode>> = Vec::new();
            NeuralMCTS::renormalize_priors(&mut children); // must not panic
            assert!(children.is_empty());
        }
    }

    #[test]
    fn nodes_expanded_is_one_when_the_start_already_satisfies_the_goal() {
        // No simulation runs at all here (`search_best_effort` returns before building a
        // tree), so the only node that can exist is the one the caller started with.
        let mcts = NeuralMCTS::with_config(standard_rules(), Verifier::new(), small_config());
        let (_symbols, x) = var();
        let outcome = mcts.search_best_effort(x.clone(), |e| *e == x);
        assert!(outcome.reached_goal);
        assert_eq!(outcome.nodes_expanded, 1);
    }

    #[test]
    fn nodes_expanded_reflects_the_actual_tree_the_search_built() {
        // x + 0 reaches a trivial goal in one step, so the tree is at minimum {root, one
        // child reached by identity_add_zero}. This does not pin an exact count -- MCTS may
        // expand siblings too -- only the floor a tree that did any work at all must clear,
        // and that it is not the degenerate `1` from the zero-simulation branch above.
        let mcts = NeuralMCTS::with_config(standard_rules(), Verifier::new(), small_config());
        let (_symbols, x) = var();
        let start = add_zero(&x);
        let outcome = mcts.search_best_effort(start, |e| *e == x);
        assert!(outcome.reached_goal, "x + 0 -> x must be reachable");
        assert!(
            outcome.nodes_expanded >= 2,
            "a search that expanded at least once must report more than the trivial 1-node case, \
             got {}",
            outcome.nodes_expanded
        );
    }

    #[test]
    fn the_default_policy_is_reported_as_untrained() {
        let mcts = NeuralMCTS::new(standard_rules(), Verifier::new());
        assert!(!mcts.provenance().is_trained());
        assert_eq!(mcts.vocabulary().len(), standard_rules().len());
    }

    #[test]
    fn every_registered_rule_has_a_readable_prior() {
        // The old code indexed a 25-column tensor with a rule identifier of up to five
        // digits, so nearly every rule silently fell back to a constant 0.01.
        let mcts = NeuralMCTS::new(standard_rules(), Verifier::new());
        let priors = mcts.policy.rule_priors(&Expr::int(5)).unwrap();
        assert_eq!(priors.len(), mcts.vocabulary().len());

        for rule in mcts.rules.all() {
            mcts.vocabulary()
                .prior_for_rule(&priors, rule.id)
                .unwrap_or_else(|e| panic!("rule {} has no prior: {e}", rule.name));
        }
    }

    #[test]
    fn an_untrained_policy_is_not_consulted() {
        // Random weights are not guidance. Reading them would also make the search differ
        // between processes, since candle seeds its initialisation from the OS.
        let mcts = NeuralMCTS::new(standard_rules(), Verifier::new());
        assert!(!mcts.provenance().is_trained());
        assert!(mcts.learned_priors(&Expr::int(5)).is_none());
        assert_eq!(mcts.evaluate(&Expr::int(5)), 0.0);
    }

    #[test]
    fn simplify_produces_a_replayable_trace() {
        let mcts = NeuralMCTS::with_config(standard_rules(), Verifier::new(), small_config());
        let expr = Expr::Add(Box::new(Expr::int(2)), Box::new(Expr::int(3)));
        let solution = mcts.simplify(expr.clone());

        assert_eq!(solution.result.canonicalize(), Expr::int(5));
        assert_eq!(
            crate::assess_trace(&solution.problem, &solution.result, &solution.steps),
            solution.status,
            "the reported status must be the one the trace implies"
        );
        assert!(
            solution.status.replays(),
            "recorded steps must lead from the input to the reported result, got {}",
            solution.status
        );
    }

    #[test]
    fn no_untraced_jump_to_the_final_answer() {
        let mcts = NeuralMCTS::with_config(standard_rules(), Verifier::new(), small_config());
        let expr = Expr::Add(Box::new(Expr::int(2)), Box::new(Expr::int(3)));
        let solution = mcts.simplify(expr);

        assert!(!solution.steps.is_empty());
        assert_eq!(solution.steps.last().unwrap().after, solution.result);
    }
}
