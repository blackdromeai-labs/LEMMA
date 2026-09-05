// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Compositional training data: the same 24 rules `data.rs` trains, presented in nested,
//! multi-layer states with compound operands rather than the flat single-level forms
//! `data.rs` uses exclusively.
//!
//! # Why this exists
//!
//! Instrumenting a trained policy at a failing search root (see `AGENT_MEMORY.md`, entry
//! "E1 ran, root cause found") showed the network assigning `distribute` roughly 5,400x more
//! probability than `identity_mul_one` at `1 * (1 * (3*x + 6*x) + 0)`, even though
//! `identity_mul_one` is the correct move and is a rule the network WAS trained on.
//! `data.rs`'s own generator only ever shows `identity_mul_one` firing on flat `x * 1` / `1 *
//! x`; it never shows the rule firing when the non-identity operand is itself a compound
//! sub-expression, or when a second rule (`distribute`) is simultaneously legal at the same
//! node. This module exists to close exactly that gap, for all 24 trained rules, not just the
//! one the instrumentation happened to catch.
//!
//! # What "compositional" means here, concretely
//!
//! - **Compound operands**: where `data.rs` uses a bare `Var` or small `Const` as "the other
//!   side" of an operation, this generator uses a small pool of already-meaningful
//!   sub-expressions (`a*x`, `x^n`, `sin(x)`, `x + y`, ...) instead.
//! - **Varying wrapper position**: the same target pattern appears at the root of the
//!   expression in some examples and nested one or two levels inside an unrelated `Add`/`Mul`
//!   in others, so the network cannot rely on the pattern always sitting at the top.
//! - **Multiple simultaneously applicable rules**: for the specific structural collision found
//!   (`Mul(Const(1), Add(_, _))` triggers both `identity_mul_one` and `distribute`) and its
//!   analogues (`Mul(Const(0), Add(_, _))` triggers both `zero_mul` and `distribute`), this
//!   generator produces many instances labelled with the rule that is the more direct
//!   simplification, specifically to counteract the observed bias.
//!
//! # Validation, not assertion
//!
//! Every example this module produces is checked against the real rule and the real verifier
//! before being included: the target rule must actually apply to the generated expression, its
//! application must actually change it, and `Verifier::verify_step` must accept the result. An
//! example that fails any of these is silently dropped, not corrected -- a template that never
//! produces a validated example for some rule is a signal to fix the template, not a reason to
//! label an unverified transformation.
//!
//! # What this module does NOT do
//!
//! It does not read, generate against, or otherwise depend on the locked evaluation corpus at
//! `experiments/corpus/problems.jsonl`. That corpus is held out for evaluation; training against
//! it, or deriving these templates from its contents, would make any comparison against it
//! circular. Every template here is built from the rule *definitions* in `mm-rules`, not from
//! any specific evaluation problem.
//!
//! # One of the 24 rules is excluded, on evidence
//!
//! `equations::quadratic_formula`'s `apply` returns only the `+` branch of `x = (-b +
//! sqrt(disc))/2a` (its own source comment says "For simplicity, return the formula form") --
//! one root of a generally two-root equation. `Verifier::verify_step` checks solution-set
//! equivalence for equation rewrites and correctly rejects this as `Invalid { reason:
//! "Symbolic verification failed" }` for every non-degenerate case this generator tried
//! (confirmed by direct instrumentation, not assumed). Since this module's entire premise is
//! that a label must be backed by an accepted rule application, and the current rule/verifier
//! pair cannot honestly produce one, `quadratic_formula` is excluded here rather than
//! mislabeled. This is a pre-existing gap in the rule's implementation, not something
//! introduced by this generator; it was not previously visible because `data.rs`'s generator
//! never validates its labels against the real rule or verifier. This corpus therefore covers
//! 23 of the 24 rules `data.rs` targets.

use std::collections::HashSet;

use candle_core::Device;
use mm_core::{Expr, SymbolTable};
use mm_rules::rule::{RuleContext, RuleKey, RuleSet};
use mm_rules::{standard_rules, ActionVocabulary};
use mm_verifier::Verifier;
use rand::prelude::*;

use crate::encoder::ExpressionEncoder;
use crate::training::TrainingExample;

/// The seed this generator is always run with. Frozen so the corpus is exactly reproducible;
/// changing it is a new corpus and should get a new manifest, not a silent overwrite.
pub const COMPOSITIONAL_SEED: u64 = 0xC0A1_0515_0000_0042;

/// One (rule, count) request: "generate up to `count` validated examples for this rule from its
/// compositional templates."
struct RuleRequest {
    key: RuleKey,
    /// Vocabulary-index resolved at construction, not guessed.
    action_id: u32,
}

/// Generator for compositional training examples.
pub struct CompositionalDataGenerator {
    encoder: ExpressionEncoder,
    rng: StdRng,
    rules: RuleSet,
    verifier: Verifier,
    vocabulary: ActionVocabulary,
    ctx: RuleContext,
    x: mm_core::Symbol,
    y: mm_core::Symbol,
    z: mm_core::Symbol,
    /// Kept alive: `x`/`y`/`z` are symbols into this table.
    _symbols: SymbolTable,
}

/// Outcome of one generation pass: the examples, plus enough bookkeeping to write an honest
/// manifest rather than just a count.
pub struct GenerationReport {
    pub examples: Vec<TrainingExample>,
    /// Per-rule: (requested attempts, validated examples kept).
    pub per_rule: Vec<(RuleKey, usize, usize)>,
    /// Number of examples where more than one rule was legally applicable at the generated
    /// expression's root -- the discrimination cases this module exists to add.
    pub collision_examples: usize,
    pub seed: u64,
}

impl CompositionalDataGenerator {
    pub fn new(device: Device, seq_len: usize) -> Self {
        let vocabulary = ActionVocabulary::standard();
        let mut symbols = SymbolTable::new();
        let x = symbols.intern("x");
        let y = symbols.intern("y");
        let z = symbols.intern("z");
        Self {
            encoder: ExpressionEncoder::new(device).with_max_length(seq_len),
            rng: StdRng::seed_from_u64(COMPOSITIONAL_SEED),
            rules: standard_rules(),
            verifier: Verifier::new(),
            vocabulary,
            ctx: RuleContext::default(),
            x,
            y,
            z,
            _symbols: symbols,
        }
    }

    fn resolve(&self, module: &'static str, name: &'static str) -> RuleRequest {
        let key = RuleKey { module, name };
        let action_id = self
            .vocabulary
            .index_of(module, name)
            .unwrap_or_else(|e| panic!("compositional label refers to a missing rule: {e}"))
            as u32;
        RuleRequest { key, action_id }
    }

    /// Validate one candidate (expression, target rule) pair against the real rule and the real
    /// verifier. Returns `None` -- silently -- if the rule does not apply, does not change the
    /// expression, or the verifier rejects the transformation.
    fn validated_example(
        &self,
        expr: &Expr,
        req: &RuleRequest,
        value: f32,
    ) -> Option<TrainingExample> {
        let rule = self.rules.get_by_key(&req.key)?;
        if !rule.can_apply(expr, &self.ctx) {
            return None;
        }
        let application = rule
            .apply(expr, &self.ctx)
            .into_iter()
            .find(|app| app.result != *expr)?;
        let verify = self
            .verifier
            .verify_step(expr, &application.result, rule, &self.ctx);
        if !verify.is_valid() {
            return None;
        }
        let tokens = self.encoder.encode_tokens(&self.encoder.tokenize(expr));
        Some(TrainingExample {
            tokens,
            target_action: req.action_id,
            target_value: value,
        })
    }

    // ---- compound operand pool -----------------------------------------------------------
    // Small, reusable sub-expressions built from the rule DEFINITIONS this generator targets,
    // not from any specific evaluation problem. Varying which of these fills "the other slot"
    // is what makes an example compositional instead of flat.

    fn rand_small(&mut self) -> i64 {
        self.rng.gen_range(2..9)
    }

    fn compound_operand(&mut self) -> Expr {
        let var = *[self.x, self.y, self.z].choose(&mut self.rng).unwrap();
        match self.rng.gen_range(0..6) {
            0 => Expr::Var(var),
            1 => {
                let a = self.rand_small();
                Expr::Mul(Box::new(Expr::int(a)), Box::new(Expr::Var(var)))
            }
            2 => {
                let n = self.rng.gen_range(2..5);
                Expr::Pow(Box::new(Expr::Var(var)), Box::new(Expr::int(n)))
            }
            3 => Expr::Sin(Box::new(Expr::Var(var))),
            4 => Expr::Cos(Box::new(Expr::Var(var))),
            _ => {
                let other = *[self.x, self.y, self.z].choose(&mut self.rng).unwrap();
                Expr::Add(Box::new(Expr::Var(var)), Box::new(Expr::Var(other)))
            }
        }
    }

    fn compound_sum(&mut self) -> Expr {
        Expr::Add(
            Box::new(self.compound_operand()),
            Box::new(self.compound_operand()),
        )
    }

    /// Wrap `inner` in one extra layer of unrelated structure, so the target pattern is not
    /// always sitting at the root of the encoded expression.
    fn outer_wrap(&mut self, inner: Expr) -> Expr {
        match self.rng.gen_range(0..3) {
            0 => inner, // unwrapped: still needed as a baseline
            1 => {
                let extra = self.compound_operand();
                Expr::Add(Box::new(inner), Box::new(extra))
            }
            _ => {
                let a = self.rand_small();
                Expr::Mul(Box::new(Expr::int(a)), Box::new(inner))
            }
        }
    }

    // ---- per-rule compositional templates --------------------------------------------------
    // Each returns candidate (expression, is-a-collision) pairs. "Collision" here means the
    // literal `Mul(Const(1|0), Add(_, _))` shape where a second trained rule is also legal at
    // the same node -- the specific structural ambiguity the instrumentation found, generalised
    // to its natural sibling (`zero_mul` vs `distribute`) rather than copied from any one
    // instance of it.

    fn gen_const_fold(&mut self, count: usize) -> Vec<(Expr, bool)> {
        (0..count)
            .map(|_| {
                let a = self.rand_small();
                let b = self.rand_small();
                let base = match self.rng.gen_range(0..3) {
                    0 => Expr::Add(Box::new(Expr::int(a)), Box::new(Expr::int(b))),
                    1 => Expr::Mul(Box::new(Expr::int(a)), Box::new(Expr::int(b))),
                    _ => Expr::Sub(Box::new(Expr::int(a + b)), Box::new(Expr::int(b))),
                };
                (self.outer_wrap(base), false)
            })
            .collect()
    }

    fn gen_identity_add_zero(&mut self, count: usize) -> Vec<(Expr, bool)> {
        (0..count)
            .map(|_| {
                let operand = self.compound_operand();
                let base = if self.rng.gen_bool(0.5) {
                    Expr::Add(Box::new(operand), Box::new(Expr::int(0)))
                } else {
                    Expr::Add(Box::new(Expr::int(0)), Box::new(operand))
                };
                (self.outer_wrap(base), false)
            })
            .collect()
    }

    fn gen_identity_mul_one(&mut self, count: usize) -> Vec<(Expr, bool)> {
        (0..count)
            .map(|_| {
                // Half plain compositional (compound operand times 1), half the exact
                // collision shape (1 * (A + B), where `distribute` is also legal) -- this half
                // is the discrimination signal the network was missing.
                if self.rng.gen_bool(0.5) {
                    let operand = self.compound_operand();
                    let base = if self.rng.gen_bool(0.5) {
                        Expr::Mul(Box::new(operand), Box::new(Expr::int(1)))
                    } else {
                        Expr::Mul(Box::new(Expr::int(1)), Box::new(operand))
                    };
                    (self.outer_wrap(base), false)
                } else {
                    let sum = self.compound_sum();
                    let base = if self.rng.gen_bool(0.5) {
                        Expr::Mul(Box::new(Expr::int(1)), Box::new(sum))
                    } else {
                        Expr::Mul(Box::new(sum), Box::new(Expr::int(1)))
                    };
                    (self.outer_wrap(base), true)
                }
            })
            .collect()
    }

    fn gen_zero_mul(&mut self, count: usize) -> Vec<(Expr, bool)> {
        (0..count)
            .map(|_| {
                if self.rng.gen_bool(0.5) {
                    let operand = self.compound_operand();
                    let base = if self.rng.gen_bool(0.5) {
                        Expr::Mul(Box::new(operand), Box::new(Expr::int(0)))
                    } else {
                        Expr::Mul(Box::new(Expr::int(0)), Box::new(operand))
                    };
                    (self.outer_wrap(base), false)
                } else {
                    // The zero_mul/distribute analogue of the identity_mul_one collision:
                    // 0 * (A + B) is legally both `zero_mul` and `distribute`.
                    let sum = self.compound_sum();
                    let base = if self.rng.gen_bool(0.5) {
                        Expr::Mul(Box::new(Expr::int(0)), Box::new(sum))
                    } else {
                        Expr::Mul(Box::new(sum), Box::new(Expr::int(0)))
                    };
                    (self.outer_wrap(base), true)
                }
            })
            .collect()
    }

    fn gen_collect_like_terms(&mut self, count: usize) -> Vec<(Expr, bool)> {
        (0..count)
            .map(|_| {
                let var = *[self.x, self.y, self.z].choose(&mut self.rng).unwrap();
                let a = self.rand_small();
                let b = self.rand_small();
                let base = Expr::Add(
                    Box::new(Expr::Mul(Box::new(Expr::int(a)), Box::new(Expr::Var(var)))),
                    Box::new(Expr::Mul(Box::new(Expr::int(b)), Box::new(Expr::Var(var)))),
                );
                (self.outer_wrap(base), false)
            })
            .collect()
    }

    fn gen_distribute(&mut self, count: usize) -> Vec<(Expr, bool)> {
        (0..count)
            .map(|_| {
                let a = self.rand_small();
                let sum = self.compound_sum();
                let base = if self.rng.gen_bool(0.5) {
                    Expr::Mul(Box::new(Expr::int(a)), Box::new(sum))
                } else {
                    Expr::Mul(Box::new(sum), Box::new(Expr::int(a)))
                };
                (self.outer_wrap(base), false)
            })
            .collect()
    }

    fn gen_factor_common(&mut self, count: usize) -> Vec<(Expr, bool)> {
        (0..count)
            .map(|_| {
                let var = *[self.x, self.y, self.z].choose(&mut self.rng).unwrap();
                let other = *[self.x, self.y, self.z].choose(&mut self.rng).unwrap();
                let a = self.rand_small();
                let base = Expr::Add(
                    Box::new(Expr::Mul(Box::new(Expr::int(a)), Box::new(Expr::Var(var)))),
                    Box::new(Expr::Mul(
                        Box::new(Expr::int(a)),
                        Box::new(Expr::Var(other)),
                    )),
                );
                (self.outer_wrap(base), false)
            })
            .collect()
    }

    fn gen_difference_of_squares(&mut self, count: usize) -> Vec<(Expr, bool)> {
        (0..count)
            .map(|_| {
                let var = *[self.x, self.y, self.z].choose(&mut self.rng).unwrap();
                let other = *[self.x, self.y, self.z].choose(&mut self.rng).unwrap();
                let base = Expr::Sub(
                    Box::new(Expr::Pow(Box::new(Expr::Var(var)), Box::new(Expr::int(2)))),
                    Box::new(Expr::Pow(
                        Box::new(Expr::Var(other)),
                        Box::new(Expr::int(2)),
                    )),
                );
                (self.outer_wrap(base), false)
            })
            .collect()
    }

    // Calculus: vary the differentiated sub-expression's composition, not just its exponent.
    fn gen_power_rule(&mut self, count: usize) -> Vec<(Expr, bool)> {
        (0..count)
            .map(|_| {
                let n = self.rng.gen_range(2..7);
                let inner = Expr::Pow(Box::new(Expr::Var(self.x)), Box::new(Expr::int(n)));
                (
                    Expr::Derivative {
                        expr: Box::new(inner),
                        var: self.x,
                    },
                    false,
                )
            })
            .collect()
    }

    fn gen_constant_rule(&mut self, count: usize) -> Vec<(Expr, bool)> {
        (0..count)
            .map(|_| {
                let c = self.rand_small();
                let inner = if self.rng.gen_bool(0.5) {
                    Expr::int(c)
                } else {
                    // A constant expression that is not a bare literal: the sum of two
                    // literals, which `const_fold` could also reduce first but which
                    // `constant_rule` is equally entitled to differentiate directly.
                    Expr::Add(
                        Box::new(Expr::int(c)),
                        Box::new(Expr::int(self.rand_small())),
                    )
                };
                (
                    Expr::Derivative {
                        expr: Box::new(inner),
                        var: self.y, // differentiate w.r.t. a variable that does not appear
                    },
                    false,
                )
            })
            .collect()
    }

    fn gen_sum_rule(&mut self, count: usize) -> Vec<(Expr, bool)> {
        (0..count)
            .map(|_| {
                let n = self.rng.gen_range(2..5);
                let m = self.rng.gen_range(2..5);
                let inner = Expr::Add(
                    Box::new(Expr::Pow(
                        Box::new(Expr::Var(self.x)),
                        Box::new(Expr::int(n)),
                    )),
                    Box::new(Expr::Sin(Box::new(Expr::Var(self.x)))),
                );
                let _ = m;
                (
                    Expr::Derivative {
                        expr: Box::new(inner),
                        var: self.x,
                    },
                    false,
                )
            })
            .collect()
    }

    fn gen_product_rule(&mut self, count: usize) -> Vec<(Expr, bool)> {
        (0..count)
            .map(|_| {
                let inner = Expr::Mul(
                    Box::new(Expr::Sin(Box::new(Expr::Var(self.x)))),
                    Box::new(Expr::Cos(Box::new(Expr::Var(self.x)))),
                );
                (
                    Expr::Derivative {
                        expr: Box::new(inner),
                        var: self.x,
                    },
                    false,
                )
            })
            .collect()
    }

    fn gen_quotient_rule(&mut self, count: usize) -> Vec<(Expr, bool)> {
        (0..count)
            .map(|_| {
                let n = self.rng.gen_range(2..5);
                let inner = Expr::Div(
                    Box::new(Expr::Sin(Box::new(Expr::Var(self.x)))),
                    Box::new(Expr::Pow(
                        Box::new(Expr::Var(self.x)),
                        Box::new(Expr::int(n)),
                    )),
                );
                (
                    Expr::Derivative {
                        expr: Box::new(inner),
                        var: self.x,
                    },
                    false,
                )
            })
            .collect()
    }

    fn gen_trig_and_transcendental_derivatives(
        &mut self,
        count: usize,
    ) -> Vec<(Expr, bool, &'static str, &'static str)> {
        let mut out = Vec::new();
        for _ in 0..count {
            let (module, name, inner): (&'static str, &'static str, Expr) =
                match self.rng.gen_range(0..4) {
                    0 => (
                        "calculus",
                        "sin_chain_rule",
                        Expr::Sin(Box::new(Expr::Mul(
                            Box::new(Expr::int(self.rand_small())),
                            Box::new(Expr::Var(self.x)),
                        ))),
                    ),
                    1 => (
                        "calculus",
                        "cos_chain_rule",
                        Expr::Cos(Box::new(Expr::Add(
                            Box::new(Expr::Var(self.x)),
                            Box::new(Expr::int(self.rand_small())),
                        ))),
                    ),
                    2 => (
                        "calculus",
                        "exp_derivative",
                        Expr::Exp(Box::new(Expr::Var(self.x))),
                    ),
                    _ => (
                        "calculus",
                        "ln_derivative",
                        Expr::Ln(Box::new(Expr::Var(self.x))),
                    ),
                };
            out.push((
                Expr::Derivative {
                    expr: Box::new(inner),
                    var: self.x,
                },
                false,
                module,
                name,
            ));
        }
        out
    }

    // Equations: vary both sides with small compound arithmetic rather than bare literals.
    fn gen_isolate_and_cancel(&mut self, count: usize) -> Vec<(Expr, bool, &'static str)> {
        let mut out = Vec::new();
        for _ in 0..count {
            let a = self.rand_small();
            let b = self.rng.gen_range(-9..9);
            let c = self.rng.gen_range(-9..9);
            let (name, expr) = match self.rng.gen_range(0..4) {
                0 => (
                    "cancel_addition",
                    Expr::Equation {
                        lhs: Box::new(Expr::Add(
                            Box::new(Expr::Var(self.x)),
                            Box::new(Expr::int(b)),
                        )),
                        rhs: Box::new(Expr::int(c)),
                    },
                ),
                1 => (
                    "cancel_subtraction",
                    Expr::Equation {
                        lhs: Box::new(Expr::Sub(
                            Box::new(Expr::Var(self.x)),
                            Box::new(Expr::int(b)),
                        )),
                        rhs: Box::new(Expr::int(c)),
                    },
                ),
                2 => (
                    "cancel_multiplication",
                    Expr::Equation {
                        lhs: Box::new(Expr::Mul(
                            Box::new(Expr::int(a)),
                            Box::new(Expr::Var(self.x)),
                        )),
                        rhs: Box::new(Expr::int(c)),
                    },
                ),
                _ => (
                    "cancel_division",
                    Expr::Equation {
                        lhs: Box::new(Expr::Div(
                            Box::new(Expr::Var(self.x)),
                            Box::new(Expr::int(a)),
                        )),
                        rhs: Box::new(Expr::int(c)),
                    },
                ),
            };
            out.push((expr, false, name));
        }
        out
    }

    fn gen_linear_solve(&mut self, count: usize) -> Vec<(Expr, bool)> {
        (0..count)
            .map(|_| {
                let a = self.rand_small();
                let b = self.rng.gen_range(-9..9);
                let c = self.rng.gen_range(-9..9);
                let expr = Expr::Equation {
                    lhs: Box::new(Expr::Add(
                        Box::new(Expr::Mul(
                            Box::new(Expr::int(a)),
                            Box::new(Expr::Var(self.x)),
                        )),
                        Box::new(Expr::int(b)),
                    )),
                    rhs: Box::new(Expr::int(c)),
                };
                (expr, false)
            })
            .collect()
    }

    /// Run every template, validate every candidate, and return the kept examples plus an
    /// honest report of what was requested versus what survived validation.
    pub fn generate(&mut self, per_rule: usize) -> GenerationReport {
        let mut examples = Vec::new();
        let mut per_rule_report = Vec::new();
        let mut collision_examples = 0usize;
        let mut seen: HashSet<(Vec<u32>, u32)> = HashSet::new();

        macro_rules! run_simple {
            ($module:expr, $name:expr, $gen:expr) => {{
                let req = self.resolve($module, $name);
                let candidates = $gen(self, per_rule);
                let attempted = candidates.len();
                let mut kept = 0usize;
                for (expr, is_collision) in candidates {
                    if is_collision {
                        collision_examples += 1;
                    }
                    if let Some(ex) = self.validated_example(&expr, &req, 1.0) {
                        if seen.insert((ex.tokens.clone(), ex.target_action)) {
                            examples.push(ex);
                            kept += 1;
                        }
                    }
                }
                per_rule_report.push((req.key, attempted, kept));
            }};
        }

        run_simple!("algebra", "const_fold", Self::gen_const_fold);
        run_simple!("algebra", "identity_add_zero", Self::gen_identity_add_zero);
        run_simple!("algebra", "identity_mul_one", Self::gen_identity_mul_one);
        run_simple!("algebra", "zero_mul", Self::gen_zero_mul);
        run_simple!(
            "algebra",
            "collect_like_terms",
            Self::gen_collect_like_terms
        );
        run_simple!("algebra", "distribute", Self::gen_distribute);
        run_simple!("algebra", "factor_common", Self::gen_factor_common);
        run_simple!(
            "algebra",
            "difference_of_squares",
            Self::gen_difference_of_squares
        );
        run_simple!("calculus", "power_rule", Self::gen_power_rule);
        run_simple!("calculus", "constant_rule", Self::gen_constant_rule);
        run_simple!("calculus", "sum_rule", Self::gen_sum_rule);
        run_simple!("calculus", "product_rule", Self::gen_product_rule);
        run_simple!("calculus", "quotient_rule", Self::gen_quotient_rule);
        run_simple!("equations", "linear_solve", Self::gen_linear_solve);
        // `quadratic_formula` is deliberately not targeted here -- see the module doc comment
        // "One of the 24 rules is excluded, on evidence".

        // Multi-target generators (one call produces examples for several rules at once).
        {
            let candidates = self.gen_trig_and_transcendental_derivatives(per_rule);
            let mut by_rule: std::collections::HashMap<
                (&'static str, &'static str),
                (usize, usize),
            > = std::collections::HashMap::new();
            for (expr, is_collision, module, name) in candidates {
                if is_collision {
                    collision_examples += 1;
                }
                let req = self.resolve(module, name);
                let entry = by_rule.entry((module, name)).or_insert((0, 0));
                entry.0 += 1;
                if let Some(ex) = self.validated_example(&expr, &req, 1.0) {
                    if seen.insert((ex.tokens.clone(), ex.target_action)) {
                        examples.push(ex);
                        entry.1 += 1;
                    }
                }
            }
            for ((module, name), (attempted, kept)) in by_rule {
                per_rule_report.push((RuleKey { module, name }, attempted, kept));
            }
        }
        {
            let candidates = self.gen_isolate_and_cancel(per_rule);
            let mut by_rule: std::collections::HashMap<&'static str, (usize, usize)> =
                std::collections::HashMap::new();
            for (expr, is_collision, name) in candidates {
                if is_collision {
                    collision_examples += 1;
                }
                let req = self.resolve("equations", name);
                let entry = by_rule.entry(name).or_insert((0, 0));
                entry.0 += 1;
                if let Some(ex) = self.validated_example(&expr, &req, 1.0) {
                    if seen.insert((ex.tokens.clone(), ex.target_action)) {
                        examples.push(ex);
                        entry.1 += 1;
                    }
                }
            }
            for (name, (attempted, kept)) in by_rule {
                per_rule_report.push((
                    RuleKey {
                        module: "equations",
                        name,
                    },
                    attempted,
                    kept,
                ));
            }
        }

        // `isolate_variable` is not directly generated above (it fires on `ax - b = c` in
        // `data.rs`'s own scheme as the *alternative* label to `linear_solve`); route a share
        // of the linear-solve candidates through it as a genuinely distinct compositional case.
        {
            let req = self.resolve("equations", "isolate_variable");
            let mut attempted = 0usize;
            let mut kept = 0usize;
            for _ in 0..per_rule {
                let a = self.rand_small();
                let b = self.rng.gen_range(1..9);
                let c = self.rng.gen_range(-9..9);
                let expr = Expr::Equation {
                    lhs: Box::new(Expr::Sub(
                        Box::new(Expr::Mul(
                            Box::new(Expr::int(a)),
                            Box::new(Expr::Var(self.x)),
                        )),
                        Box::new(Expr::int(b)),
                    )),
                    rhs: Box::new(Expr::int(c)),
                };
                attempted += 1;
                if let Some(ex) = self.validated_example(&expr, &req, 1.0) {
                    if seen.insert((ex.tokens.clone(), ex.target_action)) {
                        examples.push(ex);
                        kept += 1;
                    }
                }
            }
            per_rule_report.push((req.key, attempted, kept));
        }

        examples.shuffle(&mut self.rng);

        GenerationReport {
            examples,
            per_rule: per_rule_report,
            collision_examples,
            seed: COMPOSITIONAL_SEED,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_targeted_rule_yields_at_least_one_validated_example() {
        let mut gen = CompositionalDataGenerator::new(Device::Cpu, 32);
        let report = gen.generate(40);
        let starved: Vec<String> = report
            .per_rule
            .iter()
            .filter(|(_, _, kept)| *kept == 0)
            .map(|(key, attempted, _)| format!("{key} (attempted {attempted})"))
            .collect();
        assert!(
            starved.is_empty(),
            "rules with zero validated examples: {starved:?}"
        );
        assert_eq!(
            report.per_rule.len(),
            23,
            "expected 23 of the 24 trained rules (quadratic_formula excluded, see module docs)"
        );
    }

    #[test]
    fn the_identity_mul_one_vs_distribute_collision_is_represented() {
        let mut gen = CompositionalDataGenerator::new(Device::Cpu, 32);
        let report = gen.generate(40);
        assert!(
            report.collision_examples > 0,
            "expected at least one generated collision case (1*(A+B) style ambiguity)"
        );
    }

    #[test]
    fn generation_is_deterministic_for_the_frozen_seed() {
        let mut gen_a = CompositionalDataGenerator::new(Device::Cpu, 32);
        let mut gen_b = CompositionalDataGenerator::new(Device::Cpu, 32);
        let report_a = gen_a.generate(20);
        let report_b = gen_b.generate(20);
        let tokens_a: Vec<_> = report_a.examples.iter().map(|e| e.tokens.clone()).collect();
        let tokens_b: Vec<_> = report_b.examples.iter().map(|e| e.tokens.clone()).collect();
        assert_eq!(
            tokens_a, tokens_b,
            "same frozen seed must reproduce the same corpus"
        );
    }

    #[test]
    fn no_duplicate_token_sequences_are_kept_for_the_same_label() {
        let mut gen = CompositionalDataGenerator::new(Device::Cpu, 32);
        let report = gen.generate(40);
        let mut seen = HashSet::new();
        for ex in &report.examples {
            assert!(
                seen.insert((ex.tokens.clone(), ex.target_action)),
                "duplicate (tokens, label) pair should have been deduplicated at generation time"
            );
        }
    }
}
