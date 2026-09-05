// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Author: Pushp Kharat

//! Synthetic training data generation.
//!
//! Labels are dense [`ActionVocabulary`] indices resolved from stable `module::name` keys, so
//! a training label and an inference column always refer to the same rule.
//!
//! Covers a small, hand-written slice of the registry:
//! - Algebra: constant folding, identities, distribution, factoring
//! - Calculus: power, sum, product, quotient, chain rules
//! - Trig: sin/cos derivatives
//! - Equations: linear, quadratic solving
//!
//! The remaining actions in the vocabulary get no synthetic examples at all; this data is not
//! coverage of the rule corpus.

use candle_core::Device;
use mm_core::{Expr, SymbolTable};
use mm_rules::ActionVocabulary;
use rand::prelude::*;

use crate::encoder::ExpressionEncoder;
use crate::training::TrainingExample;

/// Dense action indices for the rules the synthetic generator writes labels for.
///
/// Resolved from an [`ActionVocabulary`] by stable `module::name` key. The previous version
/// of this module hard-coded small integers that were meant to be rule identifiers but were
/// used as tensor columns; they were off by one for every algebra rule (class 0 was
/// documented as `const_fold`, whose identifier is 1) and matched nothing at all for the
/// rest of the registry.
struct ActionIds {
    const_fold: u32,
    identity_add_zero: u32,
    identity_mul_one: u32,
    zero_mul: u32,
    collect_like_terms: u32,
    distribute: u32,
    factor_common: u32,
    diff_of_squares: u32,
    power_rule: u32,
    constant_rule: u32,
    sum_rule: u32,
    product_rule: u32,
    quotient_rule: u32,
    sin_derivative: u32,
    cos_derivative: u32,
    exp_derivative: u32,
    ln_derivative: u32,
    isolate_variable: u32,
    cancel_addition: u32,
    cancel_subtraction: u32,
    cancel_multiplication: u32,
    cancel_division: u32,
    linear_solve: u32,
    quadratic_formula: u32,
    /// Reserved terminal class: "no rule applies here".
    no_op: u32,
}

impl ActionIds {
    fn resolve(vocab: &ActionVocabulary) -> Self {
        let idx = |module: &str, name: &str| -> u32 {
            vocab
                .index_of(module, name)
                .unwrap_or_else(|e| panic!("synthetic label refers to a missing rule: {e}"))
                as u32
        };

        Self {
            const_fold: idx("algebra", "const_fold"),
            identity_add_zero: idx("algebra", "identity_add_zero"),
            identity_mul_one: idx("algebra", "identity_mul_one"),
            zero_mul: idx("algebra", "zero_mul"),
            collect_like_terms: idx("algebra", "collect_like_terms"),
            distribute: idx("algebra", "distribute"),
            factor_common: idx("algebra", "factor_common"),
            diff_of_squares: idx("algebra", "difference_of_squares"),
            power_rule: idx("calculus", "power_rule"),
            constant_rule: idx("calculus", "constant_rule"),
            sum_rule: idx("calculus", "sum_rule"),
            product_rule: idx("calculus", "product_rule"),
            quotient_rule: idx("calculus", "quotient_rule"),
            sin_derivative: idx("calculus", "sin_chain_rule"),
            cos_derivative: idx("calculus", "cos_chain_rule"),
            exp_derivative: idx("calculus", "exp_derivative"),
            ln_derivative: idx("calculus", "ln_derivative"),
            isolate_variable: idx("equations", "isolate_variable"),
            cancel_addition: idx("equations", "cancel_addition"),
            cancel_subtraction: idx("equations", "cancel_subtraction"),
            cancel_multiplication: idx("equations", "cancel_multiplication"),
            cancel_division: idx("equations", "cancel_division"),
            linear_solve: idx("equations", "linear_solve"),
            quadratic_formula: idx("equations", "quadratic_formula"),
            no_op: vocab.len() as u32,
        }
    }
}

/// Generator for synthetic training data.
pub struct DataGenerator {
    encoder: ExpressionEncoder,
    rng: StdRng,
    x: mm_core::Symbol,
    y: mm_core::Symbol,
    z: mm_core::Symbol,
    vocabulary: ActionVocabulary,
    actions: ActionIds,
}

impl DataGenerator {
    /// Create a new data generator for the standard action vocabulary.
    pub fn new(device: Device) -> Self {
        Self::with_seed(device, 42)
    }

    /// Create with a specific random seed.
    pub fn with_seed(device: Device, seed: u64) -> Self {
        Self::with_vocabulary(device, seed, ActionVocabulary::standard())
    }

    /// Create with an explicit action vocabulary.
    pub fn with_vocabulary(device: Device, seed: u64, vocabulary: ActionVocabulary) -> Self {
        let mut symbols = SymbolTable::new();
        let x = symbols.intern("x");
        let y = symbols.intern("y");
        let z = symbols.intern("z");
        let actions = ActionIds::resolve(&vocabulary);

        Self {
            encoder: ExpressionEncoder::new(device),
            rng: StdRng::seed_from_u64(seed),
            x,
            y,
            z,
            vocabulary,
            actions,
        }
    }

    /// Set the encoder's padding width.
    ///
    /// Must match the `max_seq_len` of the network being trained: the encoder pads every
    /// example to this width and the network is built for exactly that width. The default (64)
    /// is far wider than these expressions need -- most tokenize to well under a dozen tokens
    /// -- and since attention cost grows with the square of the padded width, training at the
    /// default spends most of its time on padding.
    pub fn with_max_length(mut self, len: usize) -> Self {
        self.encoder = ExpressionEncoder::new(self.encoder.device().clone()).with_max_length(len);
        self
    }

    /// The action vocabulary the generated labels are expressed in.
    pub fn vocabulary(&self) -> &ActionVocabulary {
        &self.vocabulary
    }

    fn make_example(&self, expr: &Expr, action: u32, value: f32) -> TrainingExample {
        let tokens = self.encoder.encode_tokens(&self.encoder.tokenize(expr));
        TrainingExample {
            tokens,
            target_action: action,
            target_value: value,
        }
    }

    fn rand_small(&mut self) -> i64 {
        self.rng.gen_range(1..15)
    }

    fn rand_nonzero(&mut self) -> i64 {
        let v = self.rng.gen_range(1..20);
        if self.rng.gen_bool(0.5) {
            v
        } else {
            -v
        }
    }

    // =========================================================================
    // ALGEBRA RULES
    // =========================================================================

    /// Constant folding: 2 + 3 → 5, 4 * 5 → 20
    pub fn generate_constant_folding(&mut self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        for _ in 0..count {
            let a = self.rand_small();
            let b = self.rand_small();

            // Addition
            examples.push(self.make_example(
                &Expr::Add(Box::new(Expr::int(a)), Box::new(Expr::int(b))),
                self.actions.const_fold,
                1.0,
            ));

            // Subtraction
            examples.push(self.make_example(
                &Expr::Sub(Box::new(Expr::int(a + b)), Box::new(Expr::int(b))),
                self.actions.const_fold,
                1.0,
            ));

            // Multiplication
            examples.push(self.make_example(
                &Expr::Mul(Box::new(Expr::int(a)), Box::new(Expr::int(b))),
                self.actions.const_fold,
                1.0,
            ));

            // Division (avoid zero)
            if b != 0 {
                examples.push(self.make_example(
                    &Expr::Div(Box::new(Expr::int(a * b)), Box::new(Expr::int(b))),
                    self.actions.const_fold,
                    1.0,
                ));
            }

            // Power
            let exp = self.rng.gen_range(0..5);
            examples.push(self.make_example(
                &Expr::Pow(Box::new(Expr::int(2)), Box::new(Expr::int(exp))),
                self.actions.const_fold,
                1.0,
            ));
        }

        examples
    }

    /// Identity rules: x + 0 → x, x * 1 → x, x * 0 → 0
    pub fn generate_identity_rules(&mut self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();
        let vars = [self.x, self.y, self.z];

        for i in 0..count {
            let v = vars[i % 3];

            // x + 0 → x
            examples.push(self.make_example(
                &Expr::Add(Box::new(Expr::Var(v)), Box::new(Expr::int(0))),
                self.actions.identity_add_zero,
                1.0,
            ));

            // 0 + x → x
            examples.push(self.make_example(
                &Expr::Add(Box::new(Expr::int(0)), Box::new(Expr::Var(v))),
                self.actions.identity_add_zero,
                1.0,
            ));

            // x * 1 → x
            examples.push(self.make_example(
                &Expr::Mul(Box::new(Expr::Var(v)), Box::new(Expr::int(1))),
                self.actions.identity_mul_one,
                1.0,
            ));

            // 1 * x → x
            examples.push(self.make_example(
                &Expr::Mul(Box::new(Expr::int(1)), Box::new(Expr::Var(v))),
                self.actions.identity_mul_one,
                1.0,
            ));

            // x * 0 → 0
            examples.push(self.make_example(
                &Expr::Mul(Box::new(Expr::Var(v)), Box::new(Expr::int(0))),
                self.actions.zero_mul,
                1.0,
            ));

            // 0 * x → 0
            examples.push(self.make_example(
                &Expr::Mul(Box::new(Expr::int(0)), Box::new(Expr::Var(v))),
                self.actions.zero_mul,
                1.0,
            ));
        }

        examples
    }

    /// Distribution: a(b + c) → ab + ac
    pub fn generate_distribute(&mut self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        for _ in 0..count {
            let a = self.rand_small();

            // a * (x + y)
            examples.push(self.make_example(
                &Expr::Mul(
                    Box::new(Expr::int(a)),
                    Box::new(Expr::Add(
                        Box::new(Expr::Var(self.x)),
                        Box::new(Expr::Var(self.y)),
                    )),
                ),
                self.actions.distribute,
                1.0,
            ));

            // (x + y) * a
            examples.push(self.make_example(
                &Expr::Mul(
                    Box::new(Expr::Add(
                        Box::new(Expr::Var(self.x)),
                        Box::new(Expr::Var(self.y)),
                    )),
                    Box::new(Expr::int(a)),
                ),
                self.actions.distribute,
                1.0,
            ));

            // a * (x - y)
            examples.push(self.make_example(
                &Expr::Mul(
                    Box::new(Expr::int(a)),
                    Box::new(Expr::Add(
                        Box::new(Expr::Var(self.x)),
                        Box::new(Expr::Neg(Box::new(Expr::Var(self.y)))),
                    )),
                ),
                self.actions.distribute,
                1.0,
            ));
        }

        examples
    }

    /// Factor common: ab + ac → a(b + c)
    pub fn generate_factor_common(&mut self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        for _ in 0..count {
            let a = self.rand_small();

            // ax + ay (common factor a)
            examples.push(self.make_example(
                &Expr::Add(
                    Box::new(Expr::Mul(
                        Box::new(Expr::int(a)),
                        Box::new(Expr::Var(self.x)),
                    )),
                    Box::new(Expr::Mul(
                        Box::new(Expr::int(a)),
                        Box::new(Expr::Var(self.y)),
                    )),
                ),
                self.actions.factor_common,
                1.0,
            ));
        }

        examples
    }

    /// Difference of squares: a² - b² → (a+b)(a-b)
    pub fn generate_difference_of_squares(&mut self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        for _ in 0..count {
            // x² - y²
            examples.push(self.make_example(
                &Expr::Sub(
                    Box::new(Expr::Pow(
                        Box::new(Expr::Var(self.x)),
                        Box::new(Expr::int(2)),
                    )),
                    Box::new(Expr::Pow(
                        Box::new(Expr::Var(self.y)),
                        Box::new(Expr::int(2)),
                    )),
                ),
                self.actions.diff_of_squares,
                1.0,
            ));

            // x² - 4 (where 4 = 2²)
            let a = self.rand_small();
            examples.push(self.make_example(
                &Expr::Sub(
                    Box::new(Expr::Pow(
                        Box::new(Expr::Var(self.x)),
                        Box::new(Expr::int(2)),
                    )),
                    Box::new(Expr::Pow(Box::new(Expr::int(a)), Box::new(Expr::int(2)))),
                ),
                self.actions.diff_of_squares,
                1.0,
            ));
        }

        examples
    }

    /// Collect like terms: ax + bx → (a+b)x
    pub fn generate_collect_like_terms(&mut self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        for _ in 0..count {
            let a = self.rand_small();
            let b = self.rand_small();

            // ax + bx
            examples.push(self.make_example(
                &Expr::Add(
                    Box::new(Expr::Mul(
                        Box::new(Expr::int(a)),
                        Box::new(Expr::Var(self.x)),
                    )),
                    Box::new(Expr::Mul(
                        Box::new(Expr::int(b)),
                        Box::new(Expr::Var(self.x)),
                    )),
                ),
                self.actions.collect_like_terms,
                1.0,
            ));

            // x + x → 2x
            examples.push(self.make_example(
                &Expr::Add(Box::new(Expr::Var(self.x)), Box::new(Expr::Var(self.x))),
                self.actions.collect_like_terms,
                1.0,
            ));
        }

        examples
    }

    // =========================================================================
    // CALCULUS RULES
    // =========================================================================

    /// Power rule: d/dx(x^n) = n*x^(n-1)
    pub fn generate_power_rule(&mut self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        for i in 0..count {
            let n = (i % 10) as i64 + 1;

            // d/dx(x^n)
            examples.push(self.make_example(
                &Expr::Derivative {
                    expr: Box::new(Expr::Pow(
                        Box::new(Expr::Var(self.x)),
                        Box::new(Expr::int(n)),
                    )),
                    var: self.x,
                },
                self.actions.power_rule,
                1.0,
            ));

            // d/dx(x) = 1
            examples.push(self.make_example(
                &Expr::Derivative {
                    expr: Box::new(Expr::Var(self.x)),
                    var: self.x,
                },
                self.actions.power_rule,
                1.0,
            ));
        }

        examples
    }

    /// Constant rule: d/dx(c) = 0
    pub fn generate_constant_rule(&mut self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        for _ in 0..count {
            let c = self.rand_small();

            examples.push(self.make_example(
                &Expr::Derivative {
                    expr: Box::new(Expr::int(c)),
                    var: self.x,
                },
                self.actions.constant_rule,
                1.0,
            ));

            // d/dy(x) where y is different variable
            examples.push(self.make_example(
                &Expr::Derivative {
                    expr: Box::new(Expr::Var(self.x)),
                    var: self.y,
                },
                self.actions.constant_rule,
                1.0,
            ));
        }

        examples
    }

    /// Sum rule: d/dx(f + g) = f' + g'
    pub fn generate_sum_rule(&mut self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        for i in 0..count {
            let n = (i % 5) as i64 + 1;
            let m = (i % 4) as i64 + 2;

            // d/dx(x^n + x^m)
            examples.push(self.make_example(
                &Expr::Derivative {
                    expr: Box::new(Expr::Add(
                        Box::new(Expr::Pow(
                            Box::new(Expr::Var(self.x)),
                            Box::new(Expr::int(n)),
                        )),
                        Box::new(Expr::Pow(
                            Box::new(Expr::Var(self.x)),
                            Box::new(Expr::int(m)),
                        )),
                    )),
                    var: self.x,
                },
                self.actions.sum_rule,
                1.0,
            ));

            // d/dx(x + c)
            let c = self.rand_small();
            examples.push(self.make_example(
                &Expr::Derivative {
                    expr: Box::new(Expr::Add(
                        Box::new(Expr::Var(self.x)),
                        Box::new(Expr::int(c)),
                    )),
                    var: self.x,
                },
                self.actions.sum_rule,
                1.0,
            ));
        }

        examples
    }

    /// Product rule: d/dx(fg) = f'g + fg'
    pub fn generate_product_rule(&mut self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        for i in 0..count {
            let n = (i % 5) as i64 + 1;

            // d/dx(x * x^n) = d/dx(x^(n+1))
            examples.push(self.make_example(
                &Expr::Derivative {
                    expr: Box::new(Expr::Mul(
                        Box::new(Expr::Var(self.x)),
                        Box::new(Expr::Pow(
                            Box::new(Expr::Var(self.x)),
                            Box::new(Expr::int(n)),
                        )),
                    )),
                    var: self.x,
                },
                self.actions.product_rule,
                1.0,
            ));

            // d/dx(x * sin(x))
            examples.push(self.make_example(
                &Expr::Derivative {
                    expr: Box::new(Expr::Mul(
                        Box::new(Expr::Var(self.x)),
                        Box::new(Expr::Sin(Box::new(Expr::Var(self.x)))),
                    )),
                    var: self.x,
                },
                self.actions.product_rule,
                1.0,
            ));
        }

        examples
    }

    /// Quotient rule: d/dx(f/g) = (f'g - fg')/g²
    pub fn generate_quotient_rule(&mut self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        for i in 0..count {
            let n = (i % 5) as i64 + 2;

            // d/dx(x / x^n)
            examples.push(self.make_example(
                &Expr::Derivative {
                    expr: Box::new(Expr::Div(
                        Box::new(Expr::Var(self.x)),
                        Box::new(Expr::Pow(
                            Box::new(Expr::Var(self.x)),
                            Box::new(Expr::int(n)),
                        )),
                    )),
                    var: self.x,
                },
                self.actions.quotient_rule,
                1.0,
            ));

            // d/dx(1 / x)
            examples.push(self.make_example(
                &Expr::Derivative {
                    expr: Box::new(Expr::Div(
                        Box::new(Expr::int(1)),
                        Box::new(Expr::Var(self.x)),
                    )),
                    var: self.x,
                },
                self.actions.quotient_rule,
                1.0,
            ));

            // d/dx(x / (x + 1))
            examples.push(self.make_example(
                &Expr::Derivative {
                    expr: Box::new(Expr::Div(
                        Box::new(Expr::Var(self.x)),
                        Box::new(Expr::Add(
                            Box::new(Expr::Var(self.x)),
                            Box::new(Expr::int(1)),
                        )),
                    )),
                    var: self.x,
                },
                self.actions.quotient_rule,
                1.0,
            ));
        }

        examples
    }

    /// Trig derivatives: d/dx(sin(x)) = cos(x), d/dx(cos(x)) = -sin(x)
    pub fn generate_trig_derivatives(&mut self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        for _ in 0..count {
            // d/dx(sin(x)) = cos(x)
            examples.push(self.make_example(
                &Expr::Derivative {
                    expr: Box::new(Expr::Sin(Box::new(Expr::Var(self.x)))),
                    var: self.x,
                },
                self.actions.sin_derivative,
                1.0,
            ));

            // d/dx(cos(x)) = -sin(x)
            examples.push(self.make_example(
                &Expr::Derivative {
                    expr: Box::new(Expr::Cos(Box::new(Expr::Var(self.x)))),
                    var: self.x,
                },
                self.actions.cos_derivative,
                1.0,
            ));

            // d/dx(exp(x)) = exp(x)
            examples.push(self.make_example(
                &Expr::Derivative {
                    expr: Box::new(Expr::Exp(Box::new(Expr::Var(self.x)))),
                    var: self.x,
                },
                self.actions.exp_derivative,
                1.0,
            ));

            // d/dx(ln(x)) = 1/x
            examples.push(self.make_example(
                &Expr::Derivative {
                    expr: Box::new(Expr::Ln(Box::new(Expr::Var(self.x)))),
                    var: self.x,
                },
                self.actions.ln_derivative,
                1.0,
            ));
        }

        examples
    }

    // =========================================================================
    // EQUATION SOLVING RULES
    // =========================================================================

    /// Cancel addition: x + a = b → x = b - a
    pub fn generate_equation_addition(&mut self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        for _ in 0..count {
            let a = self.rand_nonzero();
            let b = self.rand_nonzero();

            // x + a = b
            examples.push(self.make_example(
                &Expr::Equation {
                    lhs: Box::new(Expr::Add(
                        Box::new(Expr::Var(self.x)),
                        Box::new(Expr::int(a)),
                    )),
                    rhs: Box::new(Expr::int(b)),
                },
                self.actions.cancel_addition,
                1.0,
            ));

            // a + x = b
            examples.push(self.make_example(
                &Expr::Equation {
                    lhs: Box::new(Expr::Add(
                        Box::new(Expr::int(a)),
                        Box::new(Expr::Var(self.x)),
                    )),
                    rhs: Box::new(Expr::int(b)),
                },
                self.actions.cancel_addition,
                1.0,
            ));
        }

        examples
    }

    /// Cancel subtraction: x - a = b → x = b + a
    pub fn generate_equation_subtraction(&mut self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        for _ in 0..count {
            let a = self.rand_nonzero();
            let b = self.rand_nonzero();

            // x - a = b
            examples.push(self.make_example(
                &Expr::Equation {
                    lhs: Box::new(Expr::Sub(
                        Box::new(Expr::Var(self.x)),
                        Box::new(Expr::int(a)),
                    )),
                    rhs: Box::new(Expr::int(b)),
                },
                self.actions.cancel_subtraction,
                1.0,
            ));
        }

        examples
    }

    /// Cancel multiplication: ax = b → x = b/a
    pub fn generate_equation_multiplication(&mut self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        for _ in 0..count {
            let a = self.rand_small();
            let b = self.rand_nonzero();

            if a != 0 {
                // ax = b
                examples.push(self.make_example(
                    &Expr::Equation {
                        lhs: Box::new(Expr::Mul(
                            Box::new(Expr::int(a)),
                            Box::new(Expr::Var(self.x)),
                        )),
                        rhs: Box::new(Expr::int(b)),
                    },
                    self.actions.cancel_multiplication,
                    1.0,
                ));
            }
        }

        examples
    }

    /// Cancel division: x/a = b → x = ab
    pub fn generate_equation_division(&mut self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        for _ in 0..count {
            let a = self.rand_small();
            let b = self.rand_nonzero();

            if a != 0 {
                // x/a = b
                examples.push(self.make_example(
                    &Expr::Equation {
                        lhs: Box::new(Expr::Div(
                            Box::new(Expr::Var(self.x)),
                            Box::new(Expr::int(a)),
                        )),
                        rhs: Box::new(Expr::int(b)),
                    },
                    self.actions.cancel_division,
                    1.0,
                ));
            }
        }

        examples
    }

    /// Linear solve: ax + b = c → x = (c-b)/a
    pub fn generate_linear_equations(&mut self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        for _ in 0..count {
            let a = self.rand_small();
            let b = self.rand_nonzero();
            let c = self.rand_nonzero();

            if a != 0 {
                // ax + b = c
                examples.push(self.make_example(
                    &Expr::Equation {
                        lhs: Box::new(Expr::Add(
                            Box::new(Expr::Mul(
                                Box::new(Expr::int(a)),
                                Box::new(Expr::Var(self.x)),
                            )),
                            Box::new(Expr::int(b)),
                        )),
                        rhs: Box::new(Expr::int(c)),
                    },
                    self.actions.linear_solve,
                    1.0,
                ));

                // ax - b = c
                examples.push(self.make_example(
                    &Expr::Equation {
                        lhs: Box::new(Expr::Sub(
                            Box::new(Expr::Mul(
                                Box::new(Expr::int(a)),
                                Box::new(Expr::Var(self.x)),
                            )),
                            Box::new(Expr::int(b)),
                        )),
                        rhs: Box::new(Expr::int(c)),
                    },
                    self.actions.isolate_variable,
                    1.0,
                ));
            }
        }

        examples
    }

    /// Quadratic equations: ax² + bx + c = 0
    pub fn generate_quadratic_equations(&mut self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        for _ in 0..count {
            let a = self.rand_small();
            let b = self.rand_nonzero();
            let c = self.rand_nonzero();

            if a != 0 {
                // ax² + bx + c = 0
                examples.push(self.make_example(
                    &Expr::Equation {
                        lhs: Box::new(Expr::Add(
                            Box::new(Expr::Add(
                                Box::new(Expr::Mul(
                                    Box::new(Expr::int(a)),
                                    Box::new(Expr::Pow(
                                        Box::new(Expr::Var(self.x)),
                                        Box::new(Expr::int(2)),
                                    )),
                                )),
                                Box::new(Expr::Mul(
                                    Box::new(Expr::int(b)),
                                    Box::new(Expr::Var(self.x)),
                                )),
                            )),
                            Box::new(Expr::int(c)),
                        )),
                        rhs: Box::new(Expr::int(0)),
                    },
                    self.actions.quadratic_formula,
                    1.0,
                ));

                // x² = c (simple quadratic)
                examples.push(self.make_example(
                    &Expr::Equation {
                        lhs: Box::new(Expr::Pow(
                            Box::new(Expr::Var(self.x)),
                            Box::new(Expr::int(2)),
                        )),
                        rhs: Box::new(Expr::int(c.abs())),
                    },
                    self.actions.quadratic_formula,
                    1.0,
                ));
            }
        }

        examples
    }

    // =========================================================================
    // NEGATIVE EXAMPLES
    // =========================================================================

    /// Negative examples: expressions that don't simplify
    pub fn generate_negative_examples(&mut self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        for _ in 0..count {
            let n = self.rand_small();

            // x + n (doesn't simplify when n ≠ 0)
            if n != 0 {
                examples.push(self.make_example(
                    &Expr::Add(Box::new(Expr::Var(self.x)), Box::new(Expr::int(n))),
                    self.actions.no_op,
                    -0.3,
                ));
            }

            // x * n (doesn't simplify when n ≠ 0, 1)
            if n > 1 {
                examples.push(self.make_example(
                    &Expr::Mul(Box::new(Expr::Var(self.x)), Box::new(Expr::int(n))),
                    self.actions.no_op,
                    -0.3,
                ));
            }

            // x^n (doesn't simplify when n > 1)
            if n > 1 {
                examples.push(self.make_example(
                    &Expr::Pow(Box::new(Expr::Var(self.x)), Box::new(Expr::int(n))),
                    self.actions.no_op,
                    -0.2,
                ));
            }

            // sin(x) (already simplified)
            examples.push(self.make_example(
                &Expr::Sin(Box::new(Expr::Var(self.x))),
                self.actions.no_op,
                0.0,
            ));
        }

        examples
    }

    // =========================================================================
    // DATASET GENERATION
    // =========================================================================

    /// Generate a complete 10K training dataset covering all rules.
    pub fn generate_dataset(&mut self, samples_per_category: usize) -> Vec<TrainingExample> {
        let mut all_examples = Vec::new();

        // Algebra (target: ~3000)
        println!("Generating algebra examples...");
        all_examples.extend(self.generate_constant_folding(samples_per_category));
        all_examples.extend(self.generate_identity_rules(samples_per_category));
        all_examples.extend(self.generate_distribute(samples_per_category / 2));
        all_examples.extend(self.generate_factor_common(samples_per_category / 2));
        all_examples.extend(self.generate_difference_of_squares(samples_per_category / 2));
        all_examples.extend(self.generate_collect_like_terms(samples_per_category / 2));

        // Calculus (target: ~3000)
        println!("Generating calculus examples...");
        all_examples.extend(self.generate_power_rule(samples_per_category));
        all_examples.extend(self.generate_constant_rule(samples_per_category));
        all_examples.extend(self.generate_sum_rule(samples_per_category / 2));
        all_examples.extend(self.generate_product_rule(samples_per_category / 2));
        all_examples.extend(self.generate_quotient_rule(samples_per_category / 2));
        all_examples.extend(self.generate_trig_derivatives(samples_per_category));

        // Equations (target: ~3000)
        println!("Generating equation solving examples...");
        all_examples.extend(self.generate_equation_addition(samples_per_category / 2));
        all_examples.extend(self.generate_equation_subtraction(samples_per_category / 2));
        all_examples.extend(self.generate_equation_multiplication(samples_per_category / 2));
        all_examples.extend(self.generate_equation_division(samples_per_category / 2));
        all_examples.extend(self.generate_linear_equations(samples_per_category));
        all_examples.extend(self.generate_quadratic_equations(samples_per_category / 2));

        // Negative examples (target: ~1000)
        println!("Generating negative examples...");
        all_examples.extend(self.generate_negative_examples(samples_per_category / 2));

        println!("Total training examples: {}", all_examples.len());

        // Shuffle
        all_examples.shuffle(&mut self.rng);

        all_examples
    }

    /// Generate a small test dataset for validation.
    pub fn generate_validation_set(&mut self, size: usize) -> Vec<TrainingExample> {
        let per_cat = size / 20;
        self.generate_dataset(per_cat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_constant_folding() {
        let mut gen = DataGenerator::new(Device::Cpu);
        let examples = gen.generate_constant_folding(10);
        assert!(examples.len() >= 40);
        let expected = gen.vocabulary().index_of("algebra", "const_fold").unwrap() as u32;
        assert!(examples.iter().all(|e| e.target_action == expected));
    }

    #[test]
    fn test_generate_dataset() {
        let mut gen = DataGenerator::new(Device::Cpu);
        let examples = gen.generate_dataset(50);

        // Should have around 1000+ examples with 50 samples_per_category
        assert!(examples.len() > 500);
        println!("Generated {} examples", examples.len());
    }

    #[test]
    fn test_equation_examples() {
        let mut gen = DataGenerator::new(Device::Cpu);
        let examples = gen.generate_linear_equations(10);
        assert!(!examples.is_empty());
    }

    #[test]
    fn test_quotient_rule_examples() {
        let mut gen = DataGenerator::new(Device::Cpu);
        let examples = gen.generate_quotient_rule(10);
        assert!(examples.len() >= 20);
    }
}
