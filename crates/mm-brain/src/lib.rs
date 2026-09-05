// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Author: Pushp Kharat

//! # mm-brain
//!
//! Neural network for strategy learning in the Math Monster system.
//!
//! This crate provides:
//! - [`ExpressionEncoder`] - Convert expressions to neural network inputs
//! - [`MathNetwork`] - The actual neural network for policy/value prediction
//! - [`PolicyNetwork`] - High-level API for rule selection
//! - [`Trainer`] - Training loop for the network
//! - [`DataGenerator`] - Synthetic training data generation
//! - [`ModelProvenance`] - Whether a network holds trained or random weights
//!
//! ## Architecture
//!
//! ```text
//! Expression → Tokenize → Embed → Transformer → Policy Head → Rule Probs
//!                                             → Value Head  → State Value
//! ```

pub mod data;
pub mod data_compositional;
pub mod encoder;
pub mod network;
pub mod policy;
pub mod provenance;
pub mod substitution;
pub mod text_classifier;
pub mod training;

pub use data::DataGenerator;
pub use data_compositional::{CompositionalDataGenerator, GenerationReport, COMPOSITIONAL_SEED};
pub use encoder::ExpressionEncoder;
pub use network::{MathNetwork, NetworkConfig};
pub use policy::PolicyNetwork;
pub use provenance::{ModelLoadError, ModelManifest, ModelProvenance};
pub use substitution::{SearchHint, SubstitutionPrediction, SubstitutionPredictor};
pub use text_classifier::KeywordProblemClassifier;
pub use training::{Trainer, TrainingConfig};
