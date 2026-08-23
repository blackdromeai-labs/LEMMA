// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Author: Pushp Kharat

//! Policy network for rule selection.
//!
//! The policy head has one column per action in an [`ActionVocabulary`] plus a reserved
//! terminal class. Callers read priors through [`PolicyNetwork::rule_priors`] and
//! [`ActionVocabulary::prior_for_rule`], never by indexing with a raw [`RuleId`]: rule
//! identifiers are sparse and reach five digits, so that indexing silently missed.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Result};
use candle_nn::{VarBuilder, VarMap};
use mm_core::Expr;
use mm_rules::{ActionVocabulary, RuleId};

use crate::encoder::ExpressionEncoder;
use crate::network::{MathNetwork, NetworkConfig};
use crate::provenance::{
    check_manifest, digest_bytes, ModelLoadError, ModelManifest, ModelProvenance,
};

/// Policy network for selecting which rule to apply.
///
/// Wraps the neural network and the action vocabulary its head was sized for, and records
/// where its weights came from.
pub struct PolicyNetwork {
    network: MathNetwork,
    encoder: ExpressionEncoder,
    device: Device,
    vocabulary: ActionVocabulary,
    provenance: ModelProvenance,
}

impl PolicyNetwork {
    /// Create a randomly initialised policy network for the standard action vocabulary.
    ///
    /// The name is deliberate: the weights carry no learned signal, and
    /// [`Self::provenance`] reports [`ModelProvenance::Untrained`] so callers cannot present
    /// the result as trained neural guidance.
    pub fn untrained() -> Result<Self> {
        Self::untrained_for(ActionVocabulary::standard(), Device::Cpu)
    }

    /// Create a randomly initialised policy network for a specific vocabulary and device.
    pub fn untrained_for(vocabulary: ActionVocabulary, device: Device) -> Result<Self> {
        let config = NetworkConfig::for_vocabulary(&vocabulary);
        let network = MathNetwork::new(config, &device)?;
        let encoder = ExpressionEncoder::new(device.clone());
        let provenance = ModelProvenance::Untrained {
            vocabulary_digest: vocabulary.digest(),
        };

        Ok(Self {
            network,
            encoder,
            device,
            vocabulary,
            provenance,
        })
    }

    /// Load trained weights, rejecting anything built for a different action vocabulary.
    ///
    /// `weights` is a safetensors file written by [`crate::Trainer::save`]; its manifest is
    /// read from [`ModelManifest::path_for`].
    pub fn load(
        weights: &Path,
        vocabulary: ActionVocabulary,
        device: Device,
    ) -> std::result::Result<Self, ModelLoadError> {
        let manifest_path = ModelManifest::path_for(weights);
        let manifest: ModelManifest =
            serde_json::from_str(&std::fs::read_to_string(&manifest_path)?)?;
        check_manifest(&manifest, &vocabulary)?;

        let config = NetworkConfig {
            vocab_size: manifest.token_vocab_size,
            embed_dim: manifest.embed_dim,
            hidden_dim: manifest.hidden_dim,
            num_heads: manifest.num_heads,
            num_layers: manifest.num_layers,
            max_seq_len: manifest.max_seq_len,
            num_policy_classes: manifest.num_policy_classes,
            dropout: 0.0,
        };
        let expected_classes = crate::network::policy_classes_for(&vocabulary);
        if config.num_policy_classes != expected_classes {
            return Err(ModelLoadError::ShapeMismatch {
                field: "num_policy_classes",
                found: config.num_policy_classes,
                expected: expected_classes,
            });
        }

        let bytes = std::fs::read(weights)?;
        let weights_digest = digest_bytes(&bytes);

        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let network = MathNetwork::new_with_vb(config, vb)?;
        varmap.load(weights)?;

        Ok(Self {
            network,
            encoder: ExpressionEncoder::new(device.clone()),
            device,
            provenance: ModelProvenance::Loaded {
                path: PathBuf::from(weights),
                vocabulary_digest: manifest.vocabulary_digest,
                weights_digest,
            },
            vocabulary,
        })
    }

    /// Where these weights came from.
    pub fn provenance(&self) -> &ModelProvenance {
        &self.provenance
    }

    /// The action vocabulary this head was sized for.
    pub fn vocabulary(&self) -> &ActionVocabulary {
        &self.vocabulary
    }

    /// Raw policy probabilities, including the reserved terminal class.
    pub fn forward(&self, expr: &Expr) -> Result<Vec<f32>> {
        let tokens = self.encoder.encode(expr)?;
        let tokens = tokens.unsqueeze(0)?;

        let policy = self.network.get_policy(&tokens)?;
        let policy = policy.squeeze(0)?;
        policy.to_vec1()
    }

    /// Policy probabilities for rule actions only, aligned with the action vocabulary.
    ///
    /// The returned vector has exactly `vocabulary().len()` entries, so
    /// [`ActionVocabulary::prior_for_rule`] can read it without a length guess.
    pub fn rule_priors(&self, expr: &Expr) -> Result<Vec<f32>> {
        let mut probs = self.forward(expr)?;
        probs.truncate(self.vocabulary.len());
        Ok(probs)
    }

    /// Get value estimate for an expression.
    ///
    /// Returns a value between -1 (bad state) and 1 (good state).
    pub fn get_value(&self, expr: &Expr) -> Result<f32> {
        let tokens = self.encoder.encode(expr)?;
        let tokens = tokens.unsqueeze(0)?;

        let value = self.network.get_value(&tokens)?;
        value.squeeze(0)?.squeeze(0)?.to_scalar()
    }

    /// Get the top-k most likely rules.
    ///
    /// Indices are resolved through the action vocabulary, so the returned identifiers are
    /// the rules the columns actually stand for.
    pub fn top_k(&self, expr: &Expr, k: usize) -> Result<Vec<(RuleId, f32)>> {
        let probs = self.rule_priors(expr)?;

        let mut indexed: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
        // Sort by probability, breaking ties by action index so the order is deterministic.
        indexed.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });

        Ok(indexed
            .into_iter()
            .take(k)
            .filter_map(|(idx, prob)| self.vocabulary.rule_id_at(idx).ok().map(|id| (id, prob)))
            .collect())
    }

    /// Get the network for training.
    pub fn network(&self) -> &MathNetwork {
        &self.network
    }

    /// Get mutable network for training.
    pub fn network_mut(&mut self) -> &mut MathNetwork {
        &mut self.network
    }

    /// Get the encoder.
    pub fn encoder(&self) -> &ExpressionEncoder {
        &self.encoder
    }

    /// Get the device.
    pub fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn untrained_network_is_labelled_untrained() {
        let policy = PolicyNetwork::untrained().unwrap();
        assert!(!policy.provenance().is_trained());
        assert_eq!(
            policy.provenance().vocabulary_digest(),
            policy.vocabulary().digest()
        );
    }

    #[test]
    fn policy_head_matches_the_action_vocabulary() {
        let policy = PolicyNetwork::untrained().unwrap();
        let expr = Expr::Add(Box::new(Expr::int(2)), Box::new(Expr::int(3)));

        let all = policy.forward(&expr).unwrap();
        assert_eq!(all.len(), policy.vocabulary().len() + 1);
        let sum: f32 = all.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        let rule_priors = policy.rule_priors(&expr).unwrap();
        assert_eq!(rule_priors.len(), policy.vocabulary().len());
    }

    #[test]
    fn priors_are_readable_for_every_registered_rule() {
        let policy = PolicyNetwork::untrained().unwrap();
        let expr = Expr::int(5);
        let priors = policy.rule_priors(&expr).unwrap();

        let rules = mm_rules::standard_rules();
        for rule in rules.all() {
            policy
                .vocabulary()
                .prior_for_rule(&priors, rule.id)
                .unwrap_or_else(|e| panic!("rule {} has no prior: {e}", rule.name));
        }
    }

    #[test]
    fn top_k_returns_registered_rules_in_deterministic_order() {
        let policy = PolicyNetwork::untrained().unwrap();
        let expr = Expr::int(5);

        let top = policy.top_k(&expr, 3).unwrap();
        assert_eq!(top.len(), 3);
        assert!(top[0].1 >= top[1].1);
        assert!(top[1].1 >= top[2].1);

        let rules = mm_rules::standard_rules();
        for (id, _) in &top {
            assert!(rules.get(*id).is_some(), "top_k named a non-existent rule");
        }

        let again = policy.top_k(&expr, 3).unwrap();
        assert_eq!(
            top.iter().map(|(id, _)| id.0).collect::<Vec<_>>(),
            again.iter().map(|(id, _)| id.0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn value_estimate_is_bounded() {
        let policy = PolicyNetwork::untrained().unwrap();
        let value = policy.get_value(&Expr::int(5)).unwrap();
        assert!((-1.0..=1.0).contains(&value));
    }

    #[test]
    fn loading_a_missing_model_is_an_error_not_a_silent_untrained_network() {
        let result = PolicyNetwork::load(
            Path::new("does-not-exist.safetensors"),
            ActionVocabulary::standard(),
            Device::Cpu,
        );
        match result {
            Err(ModelLoadError::Io(_)) => {}
            Err(other) => panic!("expected an I/O error, got {other}"),
            Ok(_) => panic!("a missing model must not silently produce a network"),
        }
    }
}
