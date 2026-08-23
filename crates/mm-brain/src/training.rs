// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Author: Pushp Kharat

//! Training utilities for the neural network.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{Optimizer, VarMap};
use mm_rules::ActionVocabulary;

use crate::encoder::ExpressionEncoder;
use crate::network::{MathNetwork, NetworkConfig};
use crate::provenance::ModelManifest;

/// Training configuration.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Learning rate.
    pub learning_rate: f64,
    /// Weight decay.
    pub weight_decay: f64,
    /// Batch size.
    pub batch_size: usize,
    /// Number of epochs.
    pub epochs: usize,
    /// Value loss weight (policy loss weight is 1.0).
    pub value_weight: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            weight_decay: 0.01,
            batch_size: 32,
            epochs: 100,
            value_weight: 0.5,
        }
    }
}

/// A single training example.
///
/// `target_action` is a dense [`ActionVocabulary`] index, not a [`mm_rules::RuleId`]. The two
/// were previously conflated, which put the label one column away from the rule it named.
#[derive(Debug, Clone)]
pub struct TrainingExample {
    /// The expression state.
    pub tokens: Vec<u32>,
    /// Target policy class: an action index, or the reserved terminal class.
    pub target_action: u32,
    /// Target value (-1 to 1).
    pub target_value: f32,
}

/// Trainer for the neural network.
pub struct Trainer {
    varmap: VarMap,
    network: MathNetwork,
    optimizer: AdamW,
    encoder: ExpressionEncoder,
    config: TrainingConfig,
    device: Device,
    vocabulary: ActionVocabulary,
}

impl Trainer {
    /// Create a new trainer for the standard action vocabulary.
    pub fn new(
        network_config: NetworkConfig,
        training_config: TrainingConfig,
        device: Device,
    ) -> Result<Self> {
        Self::with_vocabulary(
            network_config,
            training_config,
            device,
            ActionVocabulary::standard(),
        )
    }

    /// Create a trainer bound to a specific action vocabulary.
    ///
    /// The vocabulary is recorded in the manifest written by [`Self::save`], so weights can
    /// never be loaded against a rule set they were not trained for.
    pub fn with_vocabulary(
        network_config: NetworkConfig,
        training_config: TrainingConfig,
        device: Device,
        vocabulary: ActionVocabulary,
    ) -> Result<Self> {
        let expected = crate::network::policy_classes_for(&vocabulary);
        if network_config.num_policy_classes != expected {
            return Err(candle_core::Error::Msg(format!(
                "network has {} policy classes but the action vocabulary needs {}",
                network_config.num_policy_classes, expected
            )));
        }
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let network = MathNetwork::new_with_vb(network_config, vb)?;
        let encoder = ExpressionEncoder::new(device.clone());

        let params = ParamsAdamW {
            lr: training_config.learning_rate,
            weight_decay: training_config.weight_decay,
            ..Default::default()
        };

        let optimizer = AdamW::new(varmap.all_vars(), params).unwrap();

        Ok(Self {
            varmap,
            network,
            optimizer,
            encoder,
            config: training_config,
            device,
            vocabulary,
        })
    }

    /// The action vocabulary this trainer's labels are expressed in.
    pub fn vocabulary(&self) -> &ActionVocabulary {
        &self.vocabulary
    }

    /// Train on a batch of examples.
    ///
    /// Returns (policy_loss, value_loss).
    pub fn train_step(&mut self, examples: &[TrainingExample]) -> Result<(f32, f32)> {
        let batch_size = examples.len();
        let seq_len = self.encoder.max_length();

        // Prepare batch tensors
        let tokens: Vec<u32> = examples
            .iter()
            .flat_map(|e| {
                let mut t = e.tokens.clone();
                t.resize(seq_len, 0); // Pad
                t
            })
            .collect();

        let target_rules: Vec<u32> = examples.iter().map(|e| e.target_action).collect();
        let target_values: Vec<f32> = examples.iter().map(|e| e.target_value).collect();

        // Create tensors
        let tokens =
            Tensor::new(tokens.as_slice(), &self.device)?.reshape((batch_size, seq_len))?;
        let target_rules = Tensor::new(target_rules.as_slice(), &self.device)?;
        let target_values =
            Tensor::new(target_values.as_slice(), &self.device)?.reshape((batch_size, 1))?;

        // Forward pass
        let (policy_logits, values) = self.network.forward(&tokens)?;

        // Policy loss: cross-entropy
        let policy_loss = candle_nn::loss::cross_entropy(&policy_logits, &target_rules)?;

        // Value loss: MSE
        let value_loss = candle_nn::loss::mse(&values, &target_values)?;

        // Combined loss
        let total_loss = (&policy_loss + &value_loss * self.config.value_weight as f64)?;

        // Backward pass
        self.optimizer.backward_step(&total_loss)?;

        let policy_loss_val: f32 = policy_loss.to_scalar()?;
        let value_loss_val: f32 = value_loss.to_scalar()?;

        Ok((policy_loss_val, value_loss_val))
    }

    /// Train on all examples for multiple epochs.
    pub fn train(&mut self, examples: &[TrainingExample]) -> Result<Vec<(f32, f32)>> {
        let mut history = Vec::new();

        for epoch in 0..self.config.epochs {
            let mut epoch_policy_loss = 0.0;
            let mut epoch_value_loss = 0.0;
            let mut num_batches = 0;

            // Simple batching (no shuffle for simplicity)
            for batch_start in (0..examples.len()).step_by(self.config.batch_size) {
                let batch_end = (batch_start + self.config.batch_size).min(examples.len());
                let batch = &examples[batch_start..batch_end];

                if batch.is_empty() {
                    continue;
                }

                let (policy_loss, value_loss) = self.train_step(batch)?;
                epoch_policy_loss += policy_loss;
                epoch_value_loss += value_loss;
                num_batches += 1;
            }

            if num_batches > 0 {
                epoch_policy_loss /= num_batches as f32;
                epoch_value_loss /= num_batches as f32;
            }

            history.push((epoch_policy_loss, epoch_value_loss));

            if epoch % 10 == 0 {
                println!(
                    "Epoch {}: policy_loss={:.4}, value_loss={:.4}",
                    epoch, epoch_policy_loss, epoch_value_loss
                );
            }
        }

        Ok(history)
    }

    /// Get the trained network.
    pub fn network(&self) -> &MathNetwork {
        &self.network
    }

    /// Get the encoder.
    pub fn encoder(&self) -> &ExpressionEncoder {
        &self.encoder
    }

    /// Get the device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Save trained model weights plus the manifest that identifies their vocabulary.
    ///
    /// Without the manifest, weights could be loaded against a rule set they were not
    /// trained for and would produce priors for the wrong rules.
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        self.varmap.save(path)?;

        let config = self.network.config();
        let manifest = ModelManifest {
            vocabulary_version: mm_rules::ACTION_VOCABULARY_VERSION,
            vocabulary_digest: self.vocabulary.digest(),
            num_actions: self.vocabulary.len(),
            num_policy_classes: config.num_policy_classes,
            embed_dim: config.embed_dim,
            hidden_dim: config.hidden_dim,
            num_heads: config.num_heads,
            num_layers: config.num_layers,
            max_seq_len: config.max_seq_len,
            token_vocab_size: config.vocab_size,
        };
        let json = serde_json::to_string_pretty(&manifest)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        std::fs::write(ModelManifest::path_for(path), json)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        Ok(())
    }

    /// Load model weights from a file into this trainer's network.
    pub fn load<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<()> {
        self.varmap.load(path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainer_creation() {
        let trainer = Trainer::new(
            NetworkConfig::default(),
            TrainingConfig::default(),
            Device::Cpu,
        );
        assert!(trainer.is_ok());
    }

    #[test]
    fn test_single_train_step() {
        let mut trainer = Trainer::new(
            NetworkConfig::default(),
            TrainingConfig::default(),
            Device::Cpu,
        )
        .unwrap();

        // Dummy example: action 0 is the first rule in the vocabulary (`algebra::const_fold`).
        let example = TrainingExample {
            tokens: vec![1, 26, 4, 27, 2],
            target_action: 0,
            target_value: 1.0,
        };

        let (policy_loss, value_loss) = trainer.train_step(&[example]).unwrap();

        assert!(policy_loss >= 0.0);
        assert!(value_loss >= 0.0);
    }
}
