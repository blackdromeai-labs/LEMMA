// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Author: Pushp Kharat

//! Where a policy network's weights came from.
//!
//! A freshly constructed network holds random weights. That is a legitimate starting point
//! for training, but a search that consults one is not neural guidance, and results produced
//! with one must not be reported as though a trained model chose the moves. Every
//! [`crate::PolicyNetwork`] therefore carries a [`ModelProvenance`], and the search layer
//! propagates it into its results.

use std::fmt;
use std::path::{Path, PathBuf};

use mm_rules::ActionVocabulary;
use serde::{Deserialize, Serialize};

/// Manifest written next to a weights file so the weights can be matched to a vocabulary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Version of the action-vocabulary construction scheme.
    pub vocabulary_version: u32,
    /// Content digest of the action vocabulary the model was trained against.
    pub vocabulary_digest: u64,
    /// Number of rule actions in that vocabulary.
    pub num_actions: usize,
    /// Number of policy output classes (rule actions plus the reserved terminal class).
    pub num_policy_classes: usize,
    /// Token embedding dimension.
    pub embed_dim: usize,
    /// Feed-forward hidden dimension.
    pub hidden_dim: usize,
    /// Attention heads per block.
    pub num_heads: usize,
    /// Number of transformer blocks.
    pub num_layers: usize,
    /// Maximum token sequence length.
    pub max_seq_len: usize,
    /// Token vocabulary size of the expression encoder.
    pub token_vocab_size: usize,
}

impl ModelManifest {
    /// Manifest path that belongs to a weights path (`weights.safetensors` ->
    /// `weights.manifest.json`).
    pub fn path_for(weights: &Path) -> PathBuf {
        weights.with_extension("manifest.json")
    }
}

/// Why a model could not be loaded.
#[derive(Debug)]
pub enum ModelLoadError {
    /// The weights or manifest file could not be read.
    Io(std::io::Error),
    /// The manifest was not valid JSON.
    Manifest(serde_json::Error),
    /// The manifest does not describe the vocabulary in use.
    Incompatible(mm_rules::ActionError),
    /// The manifest describes a differently shaped network.
    ShapeMismatch {
        /// Field that disagrees.
        field: &'static str,
        /// Value recorded in the manifest.
        found: usize,
        /// Value required by the network configuration in use.
        expected: usize,
    },
    /// Candle could not load the weights themselves.
    Weights(candle_core::Error),
}

impl fmt::Display for ModelLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelLoadError::Io(e) => write!(f, "could not read model: {e}"),
            ModelLoadError::Manifest(e) => write!(f, "could not parse model manifest: {e}"),
            ModelLoadError::Incompatible(e) => write!(f, "model is incompatible: {e}"),
            ModelLoadError::ShapeMismatch {
                field,
                found,
                expected,
            } => write!(
                f,
                "model manifest {field} is {found} but {expected} is required"
            ),
            ModelLoadError::Weights(e) => write!(f, "could not load weights: {e}"),
        }
    }
}

impl std::error::Error for ModelLoadError {}

impl From<std::io::Error> for ModelLoadError {
    fn from(e: std::io::Error) -> Self {
        ModelLoadError::Io(e)
    }
}

impl From<serde_json::Error> for ModelLoadError {
    fn from(e: serde_json::Error) -> Self {
        ModelLoadError::Manifest(e)
    }
}

impl From<candle_core::Error> for ModelLoadError {
    fn from(e: candle_core::Error) -> Self {
        ModelLoadError::Weights(e)
    }
}

/// Where a network's weights came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelProvenance {
    /// Randomly initialised weights. Priors from such a network carry no learned signal.
    Untrained {
        /// Digest of the action vocabulary the (untrained) head was sized for.
        vocabulary_digest: u64,
    },
    /// Weights loaded from a file whose manifest matched the active vocabulary.
    Loaded {
        /// File the weights were read from.
        path: PathBuf,
        /// Digest of the action vocabulary recorded in the manifest.
        vocabulary_digest: u64,
        /// Digest of the weights file contents.
        weights_digest: u64,
    },
}

impl ModelProvenance {
    /// Whether these weights were trained (as opposed to randomly initialised).
    pub fn is_trained(&self) -> bool {
        matches!(self, ModelProvenance::Loaded { .. })
    }

    /// Digest of the action vocabulary the weights belong to.
    pub fn vocabulary_digest(&self) -> u64 {
        match self {
            ModelProvenance::Untrained { vocabulary_digest }
            | ModelProvenance::Loaded {
                vocabulary_digest, ..
            } => *vocabulary_digest,
        }
    }
}

impl fmt::Display for ModelProvenance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelProvenance::Untrained { vocabulary_digest } => write!(
                f,
                "untrained (random weights, vocabulary {vocabulary_digest:#018x})"
            ),
            ModelProvenance::Loaded {
                path,
                vocabulary_digest,
                weights_digest,
            } => write!(
                f,
                "loaded from {} (weights {weights_digest:#018x}, vocabulary {vocabulary_digest:#018x})",
                path.display()
            ),
        }
    }
}

/// FNV-1a digest of a byte slice, used for weights-file identity.
pub fn digest_bytes(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET;
    for b in bytes {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// Check a manifest against the vocabulary that will be used at inference time.
pub fn check_manifest(
    manifest: &ModelManifest,
    vocab: &ActionVocabulary,
) -> Result<(), ModelLoadError> {
    if manifest.vocabulary_version != mm_rules::ACTION_VOCABULARY_VERSION {
        return Err(ModelLoadError::ShapeMismatch {
            field: "vocabulary_version",
            found: manifest.vocabulary_version as usize,
            expected: mm_rules::ACTION_VOCABULARY_VERSION as usize,
        });
    }
    vocab
        .check_digest(manifest.vocabulary_digest)
        .map_err(ModelLoadError::Incompatible)?;
    if manifest.num_actions != vocab.len() {
        return Err(ModelLoadError::ShapeMismatch {
            field: "num_actions",
            found: manifest.num_actions,
            expected: vocab.len(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest_for(vocab: &ActionVocabulary) -> ModelManifest {
        ModelManifest {
            vocabulary_version: mm_rules::ACTION_VOCABULARY_VERSION,
            vocabulary_digest: vocab.digest(),
            num_actions: vocab.len(),
            num_policy_classes: vocab.len() + 1,
            embed_dim: 64,
            hidden_dim: 128,
            num_heads: 4,
            num_layers: 2,
            max_seq_len: 64,
            token_vocab_size: 64,
        }
    }

    #[test]
    fn matching_manifest_is_accepted() {
        let vocab = ActionVocabulary::standard();
        assert!(check_manifest(&manifest_for(&vocab), &vocab).is_ok());
    }

    #[test]
    fn digest_mismatch_is_rejected() {
        let vocab = ActionVocabulary::standard();
        let mut manifest = manifest_for(&vocab);
        manifest.vocabulary_digest ^= 1;
        assert!(matches!(
            check_manifest(&manifest, &vocab),
            Err(ModelLoadError::Incompatible(_))
        ));
    }

    #[test]
    fn action_count_mismatch_is_rejected() {
        let vocab = ActionVocabulary::standard();
        let mut manifest = manifest_for(&vocab);
        manifest.num_actions = 25;
        assert!(matches!(
            check_manifest(&manifest, &vocab),
            Err(ModelLoadError::Incompatible(_)) | Err(ModelLoadError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn untrained_provenance_is_not_trained() {
        let vocab = ActionVocabulary::standard();
        let p = ModelProvenance::Untrained {
            vocabulary_digest: vocab.digest(),
        };
        assert!(!p.is_trained());
        assert!(p.to_string().contains("untrained"));
    }
}
