//! Train the "compositional" policy: the second arm of the trained-vs-uniform comparison,
//! trained on nested/compound-operand states for the same rule set `train_policy` covers,
//! instead of `data.rs`'s flat single-level templates.
//!
//! This does NOT read, generate against, or touch the locked evaluation corpus at
//! `experiments/corpus/problems.jsonl`. The training data comes entirely from
//! `mm_brain::data_compositional::CompositionalDataGenerator`, which builds every example from
//! rule *definitions* and validates each one against the real rule and the real verifier before
//! keeping it (see that module's doc comment for the full rationale).
//!
//! Same network shape and training hyperparameters as `train_policy` (seq_len 32, embed 128,
//! hidden 256, 8 heads, 3 layers, AdamW, batch 512) so a difference in outcome reflects the
//! training data, not an unmatched architecture. Saves to a separate path so the existing
//! shallow-trained model (`experiments/models/policy.safetensors`) is left untouched as the
//! frozen baseline.
//!
//! Usage:
//!   train_compositional_policy [--epochs N] [--per-rule R] [--seq-len L] [--batch B]
//!                              [--val-frac F] [--out PATH]

use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use mm_brain::data_compositional::{CompositionalDataGenerator, COMPOSITIONAL_SEED};
use mm_brain::training::TrainingExample;
use mm_brain::{NetworkConfig, Trainer, TrainingConfig};
use rand::prelude::*;
use rand::rngs::StdRng;

fn arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn training_device() -> candle_core::Result<Device> {
    #[cfg(feature = "cuda")]
    {
        Device::new_cuda(0)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(candle_core::Error::Msg(
            "train_compositional_policy requires GPU support; rerun with `--features cuda`"
                .to_string(),
        ))
    }
}

fn policy_accuracy(
    trainer: &Trainer,
    examples: &[TrainingExample],
    seq_len: usize,
    batch: usize,
) -> candle_core::Result<f32> {
    let mut correct = 0usize;
    for chunk in examples.chunks(batch) {
        let tokens: Vec<u32> = chunk
            .iter()
            .flat_map(|e| {
                let mut t = e.tokens.clone();
                t.resize(seq_len, 0);
                t
            })
            .collect();
        let tokens =
            Tensor::new(tokens.as_slice(), trainer.device())?.reshape((chunk.len(), seq_len))?;
        let (logits, _) = trainer.network().forward(&tokens)?;
        let predicted = logits
            .argmax(candle_core::D::Minus1)?
            .to_dtype(DType::U32)?;
        let predicted: Vec<u32> = predicted.to_vec1()?;
        correct += predicted
            .iter()
            .zip(chunk.iter())
            .filter(|(p, e)| **p == e.target_action)
            .count();
    }
    Ok(correct as f32 / examples.len().max(1) as f32)
}

fn main() {
    let epochs: usize = arg("--epochs", 40);
    let per_rule: usize = arg("--per-rule", 400);
    let seq_len: usize = arg("--seq-len", 32);
    let batch: usize = arg("--batch", 512);
    let val_frac: f32 = arg("--val-frac", 0.15);
    let out: String = arg(
        "--out",
        "experiments/models/policy_compositional.safetensors".to_string(),
    );
    let manifest_path: String = arg(
        "--manifest",
        "experiments/COMPOSITIONAL_DATA_MANIFEST.md".to_string(),
    );

    let device = match training_device() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to initialize CUDA device 0: {e}");
            std::process::exit(2);
        }
    };
    println!("device        : {device:?}");
    println!("seed          : 0x{COMPOSITIONAL_SEED:016X} (frozen)");
    println!("seq_len       : {seq_len}");
    println!("per_rule      : {per_rule}");

    // ---- data ------------------------------------------------------------------------
    let vocabulary = mm_rules::ActionVocabulary::standard();
    let mut generator = CompositionalDataGenerator::new(device.clone(), seq_len);
    let report = generator.generate(per_rule);
    let examples = report.examples;
    println!(
        "examples      : {} validated, {} collision cases, across {} rules",
        examples.len(),
        report.collision_examples,
        report.per_rule.len()
    );
    for (key, attempted, kept) in &report.per_rule {
        println!("  {key:<40} attempted {attempted:>5}  kept {kept:>5}");
    }

    // ---- manifest ----------------------------------------------------------------------
    // Written before training starts, from the same `report` that will be trained on, so the
    // manifest can never drift from what was actually generated.
    let mut manifest = String::new();
    manifest.push_str("# Compositional Training Data Manifest\n\n");
    manifest.push_str("Generated by `cargo run --release --example train_compositional_policy -p mm-solver --features cuda`.\n\n");
    manifest.push_str(&format!("- Seed (frozen): `0x{COMPOSITIONAL_SEED:016X}`\n"));
    manifest.push_str(&format!(
        "- Requested examples per rule template: {per_rule}\n"
    ));
    manifest.push_str(&format!("- Encoder sequence length: {seq_len}\n"));
    manifest.push_str(&format!(
        "- Total validated, deduplicated examples: {}\n",
        examples.len()
    ));
    manifest.push_str(&format!(
        "- Collision-case examples (multiple simultaneously legal rules at the same node, e.g. `1 * (A + B)`): {}\n",
        report.collision_examples
    ));
    manifest.push_str(&format!(
        "- Rules covered: {} of the 24 `data.rs` targets (see `mm_brain::data_compositional` module docs for why one, `equations::quadratic_formula`, is excluded)\n\n",
        report.per_rule.len()
    ));
    manifest.push_str("## Validation method\n\n");
    manifest.push_str(
        "Every example is checked before being kept: the target rule's `can_apply` must accept \
         the generated expression, `apply` must produce a result different from the input, and \
         `Verifier::verify_step` (default `Verifier::new()`, symbolic level) must return \
         `Valid`. An example failing any of these is dropped, not corrected. No example is \
         derived from, or checked against, the locked evaluation corpus at \
         `experiments/corpus/problems.jsonl`.\n\n",
    );
    manifest.push_str("## Per-rule attempted vs. kept\n\n");
    manifest.push_str("| rule | attempted | kept | drop rate |\n");
    manifest.push_str("|---|---:|---:|---:|\n");
    for (key, attempted, kept) in &report.per_rule {
        let drop_rate = if *attempted > 0 {
            100.0 * (1.0 - *kept as f64 / *attempted as f64)
        } else {
            0.0
        };
        manifest.push_str(&format!(
            "| `{key}` | {attempted} | {kept} | {drop_rate:.1}% |\n"
        ));
    }
    if let Some(parent) = std::path::Path::new(&manifest_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Err(e) = std::fs::write(&manifest_path, &manifest) {
        eprintln!("Failed to write manifest to {manifest_path}: {e}");
        std::process::exit(6);
    }
    println!("manifest      : {manifest_path}");

    // ---- split -----------------------------------------------------------------------
    let mut rng = StdRng::seed_from_u64(COMPOSITIONAL_SEED ^ 0x5D1F_7A3C_9E11_0042);
    let mut shuffled = examples;
    shuffled.shuffle(&mut rng);
    let n_val = ((shuffled.len() as f32) * val_frac).round() as usize;
    let (val, train) = shuffled.split_at(n_val);
    println!("split         : {} train / {} val", train.len(), val.len());

    // ---- model -----------------------------------------------------------------------
    // Identical shape to train_policy.rs's network so any difference in the eventual
    // trained-vs-trained comparison is attributable to the training data, not the architecture.
    let network_config = NetworkConfig {
        vocab_size: 64,
        embed_dim: 128,
        hidden_dim: 256,
        num_heads: 8,
        num_layers: 3,
        max_seq_len: seq_len,
        num_policy_classes: mm_brain::network::policy_classes_for(&vocabulary),
        dropout: 0.1,
    };
    let training_config = TrainingConfig {
        learning_rate: 1e-3,
        weight_decay: 0.01,
        batch_size: batch,
        epochs,
        value_weight: 0.5,
    };
    let mut trainer = match Trainer::with_vocabulary(
        network_config,
        training_config,
        device.clone(),
        vocabulary.clone(),
    ) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to create trainer: {e}");
            std::process::exit(3);
        }
    };

    // ---- train -----------------------------------------------------------------------
    println!("\nepoch  policy_loss  value_loss  val_acc   secs");
    let mut order: Vec<usize> = (0..train.len()).collect();
    let mut best_acc = 0.0f32;
    let mut best_epoch = 0usize;
    let start_all = Instant::now();

    for epoch in 0..epochs {
        let t0 = Instant::now();
        order.shuffle(&mut rng);

        let mut policy_sum = 0.0f32;
        let mut value_sum = 0.0f32;
        let mut steps = 0usize;

        for chunk in order.chunks(batch) {
            let batch_examples: Vec<TrainingExample> =
                chunk.iter().map(|&i| train[i].clone()).collect();
            match trainer.train_step(&batch_examples) {
                Ok((p, v)) => {
                    policy_sum += p;
                    value_sum += v;
                    steps += 1;
                }
                Err(e) => {
                    eprintln!("train_step failed at epoch {epoch}: {e}");
                    std::process::exit(4);
                }
            }
        }

        let val_acc = policy_accuracy(&trainer, val, seq_len, batch).unwrap_or(f32::NAN);
        let secs = t0.elapsed().as_secs_f32();
        println!(
            "{epoch:>5}  {:>11.4}  {:>10.4}  {:>7.4}  {secs:>5.2}",
            policy_sum / steps.max(1) as f32,
            value_sum / steps.max(1) as f32,
            val_acc
        );

        if val_acc > best_acc {
            best_acc = val_acc;
            best_epoch = epoch;
        }
    }

    let total_secs = start_all.elapsed().as_secs_f32();
    println!("\ntotal {total_secs:.1}s | best val_acc {best_acc:.4} at epoch {best_epoch}");

    // ---- save ------------------------------------------------------------------------
    if let Some(parent) = std::path::Path::new(&out).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match trainer.save(&out) {
        Ok(()) => println!("saved: {out}"),
        Err(e) => {
            eprintln!("Failed to save model: {e}");
            std::process::exit(5);
        }
    }

    // Append the realized training outcome to the manifest so it is not a separate,
    // driftable document.
    let outcome = format!(
        "\n## Training outcome (this run)\n\n- GPU device: {device:?}\n- Train/val split: {} / {}\n- Epochs run: {epochs}\n- Best validation accuracy: {best_acc:.4} at epoch {best_epoch}\n- Total training wall time: {total_secs:.1}s\n- Saved model: `{out}`\n",
        train.len(),
        val.len()
    );
    if let Err(e) = std::fs::OpenOptions::new()
        .append(true)
        .open(&manifest_path)
        .and_then(|mut f| std::io::Write::write_all(&mut f, outcome.as_bytes()))
    {
        eprintln!("Warning: failed to append training outcome to manifest: {e}");
    }
}
