# Datasets

This directory tracks *what data this project uses and where it came from* — not the data
itself, where that data isn't ours to redistribute.

## `manifests/`

One JSON file per external dataset previously committed directly to this repository (exam
papers, mostly). Each manifest records `source`, `license`, a `sha256` of the file that was
removed (so you can verify a re-obtained copy matches what was actually used), and retrieval
instructions. None of the source material itself — PDFs, extracted text — is committed here;
board and entrance exams are not under a license that permits redistribution, and committing
large binary exam papers to git history was never good hygiene regardless.

## `samples/`

Small, freely-redistributable illustrative data (a handful of example problems, not a full
corpus) — for when a manifest-only pointer isn't enough to understand the shape of the data
without going and fetching it. Empty for now: none of the datasets removed from the repo root
had material that's safe to excerpt here. Add real samples under a license permitting
redistribution as they become available, one subdirectory per dataset, named to match its
manifest.

## Training data vs. evaluation corpora

These are kept separate on purpose. Evaluation corpora (what `mm-solver/tests/evaluation.rs`
and the seeded harness check against) are exact, hand-verified, and ground truth by
construction. Training data (substitution-hint labels, topic-classification examples) is
heuristic and never used to grade correctness. Mixing the two under one directory made it easy
to accidentally validate a change against the same data used to shape it. See
`tools/training/` for what still consumes training data, and `mm-solver/tests/evaluation.rs`
for the evaluation side.
