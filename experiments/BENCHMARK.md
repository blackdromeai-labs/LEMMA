# Neuro-symbolic search benchmark

This package performs a matched comparison between LEMMA's untrained uniform-prior MCTS and
the saved trained policy. It uses exact-goal `NeuralMCTS::search`, not `simplify`, so procedural
fallbacks cannot hide whether search reached the target.

## Corpus

- 400 deterministic controlled-synthetic problems
- ID and zero-label-family OOD splits reported separately
- construction depths 2, 3, 4, 6, and 8
- 40 problems in every split/depth cell
- five generator families in each split
- 1,840 recorded reference transformations

`construction_depth` is the length of the recorded valid path, not a claim of globally minimum
distance. The generator constructs expected results analytically and records the intended path;
the validator then independently checks that every named registry rule reproduces that path and
that the verifier accepts each transition.

Validation is fail-closed for JSON parsing, expression parsing and exact formatting round trips,
duplicate IDs or exact pairs, split/rule-label violations, missing or ambiguous rule applications,
verifier rejection, and reference paths that do not reach the expected expression.

## Reproduce

Build CUDA binaries from a Visual Studio developer shell on Windows:

```powershell
cargo build -p mm-solver --example neurosymbolic_benchmark --release --features cuda
```

Regenerate and validate the full corpus:

```powershell
.\target\release\examples\neurosymbolic_benchmark.exe generate
.\target\release\examples\neurosymbolic_benchmark.exe validate
```

Run both complete arms at a matched 150-simulation budget:

```powershell
.\target\release\examples\neurosymbolic_benchmark.exe run `
  experiments\corpus\problems.jsonl `
  experiments\models\policy.safetensors 150
```

## First locked run

| Arm | Solved | ID | OOD | Runtime |
|---|---:|---:|---:|---:|
| Uniform priors | 400/400 | 200/200 | 200/200 | 341 ms |
| Trained policy | 272/400 | 72/200 | 200/200 | 130,974 ms |

The trained ID result by construction depth was 32/40, 16/40, 8/40, 8/40, and 8/40. Every OOD
cell was 40/40 in both arms.

This run establishes that the loaded policy materially changes search, but it does not improve
it: it reduced solve rate and increased runtime. The corpus also fails the specification's desired
difficulty condition because uniform search saturated at 100%. These results must not be presented
as evidence that neural guidance improves LEMMA. The benchmark is useful as a reproducible
negative-result and diagnostic corpus; a future performance claim requires a separately designed,
pre-frozen corpus with substantially greater measured branching.

## SHA-256

```text
problems.jsonl                  DE8D484F411486739E3E39312FF135743C7FFCE1CDA662ACDD10AF514054A832
policy.safetensors              93ED585ADB14D913E6DB5B91D84A1DDAA518A23EF4F417BDCF9E6654C1CC9266
policy.manifest.json            64EA092B15607993008F72D728B65C8559FBDC452B0EC6F3AACAE3ECE7B447EA
trained_vs_uniform.json         BB7A8745BB3DC2856E08A2FCAAE85815FC5C75398095FF8827E9B9A211531AFD
neurosymbolic_benchmark.rs      1198B89B686635B56A590A320F4ED0FD4680C84239DD8396D5CCF1BCE9672EF0
```

## Compositional retraining (second locked run)

One retraining pass, run once and reported as-is, per an explicit instruction not to iterate
against this locked corpus. `problems.jsonl` was read only to run search and to replay recorded
reference paths for the prior-rank diagnostic below -- not touched, not used to derive training
data. The new training corpus (`experiments/COMPOSITIONAL_DATA_MANIFEST.md`) is built entirely
from rule definitions in `mm-rules` and validated against the real rule/verifier before any
example is kept; see that file for exactly what it contains and its known coverage gaps
(several calculus derivative rules collapsed to 1-7 kept examples after deduplication).

Three arms, all against the same unchanged 400-problem corpus:

- **uniform** -- no policy network (`NeuralMCTS::new`, random/flat priors)
- **shallow** -- the first locked run's model, `experiments/models/policy.safetensors`,
  trained on `mm_brain::data`'s flat, single-level, unvalidated-label templates
- **compositional** -- `experiments/models/policy_compositional.safetensors`, trained on
  `mm_brain::data_compositional`'s nested, compound-operand, validated-label templates
  (1,763 examples, same network architecture and hyperparameters as `shallow`)

### Fixed budget-curve solve counts (out of 200 per split)

| budget | uniform ID | uniform OOD | shallow ID | shallow OOD | compositional ID | compositional OOD |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 40 | 40 | 32 | 40 | 33 | 38 |
| 5 | 100 | 112 | 56 | 120 | 62 | 108 |
| 10 | 122 | 184 | 72 | 200 | 84 | 185 |
| 25 | 180 | 184 | 72 | 200 | 104 | 196 |
| 50 | 200 | 192 | 72 | 200 | 113 | 196 |
| 150 | 200 | 200 | 72 | 200 | 133 | 197 |

GPU wall-clock for the full 400-problem corpus at budget 150: uniform 282 ms (does not call the
network), shallow 136,106 ms, compositional 106,078 ms. Full per-cell node counts and
steps-to-solution are in `experiments/results/three_arm_comparison.json`.

### Prior-rank-of-correct-action diagnostic (budget-independent; 920 reference-path decision points per split per arm)

| arm | split | mean rank | median rank | top-1 rate | top-3 rate |
|---|---|---:|---:|---:|---:|
| shallow | ID | 1.38 | 1.0 | 61.6% | 100% |
| shallow | OOD | 1.04 | 1.0 | 95.7% | 100% |
| compositional | ID | 1.36 | 1.0 | 63.9% | 100% |
| compositional | OOD | 1.08 | 1.0 | 92.4% | 100% |

This metric replays `NeuralMCTS::expand`'s exact legal-move filter at every state along every
recorded reference path and asks where the policy ranks the historically-correct next rule among
the priors of the rules that are actually legal there -- it isolates policy quality from search
variance.

### Reading the result honestly

**The shallow model's ID solve rate is flat at 72/200 from budget 10 through budget 150** --
more search budget buys it nothing. This matches the diagnosis in `AGENT_MEMORY.md` ("E1 ran,
root cause found"): a confidently-wrong prior at certain compositional states (e.g.
`identity_mul_one` vs. `distribute` on `1 * (A + B)` shapes) starves the correct branch of
simulation budget regardless of how much budget is available.

**The compositional model does not have this ceiling.** Its ID solve rate keeps climbing with
budget (84 -> 104 -> 113 -> 133 across budgets 10/25/50/150) instead of plateauing, and it
roughly doubles the shallow model's ID solve rate at matched budget 150 (133/200 vs. 72/200).
The prior-rank diagnostic shows why this is plausible without overclaiming: ID top-1 rate only
moved modestly (61.6% -> 63.9%) -- the compositional data did not "fix" the policy's judgment in
any dramatic sense. What changed is that being wrong less often, and specifically being wrong in
ways that do not permanently trap search the way the shallow model's confident errors did, is
enough for search efficiency to keep improving with budget instead of stalling.

**Compositional training is NOT a fix, and does not beat uniform.** Uniform reaches 100% ID by
budget 50 and stays there; compositional reaches only 133/200 (66.5%) even at budget 150 -- the
largest budget tested. Compositional is not "as good as uniform, just slower": at matched budget
it solves strictly fewer problems, on top of costing real GPU time uniform does not spend at all.

**OOD moved slightly the wrong way.** Shallow reaches 200/200 OOD from budget 10 onward;
compositional peaks at 197/200. The prior-rank diagnostic shows the same direction: OOD top-1
dropped from 95.7% (shallow) to 92.4% (compositional). This is a small, plausible fit trade-off
(more capacity spent discriminating ID-family collision cases, marginally less spent on the
OOD-family wrapper patterns) -- not something this run investigated further, per the instruction
to complete one retraining-and-comparison pass and stop.

**Net assessment:** compositional retraining produced a real, measured improvement in search
efficiency on the specific pathology it targeted (removing the budget-insensitivity of a
confidently-wrong policy), at the cost of a small OOD regression, while remaining substantially
below uniform-prior search on this corpus. This is reported as the outcome of the single
retraining pass that was run; no further tuning against this locked corpus followed.

### SHA-256 (compositional retraining artifacts)

```text
policy_compositional.safetensors        C0D938835945CD9117EA0FA6D0F372C45552FE475889273A8249BCC667159336
policy_compositional.manifest.json      64EA092B15607993008F72D728B65C8559FBDC452B0EC6F3AACAE3ECE7B447EA
COMPOSITIONAL_DATA_MANIFEST.md          8470A801E89A96D6A240701BD73614E965C4D75DB5D795838E434AACBE877828
three_arm_comparison.json               B0428E954A35258F9DF1315BF33E3901D908DFA30B49828FC5738A526B78D927
data_compositional.rs                   97ADC1AEFAABE722D9715B5B2B4C9CD3FDFCE47E92739737202CA932AFE07C95
train_compositional_policy.rs           2B02B78E094241187796A9DD8894A8E22F287431F92C0762EB7182A9C7F48D4D
compare_policies.rs                     95E9919E55D6454A866E3B90A875FE34163E732430B3215C06659FE1A7E594C7
```

`policy_compositional.manifest.json` hashes identically to `policy.manifest.json`: both models
share the same network shape, vocabulary, and `max_seq_len`, and the manifest records only that
metadata, not weights -- it is not evidence the two models are the same.
