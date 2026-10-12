# Improve original A + Moirai with candidate reliability

The original trained A encoder, its stride-1 memory index and pretrained
Moirai stay frozen. Ordinary recall now defaults to 100 candidates, with
future/history ranking weights fixed at 0.8/0.2 and five selected examples.
Earlier A50 evaluation and fine-tuned-model commands explicitly retain their
original 50-candidate policy for checkpoint and benchmark compatibility.

Increasing recall alone is a control. The main mechanism preserves Moirai as
a predictor and uses historical continuations as a calibrated correction:

```text
forecast = Moirai_forecast + alpha(query_quality) * (analog_forecast - Moirai_forecast)
```

The 0.8/0.2 values rank candidates; alpha controls final prediction fusion.
Alpha is an auxiliary-information weight, not a calibrated correctness
probability. Zero gives Moirai alone and one gives the weighted analog alone.

## Selection and calibration

The existing multi-horizon quantile pinball score and raw-history NMSE remain
unchanged. A greedy soft diversity penalty discourages selecting nearly
identical overlapping historical episodes. Overlap uses the full history plus
maximum continuation horizon. It does not permit query/candidate overlap.
The penalty only changes selection; aggregation uses base compatibility scores.

Select one shared diversity strength from [0, 0.1, 0.25, 0.5, 1] on validation.
Then select per-dataset fixed analog alpha from [0, 0.25, 0.5, 0.75, 1]. These
choices are fixed before any final test evaluation. The no-penalty choice and
pure Moirai/analog endpoints are explicit controls.

Train a 113-parameter reliability MLP per dataset using the original sampled
training queries. Twelve input features use only observed query history,
Moirai predictions and known historical examples: historical match error,
pinball compatibility, continuation disagreement, quantile interval width,
model/analog disagreement, candidate weight entropy, episode overlap, learned
distance, observed trend/end level, scale ratio and candidate score spread.
Standardization statistics come from training features only. Targets are the
training queries' true futures and affect the loss only, never features.

Use validation NMSE to choose the gate checkpoint over a 200-epoch budget.
Validation also decides whether to use the learned gate or the fixed-alpha
control. Test results cannot override that choice. An epoch-zero gate means
no trained update was adopted. All valid training/validation/test candidate
histories AND their continuations must end before the query begins.

## Run and reproduce

```powershell
python -m v7.benchmark_a100 --source path/to/benchmark --output path/to/fresh-run --device cpu --calibration-stride 128
```

Dense validation queries use only the pre-test validation interval. Training
history encoding and forecasts run in batches; original test queries run
individually to reproduce the existing A50 output. Both validation and test
use the same dense memory index. A50 is reconstructed from the first 50 of
the exact top100 and its score normalization is recomputed within that pool.

An optional `--reuse-root` reuses checked immutable training/test forecast
caches from a prior A100 run. It never reuses gate-selection outcomes.
Recompute validation, select policy and train fresh gates before testing.

The audit reconstructs query/candidate raw values, affine mappings, history
scores and quantile compatibility scores. It verifies all chronological
boundaries, artifact hashes and old A50 reproduction, and independently
recomputes saved forecast metrics. Reports expose gate adoption and alpha
values so Moirai-only or constant-alpha gains are not attributed to learned
query-dependent selection.

For history-only production inference, use the validated gate checkpoint:

```powershell
python -m v7 query-a-rerank --store path/to/store --checkpoint path/to/a/best.pt --index path/to/index_a --history path/to/history.npy --sid 0 --start 15000 --quality-gate path/to/fresh-run/ETTh1/quality_gate.pt --output path/to/new-prediction.json
```

The gate command uses its saved diversity/fusion policy, and verifies the
exact A/store/index identities. Omitting `--quality-gate` runs ordinary A100
reranking. Standalone inference is also available as `v7.query_a100_quality`.

```powershell
python -m v7.verify_a100_query --root path/to/fresh-run --source path/to/benchmark --dataset ETTm1
python -m v7.evaluate_a100_holdout --root path/to/fresh-run --source path/to/benchmark
```

The second command evaluates remaining full test episodes absent from the
original query list. They are disjoint from the prior episodes on the same
window lattice. Checkpoints and policies stay immutable throughout this
additional evaluation; its results do not select hyperparameters. Single-seed
development results remain exploratory and should not imply guaranteed gains
on every unseen dataset.
