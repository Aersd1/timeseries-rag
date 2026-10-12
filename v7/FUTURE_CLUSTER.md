# Self-trained future-regime retrieval encoder

This experiment removes pretrained Moirai representations entirely. A randomly
initialized, univariate PatchTST history encoder learns a continuous retrieval
vector and, in the cluster variant, a conditional future-regime distribution.
Only observed query history is accepted by the inference API.

## Future supervision

Use all complete training windows at stride 32 within the existing training
split. Normalize their continuations using each observed history's mean and
standard deviation, with the memory-only scale floor. Keep the full ordered
future values relative to the observed history mean at each forecast horizon;
append first differences. Each horizon has equal squared-distance weight.
There is no CDF or future-dependent normalization.

Fit 16 k-means++ prototypes on TRAIN features only, with three seeded restarts
and at most 50 iterations. Soft labels are normalized exponential distances to
the prototypes. The assignment temperature is fixed from training distances.
Validation and test futures do not fit prototypes or their temperature.

Train two controls from scratch with the same windows, seed, batch order,
20-epoch budget, AdamW learning rate 0.0003 and batch size 64:

- `scratch`: future-neighborhood matching, auxiliary future regression and
  vector variance regularization.
- `cluster`: the same objectives plus soft future-cluster classification and
  matching of the predicted regime distribution to historical future keys.

The backbone produces a 32-dimensional normalized retrieval vector. The
classifier predicts 16 probabilities from history; it never reads query future.
Classification supervision and retrieval supervision are distinct from
knowing the actual future at inference.

## No self or overlapping training retrieval

The training candidate bank contains only the earliest memory prefix, sampled
at stride 32. For every query/candidate pair, require the same variable and:

```text
candidate_start + history_length + maximum_horizon <= query_start
```

Both intervals include history and continuation and are half open. Equality
at the boundary is allowed. This excludes the query itself, overlapping
histories, and candidates whose continuation overlaps the query even when
their histories do not. Never silently relax this rule when candidates are
scarce. Loss masks give forbidden candidates zero target and zero gradient.

Each query gets up to 32 hard candidates by history similarity and 32 other
random eligible candidates. True TRAIN futures determine a soft target over
these candidates. Query futures are supervision labels, never retrieval input.
The fixed bank also makes all retrieved candidate data strictly historical.
Candidate-to-candidate overlap is allowed; this experiment does not require
the five selected candidates to represent distinct episodes.

## Retrieval and evaluation

Build a dense final memory bank at stride 1. Compare raw-history nearest
neighbors, scratch vectors, cluster-supervised vectors and future-regime
retrieval. Future-regime retrieval compares the query's predicted probabilities
to the historical candidate's known-future soft labels using squared Hellinger
distance, recalls 50, and reranks with 0.8 future-regime / 0.2 historical-shape
scores after candidate-pool median normalization. Keep five and map their
continuations back to query history scale for weighted analog prediction.
The old historical continuation is available data, not the query's unknown
future. This asymmetry is intentional and is not an oracle forecast.

The main `history_future_cluster` policy first recalls 50 by raw history shape
and then reranks that pool using the future-regime score and historical score.
The `history_future_regression` control uses the same history-only shortlist,
but its future score compares full ordered historical continuations to the
scratch model's predicted continuous future across the configured horizons.
All scores use the same observed-history mean/scale as analog prediction.
The `future_cluster` global-recall policy remains available as a diagnostic.

Select each checkpoint using validation analog NMSE on the coarse memory bank
(cluster model: history shortlist plus regime reranking);
evaluate the selected model on dense memory with the unchanged held-out queries.
Report forecast NMSE and the actual selected candidates' future mismatch
separately. The latter uses test futures for retrospective diagnosis only.
The report independently recomputes metrics, checks all training-pair masks,
reconstructs every indexed window from raw memory and verifies data hashes.
Earlier A/A50 scores are paired by query identity, but their training budgets
were smaller, so they are not budget-matched ablations.

## Run

```powershell
python -m v7.benchmark_future_cluster --source path/to/previous-benchmark --output path/to/fresh-run --device cuda --epochs 20 --clusters 16
python -m v7.verify_future_cluster_query --root path/to/fresh-run --source path/to/previous-benchmark --dataset ETTh1
```

For new history-only queries:

```powershell
python -m v7.query_future_cluster --folder path/to/fresh-run/ETTh1 --store path/to/previous-benchmark/ETTh1/store --history path/to/history.npy --sid 0 --start 15000 --output path/to/fresh-query.json --mode history_future_cluster --device cpu
```

The checkpoint and index have their own family and identity checks and do not
load Moirai weights. Historical shape cannot uniquely determine an ambiguous
future; the predicted regime distribution represents a tendency rather than a
guarantee. Evaluate changes against the frozen test protocol before treating
this experimental encoder as an accuracy improvement.
