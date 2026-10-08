# A learned recall + history-aware Moirai reranking

This optional pipeline reuses an existing A checkpoint and index. It retrieves
the 50 nearest learned vectors from the same series, scores their historical
continuations against a Moirai forecast of the query, and aggregates the best 5.
Neither model receives the query's actual future during retrieval or reranking.

## Scoring

Each candidate's observed history is normalized by its own mean and standard
deviation, with the existing memory-based scale floor. The query history is
normalized in the same way. Their pointwise mean squared difference is the
history score `H`. This is raw history shape similarity, not learned-vector
distance; learned-vector distance controls first-stage recall.

The candidate's known future is mapped to the query's mean and scale, using
statistics computed from the two histories only. The future score `F` is the
existing B multi-horizon quantile pinball score against Moirai's forecast.

For a user-specified history weight `beta`, the combined score is:

```
S_i = (1 - beta) * F_i / max(median(F), 1e-6)
    + beta       * H_i / max(median(H), 1e-6)
```

Medians are computed over the retrieved candidate pool. Smaller scores rank
first; ties use learned distance and then start position. The top 5 mapped
futures are combined with exponentially decreasing weights based on `S`.
The weight is a coefficient in this normalized score, not a guarantee of a
particular historical error bound or a probability.

History acts as a soft preference. There is no joint-channel history cutoff.
Ordinary nearest-neighbor recall allows overlapping candidates, and the final
top 5 may overlap as well. Therefore 50 candidates need not represent 50
independent historical episodes. Evaluation records the number of disjoint
history-plus-future episodes in each pool. An undersized pool is reported
explicitly; an empty pool falls back to the last observed value.

## Query

From the repository root, with a configured Moirai environment:

```powershell
python -m v7 query-a-rerank --store path/to/store --checkpoint path/to/train/best.pt --index path/to/index_a --history path/to/history.npy --sid 0 --start 10000 --candidates 50 --top-k 5 --history-weight 0.5 --output path/to/query.json
```

`start` is the query history's start offset within the series. It must follow
the index's fixed memory prefix. The history file must contain exactly the
configured number of finite observations. The CLI weight defaults to 0.5;
pass the validation-selected value explicitly when reproducing a benchmark.

## Paired benchmark

```powershell
python -m v7.evaluate_learned_rerank --source path/to/existing-benchmark --output path/to/fresh-results --datasets ETTh1 ETTh2 ETTm1 ETTm2 --device cpu
```

The source contains each dataset's `store`, `config.json`, `train/best.pt`, and
`index_a`. No encoder retraining or index rebuild is necessary.

Before looking at test results, the runner evaluates history weights
`[0.2, 0.35, 0.5, 0.65, 0.8]` on validation. It selects one common weight by
averaging each dataset's NMSE ratio relative to the same-pool original-order
baseline. Datasets have equal weight. It writes and fingerprints `policy.json`
before evaluating test data. Zero-history and history-only controls are
evaluated on test for diagnosis and cannot be selected by this tuning rule.

Outputs include per-query metrics, candidate scores, selected weights, timing,
artifact identities, and aggregate metrics. Compare these controls:

- Original A: default non-overlapping learned top 5.
- Same-pool baseline: first 5 ordinary learned neighbors in the 50-candidate pool.
- Future-only reranking: history weight 0.
- Validation-selected history/future reranking.
- History-only reranking: history weight 1.
- Moirai direct forecast and persistence.

The same-pool baseline distinguishes reranking from the change in candidate
overlap policy. Results do not establish that any representation is inherently
less lossy; model priors, training, candidate diversity, and forecast errors all
affect performance. The runner currently requires a fresh output directory.
