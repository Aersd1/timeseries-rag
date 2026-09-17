# V6 results

Split: test; queries: 64; checkpoint epoch: 13.

## Forecast (lower NMSE is better)

|H|Method|N|Mean NMSE|Skill vs persistence|Win rate|
|---:|---|---:|---:|---:|---:|
|24|encoder_forecast|64|0.10025|-0.007|0.531|
|96|encoder_forecast|64|0.36052|0.100|0.641|
|244|encoder_forecast|64|0.90567|0.120|0.750|
|24|joint_leaves0|64|0.1274|-0.280|0.359|
|96|joint_leaves0|64|0.40098|-0.001|0.453|
|244|joint_leaves0|64|0.79874|0.224|0.547|
|24|learned_leaves0|64|0.12484|-0.254|0.328|
|96|learned_leaves0|64|0.54944|-0.372|0.469|
|244|learned_leaves0|64|1.0767|-0.046|0.531|
|24|persistence|64|0.09957|0.000|0.000|
|96|persistence|64|0.40046|0.000|0.000|
|244|persistence|64|1.0296|0.000|0.000|
|24|probabilistic_prior|64|0.10091|-0.013|0.375|
|96|probabilistic_prior|64|0.39362|0.017|0.672|
|244|probabilistic_prior|64|0.99874|0.030|0.656|

NMSE uses memory-only series variance. `history_nmse` is separately retained. Every method has paired persistence values.
Audit methods run on a smaller, seeded subset: compare them on matching query IDs, not against means over all queries.

## End-to-end query latency

|Method|p50 ms|p95 ms|p99 ms|max ms|≤100ms|complete top-k|scored fraction|
|---|---:|---:|---:|---:|---:|---:|---:|
|joint_leaves0|44.14|110.97|150.40|155.11|0.938|0.938|0.7353|
|learned_leaves0|22.13|77.08|108.37|128.91|0.984|1.000|0.2966|

## Probability calibration

|H|CRPS (normalized)|Marginal NLL|90% interval coverage|Interval width|
|---:|---:|---:|---:|---:|
|24|0.32013|0.71201|0.961|2.684|
|96|0.78479|1.4928|0.939|5.260|
|244|1.3148|2.0124|0.929|7.606|

## Retrieved histories AND futures (per-candidate errors)

|Method|Mean history NMSE|Maximum history NMSE|Mean future NMSE|Queries without matches|
|---|---:|---:|---:|---:|
|learned_leaves0|0.45823|2.4317|16.565|0|
|joint_leaves0|0.20686|0.49958|12.214|0|

Candidate future errors use query-history scale and are not the same metric as memory-normalized aggregate forecast errors. Missing matches are counted separately.

## Interpretation

- Evidence for future-aware retrieval requires learned retrieval to beat history retrieval AND persistence on paired test queries; use the no-belief ablation too.
- A calibrated 90% interval should have coverage near 0.9 without excessive width. This does not imply retrieval recall is 90%.
- `future_oracle_id_recall` compares exact starts in the indexed, possibly subsampled library; ties can make this metric pessimistic.
- Greedy non-overlap uses an oversampled bounded heap. Underfilled top-k is explicitly reported, never silently treated as full recall.
- `certified` concerns stored embedding distances only. Budgeted early stopping is approximate. True-future similarity is not guaranteed.
- Bootstrap intervals cluster by series; few series give weak uncertainty estimates. Cross-series calendar alignment is not verified.
- No 10B-point or <100ms claim follows from a small run. Timings exclude process/model/index object loading; first query is retained separately.
- Checkpoint epoch -1 means an untrained smoke fixture, NOT an experiment.

![Overview](overview.png)
