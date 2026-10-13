# History96 / horizon96: trend and residual Moirai

This experiment decomposes each observed window into a trend and residual,
retrieves only residual episodes, and conditions the residual Moirai branch
inside its computation. It does not mix final Moirai and analog forecasts.

## Decomposition

The fixed trend operator is a trailing25-point moving average, left padded
with the first observed value. Residual = original value minus this trend.
Every historical feature depends exclusively on observed history. For old
episodes and training loss targets, the same causal operator is extended
through the already observed continuation. Changing a query continuation
cannot change any of its history features.

Twenty-five samples represent different physical durations in hourly and
quarter-hourly data. This operator is an explicit initial definition of trend;
it is not a recovered ground-truth trend, and the smoothing width is not
selected using test data.

## Retrieval

The memory is the first25% of each original series. Its full192-point episodes
are decomposed separately, using their96-point observed histories for scaling.
Both query and candidate residual histories use history-only mean/std
normalization. The std floor is1% of the memory's original-value standard
deviation, never a held-out statistic.

Recall100 candidates using exact normalized residual-history squared distance.
The prior A encoder was trained on244-point original histories, so it is not
reused for these96-point residual inputs. Candidate ranking combines0.8
frozen residual-Moirai quantile pinball compatibility and0.2 history error,
each divided by its median within the candidate pool. Select5 using a fixed
soft overlap penalty of1, retaining base-score aggregation weights.

Every candidate's history AND continuation must end before the query history
begins, including training queries. Candidates can overlap each other, but a
candidate cannot overlap the query. No query future enters recall or ranking.

## Two prediction branches

1. Frozen Moirai2 predicts96 future trend samples from96 trend-history samples.
2. A residual Moirai2 predicts96 residual samples from96 residual-history
   samples. Its last two Transformer layers have rank8 query/value LoRA.
   Paired retrieval attention consumes the5 known old residual episodes:
   old histories produce keys and their paired continuations produce values.
   The current hidden representation produces queries. A learned gate controls
   the update injected into the Moirai encoder output before the forecast head.
3. Undo residual history normalization and add the two predicted components.

There is no final-prediction interpolation alpha. Retrieved values influence
the residual model's internal hidden representation. The retrieval output
projection starts at zero, preserving frozen residual forecasts at epoch0.

Train the residual branch using residual quantile pinball loss, AdamW with
learning rate1e-4,8epochs, batch32, candidate and whole-memory dropout0.1.
The trend branch remains frozen. Select the residual checkpoint on validation
NMSE of the recomposed raw forecast, including epoch0. Test scores cannot
change this choice. One fixed seed and up to256 training queries per variable are
used; no extra training instances are silently added after viewing results.

## Controls and evaluation

All methods use the same96-point raw histories and96-point true continuations.
All seven numeric ETT columns are included as univariate queries. Existing
25/60/80% memory/train/validation boundaries are preserved. Validation/test
contain every complete192-point nonoverlapping episode in their intervals.
These are custom chronological splits, not the published ETT evaluation split.

Compare frozen raw Moirai, frozen decomposed Moirai, independently LoRA-trained
decomposed Moirai without memory, and internally conditioned residual RAG.
Also disable memory in the SAME trained RAG model as an inference ablation.
Use raw MSE/MAE and per-variable memory-variance-normalized NMSE. The latter
prevents variables with larger physical units from dominating the headline.
Bootstrap synchronized temporal episodes for paired gain intervals, keeping
correlated variables in the same block.

```powershell
python -m v7.benchmark_trend_rag --output path/to/fresh-run --device cuda
python -m v7.verify_trend_rag --root path/to/fresh-run
python -m v7.query_trend_rag --store path/to/store --checkpoint path/to/fresh-run/ETTh1/rag/best.pt --history path/to/history96.npy --sid 0 --start 15000 --output path/to/prediction.json
```

The query command accepts historical values only. Independent audit checks
reconstruct causal signals from raw values using convolution, inspect every
candidate chronology boundary, rebuild selected residual pairs and score
arrays, recompute paired raw metrics, and verify validation-selected epochs.
Experiment data and model weights remain local; only code is published.
