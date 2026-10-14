# Learned fusion of the existing five retrieval cases

This experiment keeps the existing 100-candidate retrieval and reranking, the
96-point context and 96-point horizon. The horizon-preserving experimental variant consumes the original
five selected cases only. It does not retrieve another five raw-history cases.
The previously implemented 5+5 variant remains a separate comparison.

Moirai 2.0 is frozen. A small external neural adapter is trained; this is not
fine-tuning of Moirai's backbone and does not place retrieved values into its
input context. Cached foundation forecasts must match the pinned source model
and data contract before reuse.

## Why replace the earlier fusion

There are two distinct problems: selecting cases whose future is relevant, and
using those futures without overwriting a good forecast. Increasing the recall
count cannot by itself solve either problem. A residual adapter trained against
direct Moirai can also duplicate corrections already made by the existing RAG
forecast. Flattening all old future patches into one attention memory allows a
query horizon to read an unrelated old horizon.

The residual-reference variant uses steps 1–4 and 7 below, with the earlier
flattened patch attention. The further predictive variant also applies steps
5–6 and extended calendar metadata. Both are explicit experiments; the latter
is not automatically promoted to the default based on its theoretical appeal.

1. **Preserve the actual existing five-case forecast.** Let `F5` denote the
   frozen `GuardedBankPredictor` output. The training target is `y - F5`, rather
   than `y - Moirai`. The deployed result is `F5 + beta * delta`.
2. **Fit alignment using observed histories only.** For query history `x` and
   old history `p`, use `a = sigma_x * sigma_p / (sigma_p^2 +
   (sigma_x / 5)^2)` and `b = mean(x) - a * mean(p)`. Apply the same `a,b` to
   old history and its known future. A flat old history therefore cannot create
   a large correction by dividing by its nearly zero standard deviation. The
   legacy bounded scale alignment is evaluated separately.
3. **Retain state before normalization.** Query features include original
   level relative to TRAIN mean, amplitude, last-value displacement, trend,
   day and week phase. Each case also includes its raw level, amplitude,
   displacement, age, history discrepancy, old day/week phase and the relative
   phase to the query. These clock features use the dataset's sampling period;
   they do not infer actual holidays or weather events.
4. **Use a historical prediction probe.** Moirai forecasts the last 32 already
   observed points from the first 64 history points. Its error on those known
   points helps the adapter identify the current forecasting state. This probe
   never requires the current query's future.
5. **Preserve forecast horizon in attention.** The 96-point forecast has twelve
   patches of eight values. Each query patch attends to five old-history keys
   plus a null case. Its values are the *corresponding horizon patch* of each
   old future. For example, the first predicted eight points cannot read the
   final eight points of an old future. Keys use old histories and metadata;
   values use the known old futures. Attention has shape `N x 12 x 6`.
6. **Train a future-matching auxiliary score.** Only on TRAIN, compare each
   aligned old future and the null reference forecast with the query's true
   future. Convert their trajectory errors to a soft teacher distribution. A
   student score computed from current history/state and old history/state
   learns this distribution through KL loss. The same projections contribute
   to the fusion attention. At deployment this score requires no query future.
   This supervises the five-case fusion; it does not replace the upstream
   encoder used to recall the 100 candidates.
7. **Bound and confirm corrections.** The neural output uses `tanh` and a cap
   of the smaller of one quarter of the Moirai forecast quantile width and
   twice the local history scale. Each variable's first two VAL blocks select
   `beta` from `0.1, 0.25, 0.5, 1`; the last block can only confirm or reject
   that choice. Rejection sets `beta=0`, preserving `F5`. This protects against
   validation failures, not all future distribution shifts.

## What is trained

The adapter has 32-dimensional internal representations, separate history and
old-history projections, forecast quantile and future-patch projections,
state/series representations, attention, a gate, and an output head. With the
residual-reference variant it also embeds `F5` and the observed probe error.
The output head starts at zero so the untrained adapter preserves its reference.

The loss is TRAIN-scale weighted squared error plus `0.05 * MAE` and
`0.001 * correction energy`. The predictive variant adds `0.05 * KL`.
Its history-only cosine student scores use temperature `0.1`; the teacher
normalizes trajectory error by the median candidate error, floored at `0.1`.
Input values are clipped to `[-8,8]` after normalization.

AdamW uses learning rate `0.001`, weight decay `0.001`, batch size 256,
up to 48 epochs and patience 10. Epoch selection uses only the first two VAL
blocks. The three fixed seeds are 13, 41 and 73, and their outputs are averaged.
The same seeds, training budget and network capacity are used for the control.
No seed or alignment is chosen by its TEST result.

## Controls and interpretation

- Direct frozen Moirai is the foundation baseline.
- The original five-case forecast is the optimization reference.
- The earlier 5+5 forecast is a separate comparison; it is not an input to the
  final five-case model.
- In the residual-reference control, the new module receives current history,
  foundation quantiles, `F5` and the observed probe, but no additional raw
  retrieved histories/futures. Five identical query/reference pairs keep the
  memory pathway and parameter count active. Since `F5` already uses retrieval,
  label this **no additional case input**, not an entirely retrieval-free model.
- Frozen-model random-case and post-alignment future-pair permutation controls
  probe dependence on selected cases. They are inference stress tests, not
  models retrained with random cases, and do not alone establish causality.

The unconfirmed raw output and the confirmed deployed output are different
experimental results. Report both explicitly. Always report raw MSE, MAE and
TRAIN-standardized MSE: electricity channels have very different scales and
the largest channels can dominate raw MSE. A gain in standardized MSE alone
does not establish that electricity forecasting has improved overall.

## Causal boundaries and limitations

Every old case contains 96 history and 96 future points. Its complete episode
must end at or before the query **history start**, including during TRAIN:

```
case_start + 192 <= min(query_history_start, frozen_memory_limit)
```

TRAIN query episodes must themselves finish inside TRAIN. Validation and TEST
queries follow the stored chronological split. The physical old history/future
values, split boundaries and case uniqueness are audited. The original five
cases can overlap each other; the exclusion is between a retrieved episode
and the current query. The extra five in the old 5+5 variant obey their separate
within-route spacing rule.

Query input construction does not read `raw_y`. Only the TRAIN loss and
validation policy selection receive their respective query targets. Test
policies are saved before TEST feature preparation/inference. Poison tests
replace current and sibling-channel query futures in memory and verify that
the entire deployed retrieval/fusion path is unchanged.

The frozen upstream encoder was fitted on all TRAIN data. Its TRAIN references
are therefore in-sample, not forward out-of-fold predictions. This experiment
does not eliminate that teacher-fitting limitation. The TEST periods have
already been explored in earlier experiments, so these comparisons are
exploratory, not a new blind benchmark. Illness requires a separately designed
small-sample protocol; its original memory cannot support the same 100-candidate
and three-block confirmation budget.

## Files and reproduction

`prepare_case_fusion.py` prepares causal 5+5 features; `--five-only` takes the
same original first five cases. `prepare_case_reference.py` replays actual
five-case reference forecasts and the historical probe. Its `--paired` mode
copies those exact features and enables horizon correspondence, calendar
metadata and predictive supervision. It cannot copy TEST features before the
new validation policy is frozen.

```
python -m v7.prepare_case_fusion --source <frozen-dual-root> --output <ten-root> --datasets <names>
python -m v7.benchmark_case_fusion --root <ten-root> --datasets <names>
python -m v7.prepare_case_fusion --source <ten-root> --output <five-root> --five-only --datasets <names>
python -m v7.benchmark_case_fusion --root <five-root> --datasets <names>
python -m v7.prepare_case_reference --source <five-root> --output <residual-root> --datasets <names>
python -m v7.benchmark_case_fusion --root <residual-root> --datasets <names>
python -m v7.prepare_case_reference --source <residual-root> --output <predictive-root> --paired --datasets <names>
python -m v7.benchmark_case_fusion --root <predictive-root> --datasets <names>
python -m v7.verify_case_fusion --root <predictive-root> --datasets <names>
python -m v7.verify_case_fusion --root <predictive-root> --datasets <names> --live
python -m v7.diagnose_case_fusion --root <predictive-root> --datasets <names>
```

Deployment uses `LearnedCasePredictor` in `query_case_fusion.py`. With a residual
reference run, `forecasts['calibrated_wiener']` is the confirmed primary output.
Model hashes, training statistics, frozen source hashes and policy hashes are
checked before loading. The optional `case_fusion_policy.py` joint alignment
selection is a separate exploratory analysis of the non-reference five-case
run, and is not the default deployment policy.

Data, trained weights, cached predictions and experiment reports stay in local
ignored run directories. This document describes mechanisms and reproduction;
it does not publish experimental data or trained parameters.

## Paper connections

- [Align-RAG](https://arxiv.org/html/2608.05571v1): history-fitted Wiener
  alignment. Its full context packing and in-context Moirai mechanism are not
  reproduced here.
- [Cross-RAG](https://arxiv.org/html/2603.14709v2): query/history-key/future-value
  cross-attention. Its Chronos-based architecture differs from this external
  Moirai adapter.
- [PFRP](https://ojs.aaai.org/index.php/AAAI/article/view/39230): predictive
  history representation learning and confidence-aware retrieval. Its main
  evaluation setting is not the same all-channel benchmark used here.
- [TimeRAF](https://arxiv.org/html/2412.20810v1): forecasting-informed retrieval
  supervision. The auxiliary KL loss here supervises five-case fusion, not
  TimeRAF's full retriever/training framework.
