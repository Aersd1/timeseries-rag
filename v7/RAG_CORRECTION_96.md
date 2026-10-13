# Frozen raw Moirai with retrieval correction,96-to96

Keep the96-point ORIGINAL history as the frozen Moirai input. Detrending is
only for matching historical cases. Do not independently forecast a trend and
residual and replace the original foundation prediction with their sum.

The fixed causal25-point operator projects observed histories and the raw
Moirai forecast trajectories into a residual domain. Recall100 by normalized
residual history, rerank with future compatibility/history weights0.8/0.2,
and select5 with a soft overlap penalty. Compatibility is a heuristic score,
not a probability of the true future.

The external15,841-parameter correction head takes200 history/prediction/
quality features,5 residual-history keys and5 paired96-point value vectors.
Attention respects the original ranking weights as a prior. A gated rank16
temporal projection produces a96-point correction, bounded to3 memory stds
when the learned head is used. Its final projection starts at zero, exactly
preserving the raw foundation forecast. It is outside pretrained Moirai.

Two distinct value definitions are explicit experiments:

* `errors`: each old case's actual continuation minus Moirai's prediction
  from THAT old history, scaled by the variable's initial memory std.
* `innovations`: mapped old residual continuation minus the residual obtained
  by projecting the CURRENT raw Moirai prediction. The raw prediction's trend
  is preserved; retrieved residuals propose wave corrections.

Supervision is the current TRAINING query's actual model error. Query truth
never enters input features, candidate scoring or attention. AdamW,lr1e-3,
80epochs,batch64 and correction regularization0.01 are fixed. Feature scaling
uses training features only. Validation chooses the checkpoint and adoption
versus fixed transfer coefficients[0,0.1,0.25,0.5,1]. Zero retains the original
forecast. Test scores cannot choose a checkpoint or coefficient. A validation
fallback cannot guarantee no degradation on future test queries.

Controls include direct raw Moirai, fixed query-matched correction, learned
query-matched correction, global calibration without similarity matching,
and randomly mismatched old value vectors within each variable. Ablations
never select the main method. The global control's value definition follows
the selected mode: mean old model error, or mean eligible normalized old
residual continuation. Report these definitions separately.

## Offline and online memory

The default offline pool is the first25% memory prefix at stride1.
An explicit online experiment uses all episodes available BEFORE EACH query,
with stride16 for computation. Full candidate history+continuation must end
at or before query-history start. No overlap with the query is allowed.

Previous validation/test observations may enter later online queries once
their full episodes are already observed. This is a streaming evaluation,
not an experiment restricted to a fixed training-only index. Pool preloading
is an implementation detail: ineligible rows affect neither matching nor the
global-control average. The initial memory's mean/std stay fixed. Unit tests
alter all values at/after query start and verify identical retrieved inputs
and global-control features when supplied query history is held fixed.

Both memory time coverage and indexing stride change in the online experiment;
their contributions are not isolated. Reusing test periods during method
development makes these results exploratory. Do not promise that RAG
universally improves an already strong foundation model.

```powershell
python -m v7.benchmark_error_correction --output path/to/offline-run
python -m v7.benchmark_error_correction --output path/to/online-run --memory-scope observed --memory-stride 16 --value-mode innovations
python -m v7.verify_error_correction --root path/to/run
python -m v7.query_error_correction --store path/to/store --checkpoint path/to/run/ETTm2/best.pt --history path/to/history96.npy --sid 6 --start 60000 --output path/to/prediction.json
```

Independent verification reconstructs raw data, old model errors or proposal
values, all query boundaries, candidate scores, fixed/global calibration,
training-only feature standardization, adoption decisions and raw metrics.
Paired bootstrap intervals resample synchronized temporal episodes, keeping
the7 correlated variables together. They are exploratory intervals, not a
claim of universal superiority. All weights and experiment data stay local.
