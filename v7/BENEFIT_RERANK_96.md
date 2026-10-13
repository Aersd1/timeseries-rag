# 96-point utility reranking with guarded validation

The frozen Moirai input and prediction stay raw. Recall100 candidates from
the unchanged residual-history pool. Instead of learning a large temporal
transformation after selecting5, learn which of the100 corrections is useful.
This model is an external learned reranker, not a modified Moirai encoder.

The per-candidate TRAINING label is the inner product between the query's
actual foundation-model error and that candidate's correction vector:

```text
c = mean(query_error * candidate_correction)
predicted_gain(beta) =2*beta*predicted_c - beta**2*mean(candidate_correction**2)
```

Query truth is used ONLY in training labels or validation/test losses. It is
absent from pair-feature and live-inference APIs. Known old continuations are
allowed only when their full192-point episodes end before query-history start.
Train on all100 positive/negative candidate alternatives, not only five cases
that the previous compatibility score already preferred.

The59 features retain raw level, scale, trend differences and elapsed daily
phase, alongside six ordered16-point blocks of query/candidate residuals,
foundation predictions, correction profiles and interactions. This restores
state information lost by residual-only matching. Input normalization and
clipping bounds for supervised labels come exclusively from training pairs.

A regularized linear covariance model has60 coefficients. Select its penalty
from[0.01,0.1,1,10] using two forward folds WITHIN the training interval. Purge
every fitted query whose complete history+target ends after heldout-history
start, including across correlated variables. The actual validation interval
is not used to fit or select these coefficients.

Rank expected MSE benefit at beta0.25. The0.8 future/0.2 history compatibility
score remains a small prior, with a soft overlap penalty. Aggregate the five
selected correction vectors using utility weights; reject estimated harmful
corrections. This regression estimate is not a calibrated future-success
probability.

Calibrate beta per variable from[0,0.1,0.25,0.5]. Split validation episodes into
three chronological blocks. Choose a nonzero coefficient on the FIRST TWO
blocks only, requiring improvement in both and at least1% average improvement.
The THIRD block can only confirm it or revert to beta0. It cannot choose a
different coefficient. Fewer than three episodes in any block force fallback.
Fallback is exactly direct Moirai, not evidence that retrieval is useful.

Compare direct Moirai, the unchanged ranking with this SAME guarded policy,
utility ranking with guarded policy, fixed-quarter utility correction, and
mismatched correction vectors. Test metrics cannot alter the adopted policy.
Use both preexisting fixed-prefix and online candidate caches as separately
labeled experiments. This changes no recall count or candidate-pool protocol.

```powershell
python -m v7.benchmark_benefit_rerank --output path/to/fresh-run
python -m v7.verify_benefit_rerank --root path/to/fresh-run
python -m v7.verify_benefit_query --root path/to/fresh-run
python -m v7.query_benefit_rerank --store path/to/store --checkpoint-folder path/to/fresh-run/cache-name/ETTm1 --history path/to/history96.npy --sid 0 --start 60000 --output path/to/prediction.json
```

The pinned repository Moirai2 checkpoint is required for live inference.
Independent audit reconstructs all100 candidate inputs, unchanged source
artifacts, purged training boundaries, train-only model coefficients, the
three-block policy and raw metrics. Reload replay verifies predictions and
selected case identities from observed history only.

Temporal validation checks reduce selection instability; they do not guarantee
positive gains on every future episode. These experiments reuse earlier test
periods during development and remain exploratory, single-seed evaluations.
No claim of independent blind evaluation or universal superiority is warranted.
