# Moirai 2 fine-tuning with retrieved examples

The model predicts the query's future itself. The five reranked historical
examples are optional external memory inside the forecasting model, rather
than a replacement forecast obtained by averaging examples.

## Fixed retrieval pipeline

1. Freeze the existing A learned encoder and its historical index.
2. Recall 50 ordinary nearest neighbors from the same variable. Overlap is
   allowed. Every candidate's history and continuation must end within the
   fixed memory prefix and before the query history begins.
3. Freeze the pretrained Moirai used to score candidates. Combine future
   compatibility and raw-history shape scores with weights 0.8 and 0.2,
   respectively, using the established candidate-pool median normalization.
4. Select five candidates and retain their score-derived aggregation weights.
5. Cache these results separately for training, validation, and test. The query
   future is a supervision/evaluation target only, never a retrieval input.

The retrieval model is not updated during Moirai fine-tuning. This keeps the
candidate selection policy identical across forecasting-model controls.

## How memory enters Moirai

Normalize each example's history and continuation using that example's observed
history mean and standard deviation with the existing scale floor. No
continuation statistics are used in normalization. Preserve all five examples.

Divide both history and continuation into patches. A learned projection embeds
the values and padding indicators. Learned position and history/future phase
embeddings identify which part of the old example each token describes. Moirai
encoder representations attend to these memory tokens before the pretrained
output projection. The attention logits include a log-weight prior from the
reranking scores. The attention output is a residual update to Moirai's hidden
states; its final projection starts at zero, preserving the pretrained forecast
before adaptation.

This is an added retrieval-attention mechanism, not an upstream feature of
Moirai 2. Candidate continuations are already-observed historical data, not
future covariates from the current query's unknown answer.

## Fine-tuning and controls

Keep pretrained base weights frozen. Add rank-8 LoRA to the query and value
projections of the last two Moirai Transformer layers. Compare:

- Frozen pretrained Moirai (zero shot).
- Moirai with LoRA and no retrieval memory.
- Moirai with the same LoRA configuration plus retrieval attention.
- The trained retrieval model with memory disabled at evaluation, as a
  diagnostic of reliance on the memory path.

Train with query-scale-normalized pinball loss, averaged equally over the
configured forecast horizons. Use the official recursive quantile forecasting
path for both training and evaluation. Its in-place trajectory updates run
inside PyTorch's saved-tensor mutation context during backpropagation; tensors
needed for gradients are cloned when modified without changing forward values.

Both fine-tuned variants use the same seed, sampled training windows, batch
order, learning rate, epoch budget, and validation queries. Select checkpoints
by validation NMSE only; the unmodified pretrained initialization is also an
eligible checkpoint. A best epoch of zero must be reported as no adopted update.
The retrieval variant has additional trainable attention parameters, so this
control is not parameter-count matched. Only adapter weights are saved; loading
requires the pinned original pretrained model as well.

## Reproduce

Run from the repository root in the configured Moirai environment:

```powershell
python -m v7.benchmark_retrieval_finetune --source path/to/benchmark --output path/to/new-run --datasets ETTh1 ETTh2 ETTm1 ETTm2 --device cuda --epochs 8 --batch-size 32
```

The complete runner prepares memory, trains both controls, and audits the
comparison. To execute the stages separately instead:

```powershell
python -m v7.prepare_retrieval_memory --source path/to/benchmark --output path/to/new-run --datasets ETTh1 ETTh2 ETTm1 ETTm2 --device cpu
python -m v7.train_retrieval_finetune --root path/to/new-run --source path/to/benchmark --dataset ETTh1 --variant plain --device cuda --epochs 8 --batch-size 32
python -m v7.train_retrieval_finetune --root path/to/new-run --source path/to/benchmark --dataset ETTh1 --variant rag --device cuda --epochs 8 --batch-size 32
```

Repeat the two training commands for the other datasets. Use fresh output
directories. The memory manifest records the immutable retrieval artifacts,
score policy, window counts, and cache hashes. Each model run records its
training configuration, selected checkpoint, parameter count, held-out metrics,
and predictions. Test labels never select weights, epoch counts, or adapters.

For a new query, supply the selected adapter checkpoint, its neighboring
completed `run.json`, and the original A checkpoint/index:

```powershell
python -m v7.query_retrieval_finetune --store path/to/store --checkpoint-a path/to/a/best.pt --index-a path/to/index_a --checkpoint path/to/new-run/ETTh1/rag/best.pt --history path/to/history.npy --sid 0 --start 10000 --output path/to/new-query.json --device cuda
```

The query command accepts observed history only. It loads pinned pretrained
Moirai weights plus the selected adaptation, independently computes the frozen
top-five retrieval memory, and returns the model's quantile forecast and the
actual historical candidates used.
