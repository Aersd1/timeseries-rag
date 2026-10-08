# Moirai variants: paired evaluation

Split: test; queries: 64.

NMSE divides by memory-only variance. These means alone do not establish general superiority.

|H|Method|N|Mean NMSE|Win rate vs persistence|
|---:|---|---:|---:|---:|
|24|b_reranked|64|0.109156|0.422|
|24|b_v4_original_order|64|0.176634|0.188|
|24|moirai_direct|64|0.105353|0.516|
|24|persistence|64|0.0995702|0.000|
|96|b_reranked|64|0.394366|0.469|
|96|b_v4_original_order|64|0.410251|0.469|
|96|moirai_direct|64|0.431672|0.375|
|96|persistence|64|0.400458|0.000|
|244|b_reranked|64|0.842332|0.594|
|244|b_v4_original_order|64|0.844207|0.609|
|244|moirai_direct|64|1.00371|0.469|
|244|persistence|64|1.02961|0.000|

B original-order and reranked forecasts use the same candidate pool, mapping and K.
The Moirai direct baseline uses its median. No query future enters either retrieval path.
See summary.json for paired changes, per-query outputs for failures/underfill, and run.json for identities.
Model/index initialization is excluded from reported query latency; this is not a strict latency guarantee.
