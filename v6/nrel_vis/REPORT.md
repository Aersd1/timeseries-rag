# V4 analog vs V6 learned RAG

Same NREL store, same test queries, same memory bank.
V4 ranks by z-normalized Euclidean distance on the 244-point history (analog forecasting).
V6 ranks by the learned future-belief embedding (query sees history only).
Neighbors are greedily de-duplicated with start gap >= 244.
Futures are mapped with past-only mean/std scaling.

|Query|sid|start|shared starts|V4 hist-NMSE|V6 hist-NMSE|
|---|---:|---:|---:|---:|---:|
|q01|0|414664|4|2.745|1.823|
|q02|0|428328|0|1.422|0.7648|
|q03|1|477806|0|2.919|4.448|
|q04|1|511478|0|2.344|2.587|
|q05|2|453401|1|0.8597|0.6152|
|q06|2|469505|0|0.2954|0.2217|
