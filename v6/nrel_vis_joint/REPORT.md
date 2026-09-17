# V6 joint retrieval case studies

Same NREL store and test queries. History-only example filter; not a benchmark.
V4: z-normalized Euclidean distance on 244-point history.
V6 learned: future-belief embedding.
V6 joint: concatenated learned+history vectors plus raw-history NMSE gate (tau=0.5).
Candidate future NMSE uses query-history scale after past-only mean/std mapping.

|Query|sid|start|V4 analog n|V4 analog histNMSE|V4 analog futNMSE|V4 analog forecast|V6 learned n|V6 learned histNMSE|V6 learned futNMSE|V6 learned forecast|V6 joint n|V6 joint histNMSE|V6 joint futNMSE|V6 joint forecast|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|q01|0|414664|10|0.02263|6.341|2.745|10|0.2342|8.71|2.081|10|0.03418|8.626|2.188|
|q02|0|428328|10|0.169|4.026|1.422|10|0.3285|4.746|1.141|10|0.1827|3.497|0.771|
|q03|1|477806|10|0.5742|5.641|2.919|10|0.9701|43.16|16.58|2|0.4748|5.352|5.34|
|q04|1|494398|10|0.2436|3.476|1.278|10|0.4505|5.749|0.8015|10|0.2625|3.795|1.643|
|q05|2|453401|10|0.0175|1.768|0.8597|10|0.02066|1.922|1.113|10|0.01801|1.862|0.8288|
|q06|2|469505|10|0.01868|1.041|0.2954|10|0.03956|1.151|0.3047|10|0.01954|0.991|0.3041|
