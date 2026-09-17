# Direct PatchTST vs RAG — z-space MSE

Same 64 test queries. Every method is mapped to raw power, then scored as
`mean(((ŷ-μ)/σ - (y-μ)/σ)²)` with `σ = max(std(query history), 0.1·memory_std)`.
This is the same normalized space PatchTST trains in and analog mapping uses.

## MSE (normalized / σ space)

|H|Method|Mean MSE_z|Skill vs persistence|Win vs persistence|
|---:|---|---:|---:|---:|
|24|persistence|0.52204|0.000|0.000|
|24|v4_analog|0.74429|-0.426|0.234|
|24|patchtst_direct|0.59632|-0.142|0.281|
|24|patchtst_encoder|0.52025|0.003|0.531|
|24|patchtst_prior|0.52542|-0.006|0.484|
|24|patchtst_joint|0.59538|-0.140|0.328|
|24|patchtst_learned|0.66551|-0.275|0.328|
|96|persistence|4.4555|0.000|0.000|
|96|v4_analog|3.2181|0.278|0.469|
|96|patchtst_direct|3.9358|0.117|0.453|
|96|patchtst_encoder|4.3293|0.028|0.672|
|96|patchtst_prior|4.4279|0.006|0.406|
|96|patchtst_joint|3.1031|0.304|0.453|
|96|patchtst_learned|2.7217|0.389|0.500|
|244|persistence|10.881|0.000|0.000|
|244|v4_analog|7.0116|0.356|0.609|
|244|patchtst_direct|8.1176|0.254|0.625|
|244|patchtst_encoder|10.517|0.033|0.750|
|244|patchtst_prior|10.936|-0.005|0.484|
|244|patchtst_joint|6.7317|0.381|0.562|
|244|patchtst_learned|6.6258|0.391|0.609|

## Paired Δ MSE_z (positive = left better)

|H|Method vs baseline|Δ MSE_z|95% CI|Win rate|
|---:|---|---:|---|---:|
|24|patchtst_direct vs persistence|-0.074287|[-0.097082, -0.051491]|0.281|
|24|patchtst_direct vs patchtst_joint|-0.00094731|[-0.068493, 0.070187]|0.391|
|24|patchtst_direct vs patchtst_learned|0.06919|[-0.13222, 0.30937]|0.406|
|24|patchtst_direct vs v4_analog|0.14797|[0.099373, 0.22454]|0.500|
|24|patchtst_direct vs patchtst_encoder|-0.07607|[-0.10609, -0.048737]|0.234|
|24|patchtst_joint vs persistence|-0.073339|[-0.1489, 0.0041179]|0.328|
|24|patchtst_learned vs persistence|-0.14348|[-0.36763, 0.070762]|0.328|
|96|patchtst_direct vs persistence|0.51969|[0.025985, 1.2205]|0.453|
|96|patchtst_direct vs patchtst_joint|-0.83278|[-1.9555, -0.1468]|0.406|
|96|patchtst_direct vs patchtst_learned|-1.2142|[-3.6031, 0.02367]|0.500|
|96|patchtst_direct vs v4_analog|-0.71774|[-1.8596, -0.1323]|0.500|
|96|patchtst_direct vs patchtst_encoder|0.39342|[-0.15514, 1.0055]|0.422|
|96|patchtst_joint vs persistence|1.3525|[0.12247, 3.1175]|0.453|
|96|patchtst_learned vs persistence|1.7339|[-0.050009, 4.8236]|0.500|
|244|patchtst_direct vs persistence|2.7633|[0.050701, 5.7074]|0.625|
|244|patchtst_direct vs patchtst_joint|-1.3859|[-2.6128, -0.4549]|0.484|
|244|patchtst_direct vs patchtst_learned|-1.4918|[-3.3784, -0.39953]|0.422|
|244|patchtst_direct vs v4_analog|-1.1061|[-2.0912, -0.52665]|0.469|
|244|patchtst_direct vs patchtst_encoder|2.3997|[-0.24963, 5.3407]|0.516|
|244|patchtst_joint vs persistence|4.1492|[0.81745, 8.3201]|0.562|
|244|patchtst_learned vs persistence|4.2551|[0.55161, 8.9875]|0.609|
