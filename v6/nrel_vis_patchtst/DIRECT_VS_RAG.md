# Direct PatchTST vs RAG — z-space MSE

Same 64 test queries. Every method is mapped to raw power, then scored as
`mean(((ŷ-μ)/σ - (y-μ)/σ)²)` with `σ = max(std(query history), 0.1·memory_std)`.
This is the same normalized space PatchTST trains in and analog mapping uses.

## Protocol

- Train: all 26,142 train-split windows (stride 32), val 2,048 windows, AdamW + cosine, early stop patience 12.
- Direct model: PatchTST flatten head (Nie et al. 2023), loss = multi-horizon MSE in z-space.
- Reported checkpoint: hidden=64, 2 layers, 4 heads; best val 5.444 at epoch 9, stop at epoch 21.
- Larger run (hidden=128, 3 layers) peaked at epoch 2 (val 5.472) and matched this test ranking.
- RAG encoder/index unchanged: `runs/v6_nrel_patchtst/encoder/best.pt`.

## Verdict

Fully trained PatchTST **does not beat RAG** on this metric. At H=96 and H=244, learned analog has the lowest MSE_z; paired Δ vs direct excludes 0. At H=24, persistence / encoder residual win; direct ≈ joint and both lose to last-value.

## MSE (normalized / σ space)

|H|Method|Mean MSE_z|Skill vs persistence|Win vs persistence|
|---:|---|---:|---:|---:|
|24|persistence|0.52204|0.000|0.000|
|24|v4_analog|0.74429|-0.426|0.234|
|24|patchtst_direct|0.61315|-0.175|0.266|
|24|patchtst_encoder|0.52025|0.003|0.531|
|24|patchtst_prior|0.52542|-0.006|0.484|
|24|patchtst_joint|0.59538|-0.140|0.328|
|24|patchtst_learned|0.66551|-0.275|0.328|
|96|persistence|4.4555|0.000|0.000|
|96|v4_analog|3.2181|0.278|0.469|
|96|patchtst_direct|3.8733|0.131|0.453|
|96|patchtst_encoder|4.3293|0.028|0.672|
|96|patchtst_prior|4.4279|0.006|0.406|
|96|patchtst_joint|3.1031|0.304|0.453|
|96|patchtst_learned|2.7217|0.389|0.500|
|244|persistence|10.881|0.000|0.000|
|244|v4_analog|7.0116|0.356|0.609|
|244|patchtst_direct|8.1262|0.253|0.609|
|244|patchtst_encoder|10.517|0.033|0.750|
|244|patchtst_prior|10.936|-0.005|0.484|
|244|patchtst_joint|6.7317|0.381|0.562|
|244|patchtst_learned|6.6258|0.391|0.609|

## Paired Δ MSE_z (positive = left better)

|H|Method vs baseline|Δ MSE_z|95% CI|Win rate|
|---:|---|---:|---|---:|
|24|patchtst_direct vs persistence|-0.091109|[-0.16268, -0.031258]|0.266|
|24|patchtst_direct vs patchtst_joint|-0.01777|[-0.11555, 0.082357]|0.469|
|24|patchtst_direct vs patchtst_learned|0.052367|[-0.095505, 0.28173]|0.406|
|24|patchtst_direct vs v4_analog|0.13115|[0.045997, 0.24245]|0.578|
|24|patchtst_direct vs patchtst_encoder|-0.092893|[-0.16809, -0.035321]|0.297|
|24|patchtst_joint vs persistence|-0.073339|[-0.1489, 0.0041179]|0.328|
|24|patchtst_learned vs persistence|-0.14348|[-0.36763, 0.070762]|0.328|
|96|patchtst_direct vs persistence|0.58218|[-0.80731, 2.4533]|0.453|
|96|patchtst_direct vs patchtst_joint|-0.77029|[-1.2196, -0.43131]|0.422|
|96|patchtst_direct vs patchtst_learned|-1.1517|[-2.3704, -0.19666]|0.484|
|96|patchtst_direct vs v4_analog|-0.65525|[-1.1608, -0.23547]|0.469|
|96|patchtst_direct vs patchtst_encoder|0.45591|[-0.98903, 2.0315]|0.438|
|96|patchtst_joint vs persistence|1.3525|[0.12247, 3.1175]|0.453|
|96|patchtst_learned vs persistence|1.7339|[-0.050009, 4.8236]|0.500|
|244|patchtst_direct vs persistence|2.7547|[-1.4588, 7.4797]|0.609|
|244|patchtst_direct vs patchtst_joint|-1.3945|[-3.0763, -0.26683]|0.469|
|244|patchtst_direct vs patchtst_learned|-1.5004|[-2.6941, -0.30664]|0.422|
|244|patchtst_direct vs v4_analog|-1.1146|[-2.641, -0.28368]|0.422|
|244|patchtst_direct vs patchtst_encoder|2.3911|[-1.7926, 7.1131]|0.578|
|244|patchtst_joint vs persistence|4.1492|[0.81745, 8.3201]|0.562|
|244|patchtst_learned vs persistence|4.2551|[0.55161, 8.9875]|0.609|
