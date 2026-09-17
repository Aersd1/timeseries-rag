# PatchTST V6 vs CNN V6 vs V4 analog

Same NREL store, same 64 test queries, same analog mapping, memory-normalized NMSE.
V4: z-normalized Euclidean distance on the 244-point history.
CNN joint: previous V6 CNN backbone + history NMSE gate.
PatchTST joint: PatchTST backbone (patch_len=16, stride=8, 4 heads, 2 layers) then the same joint search.

## Forecast NMSE

|H|Method|Mean NMSE|Skill vs persistence|Win vs persistence|
|---:|---|---:|---:|---:|
|24|persistence|0.09957|0.000|0.000|
|24|v4_analog|0.15938|-0.601|0.234|
|24|cnn_joint|0.1274|-0.280|0.359|
|24|patchtst_joint|0.12255|-0.231|0.328|
|24|cnn_learned|0.12484|-0.254|0.328|
|24|patchtst_learned|0.13976|-0.404|0.328|
|24|cnn_encoder|0.10025|-0.007|0.531|
|24|patchtst_encoder|0.099598|-0.000|0.531|
|24|cnn_prior|0.10091|-0.013|0.375|
|24|patchtst_prior|0.099809|-0.002|0.484|
|96|persistence|0.40046|0.000|0.000|
|96|v4_analog|0.38372|0.042|0.469|
|96|cnn_joint|0.40098|-0.001|0.453|
|96|patchtst_joint|0.36149|0.097|0.453|
|96|cnn_learned|0.54944|-0.372|0.469|
|96|patchtst_learned|0.40138|-0.002|0.500|
|96|cnn_encoder|0.36052|0.100|0.641|
|96|patchtst_encoder|0.35875|0.104|0.672|
|96|cnn_prior|0.39362|0.017|0.672|
|96|patchtst_prior|0.39709|0.008|0.406|
|244|persistence|1.0296|0.000|0.000|
|244|v4_analog|0.78867|0.234|0.609|
|244|cnn_joint|0.79874|0.224|0.547|
|244|patchtst_joint|0.75925|0.263|0.562|
|244|cnn_learned|1.0767|-0.046|0.531|
|244|patchtst_learned|0.73035|0.291|0.609|
|244|cnn_encoder|0.90567|0.120|0.750|
|244|patchtst_encoder|0.90563|0.120|0.750|
|244|cnn_prior|0.99874|0.030|0.656|
|244|patchtst_prior|1.0387|-0.009|0.484|

## Paired differences (positive = method better)

|H|Method vs baseline|Δ NMSE|95% CI|Win rate|
|---:|---|---:|---|---:|
|24|patchtst_joint vs v4_analog|0.036833|[0.028791, 0.044875]|0.688|
|24|patchtst_joint vs cnn_joint|0.0048561|[-0.00037, 0.010082]|0.562|
|24|patchtst_joint vs persistence|-0.022976|[-0.038421, -0.003796]|0.328|
|24|patchtst_learned vs v4_analog|0.019621|[-0.046288, 0.07757]|0.641|
|24|patchtst_learned vs cnn_learned|-0.014917|[-0.088206, 0.033414]|0.547|
|24|cnn_joint vs v4_analog|0.031977|[0.019198, 0.045987]|0.656|
|96|patchtst_joint vs v4_analog|0.022226|[-0.025171, 0.06992]|0.562|
|96|patchtst_joint vs cnn_joint|0.039492|[-0.02519, 0.12906]|0.469|
|96|patchtst_joint vs persistence|0.038967|[-0.0036143, 0.073603]|0.453|
|96|patchtst_learned vs v4_analog|-0.017668|[-0.10426, 0.068927]|0.453|
|96|patchtst_learned vs cnn_learned|0.14805|[0.029746, 0.29087]|0.500|
|96|cnn_joint vs v4_analog|-0.017266|[-0.11885, 0.083213]|0.531|
|244|patchtst_joint vs v4_analog|0.029418|[-0.022338, 0.09525]|0.484|
|244|patchtst_joint vs cnn_joint|0.039485|[-0.011737, 0.11519]|0.438|
|244|patchtst_joint vs persistence|0.27036|[0.15041, 0.42579]|0.562|
|244|patchtst_learned vs v4_analog|0.058324|[-0.037435, 0.14278]|0.516|
|244|patchtst_learned vs cnn_learned|0.34638|[0.22732, 0.45712]|0.547|
|244|cnn_joint vs v4_analog|-0.010067|[-0.032935, 0.012716]|0.484|
