# AquaTwin AI - Training Report

Held-out test set: unseen episodes covering realism levels 0-5.

## Leak detection (XGBoost)

| Granularity | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Network alarm (per time step) | 0.726 | 0.924 | 0.577 | 0.711 | 0.905 |
| Pipe x time step | 0.997 | 0.413 | 0.283 | 0.336 | 0.968 |

Alarm calibrated on 12 unseen commissioning episodes: per-pipe normal baselines + threshold 0.625 chosen under a 5% false-alarm budget.

Mean detection delay: 2.40 steps (15 min each); leak episodes missed: 0/5.

Pipe ranking during leaks: hit@1 = 0.390, hit@5 = 0.670, hit@10 = 0.783

Recall by leak size (pipe level): medium(7-16): 0.000, large(16-35): 0.382, burst(>35): 0.492

Selected features (29): `contrast_rank`, `nbr_min_rank`, `nbr_contrast`, `flow_res`, `pipe_health`, `nbr_res`, `flow_res_rank`, `corrosion`, `sensor_health`, `dma_ufw`, `dma_ufw_roll4`, `res_end`, `gap_to_min`, `flow_res_rel`, `nbr_roll4`, `pipe_age`, `nbr_delta`, `length_m`, `res_local`, `demand`, `nbr_min`, `pressure`, `net_median_res`, `net_min_contrast`, `dma_ufw_rel`, `flow`, `hour_sin`, `nbr_count`, `hour_cos`

## Leak localisation (GNN)

| Model | Split | Top-1 | Top-5 | Mean error (m) | Median error (m) | <=100 m | <=250 m |
|---|---|---|---|---|---|---|---|
| sensitivity_baseline | validation | 0.255 | 0.584 | 583 | 235 | 0.253 | 0.510 |
| sensitivity_baseline | test | 0.286 | 0.429 | 664 | 617 | 0.286 | 0.429 |
| gcn | validation | 0.410 | 0.779 | 336 | 135 | 0.394 | 0.659 |
| gcn | test | 0.381 | 0.524 | 704 | 443 | 0.238 | 0.381 |
| sage | validation | 0.436 | 0.775 | 293 | 125 | 0.426 | 0.663 |
| sage | test | 0.190 | 0.619 | 796 | 999 | 0.190 | 0.238 |
| gat | validation | 0.432 | 0.777 | 297 | 123 | 0.440 | 0.703 |
| gat | test | 0.238 | 0.619 | 746 | 343 | 0.238 | 0.286 |

Best architecture (validation): **sage**

## Failure prediction (XGBoost regressors, GroupKFold by pipe)

| Target | MAE | RMSE | R2 |
|---|---|---|---|
| failure_risk (annual prob.) | 0.0069 | 0.0174 | 0.758 |
| remaining_useful_life (y) | 4.22 | 6.04 | 0.938 |

Next-year break AUC: 0.719; breaks captured by top-10% risk list: 0.324

## Water-loss estimation (DMA balance, test episodes)

mae_ls = 1.104, bias_ls = 0.374, rmse_ls = 2.633, volume_error_pct = 31.844, corr = 0.759

## Global explanations (mean |SHAP|)

Detection: contrast_rank (0.623), nbr_roll4 (0.513), dma_ufw_roll4 (0.435), nbr_min_rank (0.420), dma_ufw (0.385), gap_to_min (0.331), pipe_health (0.327), nbr_min (0.313)

Failure risk: diameter_mm (0.250), pipe_age (0.247), corrosion (0.235), mean_pressure (0.205), length_m (0.141), repair_count (0.125), material_code (0.114), pressure_cycles (0.023)
