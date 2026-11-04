# Evaluation Methodology

## 1. Principles

1. **Unseen worlds:** test data is never a random row split. Detection is tested on held-out
   *episodes* (the last realism-level cycle), and the realism-level suite simulates *fresh* episodes
   with new seeds and new scenario compositions.
2. **Production path:** evaluation runs exactly what the live twin runs: telemetry → twin baseline →
   features → detector → alarm persistence → onset-anchored GNN localisation → DMA balance.
3. **Truth used only for scoring:** truth columns are asserted absent from model inputs.
4. **Honest physical limits:** results are broken down by leak size and realism level, and a
   classical baseline is reported alongside the GNN.

## 2. Realism levels (`python -m aquatwin evaluate --episodes 3`)

| Level | Environment | What changes |
|---|---|---|
| 0 | Perfect environment | Exact sensors, perfect comms, nominal assets |
| 1 | Sensor noise | Gaussian noise + quantisation |
| 2 | Communication noise | Bursty loss, latency (late = missing), duplicates, outages |
| 3 | Infrastructure degradation | Background drift/bias/frozen/failed sensors, aging, blockage, valve faults, water hammer, consumer variability |
| 4 | Compound events | Leak + high demand, leak + sensor failure, dual leaks, burst + shutdown, leak + corrosion + surge |
| 5 | Realistic urban operations | All of the above + rain, settlement, construction, traffic, weekends |

The failure model is evaluated at each level on an asset register with degraded data quality:
inspection noise rises from 0.06 to 0.15, and at L3 and above 5-15% of repair records are partially lost.

## 3. Metrics

| Service | Metric | Definition |
|---|---|---|
| Detection (network alarm) | Accuracy, precision, recall, F1, ROC-AUC | Per 15-min time step: alarm (max pipe probability ≥ threshold for 2 steps) vs any active leak |
| | False positives / false negatives | Alarm steps without a leak / leak steps without an alarm |
| | Detection delay | Minutes from leak onset to first alarm |
| Detection (pipe level) | ROC-AUC, PR-AUC, F1, hit@k | Per pipe × step. hit@k = a true leaking pipe is among the top-k ranked pipes |
| | Recall by leak size | Small (< 7 mm), medium, large, burst |
| Localisation | Localisation error (m) | **Network (along-pipe) distance** between the predicted point (pipe + position) and the nearest true leak |
| | Top-1/3/5, ≤ 100 m, ≤ 250 m | Pipe-level accuracy and operational tolerance (an acoustic crew can sweep about 250 m) |
| | Baseline | Classical sensitivity-matrix correlation on the same snapshots |
| Failure prediction | MAE, RMSE, R² | On annual failure probability and RUL (GroupKFold by pipe) |
| | Break AUC, top-10% capture | Does the risk ranking anticipate next-year breaks? |
| Water loss | MAE (L/s), volume error (%) | DMA balance estimate vs true leak outflow |

## 4. Outputs

* `reports/training_report.{md,json}`: held-out test metrics, feature selection, SHAP importance.
* `reports/evaluation_levels.{md,csv,json}`: the per-level table (also plotted in the dashboard's
  Evaluation tab).
* `reports/asset_risk_ranking.csv`: current predictive-maintenance list.
* `artifacts/models/registry.json`: metrics and dataset hash per model version.
