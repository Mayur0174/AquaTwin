# AquaTwin AI - Evaluation under increasing realism

Fresh, unseen episodes per level; full production pipeline.

| Level | Environment | Leak episodes detected | Detection acc. | Detection F1 | F1 (leaks ≥1 L/s) | False pos. (steps) | False neg. (steps) | Delay (min) | Loc. error (m) | Loc. median (m) | Loc. top-1 | Loc. ≤250 m | Classical loc. (m) | Classical median (m) | Failure R2 | Loss MAE (L/s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Perfect | 3/3 | 0.992 | 0.992 | 0.988 | 0 | 3 | 15 | 173 | 133 | 0.458 | 0.726 | 272 | 98 | 0.784 | 0.089 |
| 1 | Sensor Noise | 4/4 | 0.654 | 0.722 | 0.828 | 1 | 132 | 146 | 275 | 149 | 0.440 | 0.655 | 508 | 251 | 0.781 | 0.674 |
| 2 | Communication Noise | 4/4 | 0.654 | 0.675 | 0.794 | 21 | 112 | 240 | 329 | 140 | 0.440 | 0.654 | 486 | 212 | 0.772 | 0.814 |
| 3 | Infrastructure Degradation | 3/4 | 0.729 | 0.706 | 0.799 | 6 | 98 | 155 | 300 | 148 | 0.358 | 0.600 | 615 | 454 | 0.785 | 1.061 |
| 4 | Compound Events | 2/3 | 0.727 | 0.649 | 0.802 | 1 | 104 | 7.500 | 343 | 174 | 0.357 | 0.649 | 637 | 335 | 0.781 | 0.752 |
| 5 | Realistic Urban Operations | 4/4 | 0.812 | 0.789 | 0.799 | 3 | 69 | 22 | 257 | 118 | 0.485 | 0.731 | 618 | 223 | 0.764 | 2.374 |

Localisation columns come from a controlled benchmark of fresh single-leak physics runs (≈60 leaks, ~180 onset snapshots per level, leaks ≥ 1 L/s); the classical column is the sensitivity-matrix correlation method on the same snapshots. Detection columns come from full-day operational episodes. *F1 (leaks ≥1 L/s)* excludes time steps whose only leaks are below the physical sensing floor.

_Generated in 157 s._

Level definitions: L0: Perfect environment: exact sensors, perfect communications, nominal assets; L1: Sensor noise: Gaussian noise + quantisation on every instrument; L2: Communication noise: bursty packet loss, latency, duplicates, gateway outages; L3: Infrastructure degradation: drift/bias, aging, blockage, valve faults, water hammer, consumer variability; L4: Compound events: concurrent leaks, leak + sensor failure, burst + isolation, demand surges; L5: Realistic urban operations: all of the above + weather, settlement, construction, traffic, weekends