"""Asset-history panel for the failure-prediction (predictive maintenance) model.

One row per (pipe, year).  Inputs are what a utility's GIS + CMMS + SCADA
would hold at that year -- some of them *measured with uncertainty*
(corrosion from condition inspections, pressure cycles from loggers).  The
targets come from the hidden deterioration truth (exact soil corrosivity,
exact traffic loading, exact wall loss), so the model must learn the
physics through noisy proxies rather than read the answer off its inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aquatwin.network.model import WaterNetwork
from aquatwin.physics import degradation as deg


def build_asset_history(network: WaterNetwork, years: int, rng: np.random.Generator,
                        inspection_noise: float = 0.06, cycles_noise: float = 0.15) -> pd.DataFrame:
    g = network.graph
    current = g.graph.get("current_year", 2026)
    rows = []
    for u, v, e in g.edges(data=True):
        if e.get("link_type") != "pipe":
            continue
        breaks = np.array(e.get("break_years", []), int)
        prev_rep = np.array(e.get("repair_years", []), int)
        for year in range(current - years, current + 1):
            age = year - e["installation_year"]
            if age < 1:
                continue
            hist_leaks = int((breaks < year).sum())
            repairs = hist_leaks + int((prev_rep < year).sum())
            rows.append(dict(pipe_id=e["pipe_id"], year=year, pipe_age=float(age), material=e["material"],
                             diameter_mm=e["diameter_mm"], length_m=e["length_m"], repair_count=repairs,
                             historical_leaks=hist_leaks, soil=e["soil_corrosivity"], traffic=e["traffic_load"],
                             pressure_m=e.get("est_pressure_m", 40.0), cycles=e.get("pressure_cycles", 6.0),
                             breaks_next_year=int((breaks == year).sum()), dma=e.get("dma")))
    df = pd.DataFrame(rows)
    mats = df.material.to_numpy()
    true_ci = deg.corrosion_index(mats, df.diameter_mm.to_numpy(), df.pipe_age.to_numpy(), df.soil.to_numpy())
    stress = deg.stress_index(df.pressure_m, df.cycles, df.traffic)
    rate = deg.break_rate_per_km_yr(mats, df.pipe_age, df.diameter_mm, true_ci, df.repair_count, df.pressure_m,
                                    df.cycles, df.traffic, 0.0)
    df["true_corrosion"] = true_ci
    df["failure_risk"] = deg.annual_failure_probability(rate, df.length_m)
    df["health_score"] = deg.health_score(mats, df.pipe_age, true_ci, df.repair_count, stress)
    df["remaining_useful_life"] = deg.remaining_useful_life(mats, df.diameter_mm.to_numpy(), df.pipe_age.to_numpy(),
                                                            df.soil.to_numpy(), df.repair_count.to_numpy(),
                                                            stress.to_numpy() if hasattr(stress, "to_numpy") else stress)
    # observed (inspection / logger) features
    df["corrosion"] = np.clip(true_ci + rng.normal(0, inspection_noise, len(df)), 0, 1)
    df["pressure_cycles"] = np.maximum(0, df.cycles * (1 + rng.normal(0, cycles_noise, len(df))))
    df["mean_pressure"] = df.pressure_m + rng.normal(0, 1.0, len(df))
    return df.drop(columns=["soil", "traffic", "cycles", "pressure_m"])
