"""Calibrate asset stress attributes against the hydraulic simulation.

The generator initially estimates operating pressure from static head.  Here
we run one nominal day through the truth engine and replace those estimates
with simulated mean pressure and *pressure cycles per day* (direction
reversals with > 0.5 m swing, the fatigue driver for plastic pipes), then
recompute stress, break rate, health, risk and RUL.
"""
from __future__ import annotations

import numpy as np

from aquatwin.config import AquaTwinConfig
from aquatwin.network.model import WaterNetwork
from aquatwin.physics import degradation as deg


def count_cycles(series: np.ndarray, min_swing: float = 0.5) -> int:
    """Count pressure reversals whose swing exceeds ``min_swing`` (rainflow-like turning points)."""
    if len(series) < 3:
        return 0
    turns, last_ext, direction = 0, series[0], 0
    for v in series[1:]:
        if direction >= 0 and v < last_ext - min_swing:
            turns += direction > 0
            direction, last_ext = -1, v
        elif direction <= 0 and v > last_ext + min_swing:
            turns += direction < 0
            direction, last_ext = 1, v
        elif (direction > 0 and v > last_ext) or (direction < 0 and v < last_ext):
            last_ext = v
    return int(turns)


def calibrate_network(network: WaterNetwork, cfg: AquaTwinConfig) -> dict:
    from aquatwin.scenarios.engine import ScenarioEngine
    from aquatwin.sensors.registry import place_sensors
    from aquatwin.simulation import SimulationRun

    sensors = place_sensors(network, cfg.sensors)
    steps = int(24 * 60 / cfg.simulation.timestep_minutes)
    scen = ScenarioEngine(network, sensors, [], cfg.simulation.start, steps, cfg.simulation.timestep_minutes)
    run = SimulationRun(network, sensors, cfg, 0, scen, seed=0)
    recs = run.run()
    P = np.array([r.truth.pressure for r in recs])          # steps x nodes
    node_idx = network.node_index
    pipes = network.pipes
    s_idx = np.array([node_idx[n] for n in pipes.start_node])
    e_idx = np.array([node_idx[n] for n in pipes.end_node])
    node_cycles = np.array([count_cycles(P[:, j]) for j in range(P.shape[1])], float)
    mean_p = P.mean(axis=0)
    cycles = np.maximum(node_cycles[s_idx], node_cycles[e_idx])
    p_pipe = 0.5 * (mean_p[s_idx] + mean_p[e_idx])

    mats = pipes.material.to_numpy()
    age = pipes.age_years.to_numpy(float)
    dia = pipes.diameter_mm.to_numpy(float)
    ci = pipes.corrosion_index.to_numpy(float)
    rep = pipes.repair_count.to_numpy(float)
    traffic = pipes.traffic_load.to_numpy(float)
    soil = pipes.soil_corrosivity.to_numpy(float)
    length = pipes.length_m.to_numpy(float)
    stress = deg.stress_index(p_pipe, cycles, traffic)
    rate = deg.break_rate_per_km_yr(mats, age, dia, ci, rep, p_pipe, cycles, traffic, 0.0)
    health = deg.health_score(mats, age, ci, rep, stress)
    risk = deg.annual_failure_probability(rate, length)
    rul = deg.remaining_useful_life(mats, dia, age, soil, rep, stress)
    for i, pid in enumerate(pipes.index):
        u, v = network.edge_of(pid)
        network.graph.edges[u, v].update(pressure_cycles=float(cycles[i]), est_pressure_m=float(p_pipe[i]),
                                         stress_index=float(stress[i]), break_rate=float(rate[i]),
                                         health_score=float(health[i]), failure_risk=float(risk[i]),
                                         rul_years=float(rul[i]))
    network.invalidate()
    return {"mean_pressure_m": float(mean_p.mean()), "mean_cycles": float(cycles.mean()),
            "network_break_rate_km_yr": float((rate * length).sum() / length.sum())}
