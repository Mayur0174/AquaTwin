"""The ten IMMUTABLE PHYSICAL TRUTHS, verified against the simulator itself.

These tests are the contract of the Ground-Truth World: if any engine change
breaks causality, CI fails.
"""
import numpy as np
import pandas as pd

from aquatwin.physics import degradation as deg
from aquatwin.physics.demand import pattern_value
from aquatwin.physics.hydraulics import solve
from aquatwin.scenarios.engine import ScenarioEngine
from aquatwin.scenarios.events import Event, LeakSpec, leak_area_m2
from aquatwin.simulation import SimulationRun
from tests.conftest import pipe_far_from_tanks

SIZES_MM = [3, 8, 15, 25, 45, 70]


def _leak_runs(base, pipe_id, sizes=SIZES_MM, frac=0.5):
    b, dem, levels, hw = base
    r0 = solve(b.build(dem, levels, np.array([1.0]), hw, [])[0])
    out = []
    for d in sizes:
        a = leak_area_m2(d)
        r = solve(b.build(dem, levels, np.array([1.0]), hw, [LeakSpec("t", pipe_id, frac, a, 0.02 * a)])[0])
        out.append((r, r0))
    return out


def test_rule1_larger_leaks_larger_losses(network, base):
    for pid in [network.pipes.index[10], pipe_far_from_tanks(network)]:
        losses = [r.emitter_flow[0] for r, _ in _leak_runs(base, pid)]
        assert np.all(np.diff(losses) > 0), losses


def test_rule2_larger_leaks_stronger_pressure_anomaly(network, base, junction_mask):
    pid = pipe_far_from_tanks(network)
    n0 = len(network.nodes)
    drops = [-(r.pressure[:n0] - r0.pressure)[junction_mask].min() for r, r0 in _leak_runs(base, pid)]
    assert np.all(np.diff(drops) > 0), drops


def test_rule3_older_pipes_higher_failure_probability():
    for mat in ["PVC", "HDPE", "Steel", "Cast Iron", "Ductile Iron"]:
        ages = np.array([5, 15, 30, 50, 70])
        ci = deg.corrosion_index([mat] * 5, [150] * 5, ages, [1.0] * 5)
        rate = deg.break_rate_per_km_yr([mat] * 5, ages, [150] * 5, ci, [0] * 5, [40] * 5, [6] * 5, [0.3] * 5, [0] * 5)
        assert np.all(np.diff(deg.annual_failure_probability(rate, [200] * 5)) > 0), mat


def test_rule4_corrosion_increases_leak_probability_and_opening():
    c = np.linspace(0, 0.9, 6)
    rate = deg.break_rate_per_km_yr(["Cast Iron"] * 6, [40] * 6, [150] * 6, c, [0] * 6, [40] * 6, [6] * 6, [0.3] * 6, [0] * 6)
    assert np.all(np.diff(rate) > 0)
    assert np.all(np.diff(deg.leak_opening_multiplier(c)) > 0)
    assert np.all(np.diff(deg.strength_head_m(["Cast Iron"] * 6, [150] * 6, c)) < 0)


def test_rule5_closer_sensors_observe_stronger_signature(network, base, junction_mask):
    pid = pipe_far_from_tanks(network)
    (r, r0), = _leak_runs(base, pid, sizes=[40])
    n0 = len(network.nodes)
    drop = -(r.pressure[:n0] - r0.pressure)
    u, v = network.edge_of(pid)
    D = network.distance_matrix
    ids = network.nodes.index[junction_mask]
    dist = np.array([min(D.at[u, n], D.at[v, n]) for n in ids])
    near, far = drop[junction_mask][dist < 300].mean(), drop[junction_mask][dist > 1200].mean()
    assert near > far > -1e-9
    assert np.corrcoef(dist, drop[junction_mask])[0, 1] < -0.3


def test_rule6_repairs_indicate_deterioration():
    rep = np.arange(6)
    rate = deg.break_rate_per_km_yr(["Ductile Iron"] * 6, [30] * 6, [150] * 6, [0.3] * 6, rep, [40] * 6, [6] * 6, [0.3] * 6, [0] * 6)
    health = deg.health_score(["Ductile Iron"] * 6, [30] * 6, [0.3] * 6, rep, [0.2] * 6)
    assert np.all(np.diff(rate) > 0) and np.all(np.diff(health) < 0)


def test_rule7_higher_demand_lowers_pressure(network, base, junction_mask):
    b, dem, levels, hw = base
    n0 = len(network.nodes)
    means = [solve(b.build(dem * f, levels, np.array([1.0]), hw, [])[0]).pressure[:n0][junction_mask].mean()
             for f in (0.5, 1.0, 1.5, 2.0)]
    assert np.all(np.diff(means) < 0), means


def test_rule8_demand_follows_time_of_day():
    night = np.mean([pattern_value("residential", h, False) for h in (1, 2, 3, 4)])
    morning = np.mean([pattern_value("residential", h, False) for h in (6.5, 7, 7.5, 8)])
    evening = np.mean([pattern_value("residential", h, False) for h in (18, 19, 20)])
    midday = np.mean([pattern_value("residential", h, False) for h in (12, 13, 14)])
    assert morning > midday > night and evening > midday
    assert np.mean([pattern_value("commercial", h, False) for h in (10, 11, 14)]) > \
        np.mean([pattern_value("commercial", h, False) for h in (1, 2, 3)])


def test_rule9_health_decreases_with_age_and_stress():
    ages = np.array([5, 20, 40, 60, 80])
    ci = deg.corrosion_index(["Steel"] * 5, [150] * 5, ages, [1.0] * 5)
    h_age = deg.health_score(["Steel"] * 5, ages, ci, [0] * 5, [0.2] * 5)
    h_stress = deg.health_score(["Steel"] * 5, [30] * 5, [0.3] * 5, [0] * 5, np.linspace(0, 1, 5))
    assert np.all(np.diff(h_age) < 0) and np.all(np.diff(h_stress) < 0)


def test_rule10_combined_failures_interact(network, sensors, cfg, base, junction_mask):
    """A leak under high demand loses less water per orifice area (lower pressure, FAVAD),
    and two leaks together depress pressure more than either alone."""
    b, dem, levels, hw = base
    pid = pipe_far_from_tanks(network)
    a = leak_area_m2(30)
    lk = LeakSpec("t", pid, 0.5, a, 0.02 * a)
    q_norm = solve(b.build(dem, levels, np.array([1.0]), hw, [lk])[0]).emitter_flow[0]
    q_high = solve(b.build(dem * 2.0, levels, np.array([1.0]), hw, [lk])[0]).emitter_flow[0]
    assert q_high < q_norm
    other = network.pipes.index[5]
    lk2 = LeakSpec("u", other, 0.5, a, 0.02 * a)
    n0 = len(network.nodes)
    r0 = solve(b.build(dem, levels, np.array([1.0]), hw, [])[0]).pressure[:n0]
    p1 = solve(b.build(dem, levels, np.array([1.0]), hw, [lk])[0]).pressure[:n0]
    p12 = solve(b.build(dem, levels, np.array([1.0]), hw, [lk, lk2])[0]).pressure[:n0]
    assert (r0 - p12)[junction_mask].sum() > (r0 - p1)[junction_mask].sum()
    # burst + isolation: closing the burst pipe stops the loss
    burst = LeakSpec("b", pid, 0.5, leak_area_m2(80), 0.0)
    assert solve(b.build(dem, levels, np.array([1.0]), hw, [burst], closed={pid})[0]).emitter_flow[0] < 1e-9


def test_causality_no_randomness_in_hydraulic_truth(network, sensors, cfg):
    """Same cause -> identical physics regardless of the sensor random stream (level 0 vs level 5 sensors)."""
    ev = [Event("leak", 4, None, network.pipes.index[20], dict(area_m2=leak_area_m2(20), frac=0.4, favad_slope=0.0))]
    truths = []
    for lvl, seed in ((1, 1), (2, 99)):
        scen = ScenarioEngine(network, sensors, ev, cfg.simulation.start, 12, 15)
        run = SimulationRun(network, sensors, cfg, lvl, scen, seed)
        run.truth.noise.sigma = 0.0
        run.truth.noise.state[:] = 0.0
        truths.append(np.array([r.truth.pressure for r in run.run()]))
    assert np.allclose(truths[0], truths[1])
