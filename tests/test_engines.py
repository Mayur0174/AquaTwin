import numpy as np
import pandas as pd
import pytest

from aquatwin.data.generator import RowBuilder, run_episode
from aquatwin.data.schema import LEAKAGE_COLUMNS
from aquatwin.models.features import DETECTION_FEATURES, NetworkContext, detection_features
from aquatwin.network.calibration import count_cycles
from aquatwin.physics.hydraulics import solve
from aquatwin.scenarios.catalog import SCENARIOS, ScenarioContext
from aquatwin.scenarios.engine import ScenarioEngine
from aquatwin.sensors.ingest import align_snapshot, deduplicate
from aquatwin.simulation import SimulationRun


def test_network_structure(network):
    n, p = network.nodes, network.pipes
    assert set(n.node_type) == {"junction", "reservoir", "pump_station", "tank"}
    assert set(p.material) <= {"PVC", "HDPE", "Steel", "Cast Iron", "Ductile Iron"}
    for col in ["pipe_id", "start_node", "end_node", "length_m", "diameter_mm", "material", "installation_year",
                "age_years", "corrosion_index", "repair_count", "health_score", "failure_risk"]:
        assert p[col].notna().all(), col
    assert (p.health_score.between(0, 100)).all() and (p.failure_risk.between(0, 1)).all()


def test_solver_mass_balance(base):
    b, dem, levels, hw = base
    r = solve(b.build(dem, levels, np.array([1.0]), hw, [])[0])
    assert r.converged and r.mass_balance_error < 1e-6


def test_level0_twin_matches_truth(network, sensors, cfg):
    scen = ScenarioEngine(network, sensors, [], cfg.simulation.start, 16, 15)
    recs = SimulationRun(network, sensors, cfg, 0, scen, 0).run()
    res = np.array([r.obs_pressure - r.expected["pressure"] for r in recs])
    assert np.nanmax(np.abs(res)) < 1e-6


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_every_scenario_runs(name, network, sensors, cfg):
    rng = np.random.default_rng(3)
    scen = ScenarioEngine.from_scenarios(network, sensors, [name], cfg.simulation.start, 24, 15, rng)
    for e in scen.events:
        e.start = min(e.start, 6)
    run = SimulationRun(network, sensors, cfg, max(SCENARIOS[name].min_level, 1), scen, 1)
    recs = run.run()
    assert all(r.truth.converged for r in recs)
    if SCENARIOS[name].has_leak:
        assert max(r.truth.total_loss_ls for r in recs) > 0


def test_ingestion_dedup_and_latency():
    t = pd.Timestamp("2026-01-01")
    pk = pd.DataFrame([
        dict(sensor_id="A", seq=1, timestamp=t, received_at=t + pd.Timedelta(seconds=2), value=1.0, health=1),
        dict(sensor_id="A", seq=1, timestamp=t, received_at=t + pd.Timedelta(seconds=9), value=1.0, health=1),
        dict(sensor_id="B", seq=1, timestamp=t, received_at=t + pd.Timedelta(seconds=900), value=2.0, health=1),
    ])
    assert len(deduplicate(pk)) == 2
    snap = align_snapshot(pk, cutoff_s=600)
    assert "A" in snap and "B" not in snap


def test_features_never_use_truth(network, sensors, cfg):
    assert not set(DETECTION_FEATURES) & set(LEAKAGE_COLUMNS)
    res = run_episode(network, sensors, cfg, 2, ["large_leak"], 5, "T", cfg.simulation.start, n_steps=24)
    F = detection_features(res.pipes, res.nodes, NetworkContext.build(network))
    assert len(F) == len(res.pipes) and set(DETECTION_FEATURES) <= set(F.columns)


def test_count_cycles():
    assert count_cycles(np.array([0, 2, 0, 2, 0])) == 3
    assert count_cycles(np.array([0, 0.1, 0.2])) == 0
