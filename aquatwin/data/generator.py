"""Dataset Generator: episodes -> Parquet/CSV tables.

Outputs (``cfg.dataset.output_dir``):

* ``network/``                 nodes.csv, pipes.csv, network.json, network.geojson, network.inp
* ``sensors.csv``              sensor registry
* ``pipes_timeseries.parquet`` main dataset (one row per pipe per time step; spec fields)
* ``nodes_timeseries.parquet`` one row per node per time step
* ``system_timeseries.parquet`` pumps, tanks, totals, solver audit
* ``events.parquet``           scenario + emergent events with causal explanations
* ``telemetry_raw.parquet``    raw packets incl. duplicates and latency
* ``localization_*.parquet``   leak sweep for the GNN
* ``asset_history.parquet``    pipe-year panel for failure prediction
* ``schema.json`` / ``manifest.json``
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from aquatwin.config import AquaTwinConfig
from aquatwin.data.assets import build_asset_history
from aquatwin.data.schema import NODE_SCHEMA, PIPE_SCHEMA
from aquatwin.network.calibration import calibrate_network
from aquatwin.network.generator import generate_network
from aquatwin.network.model import WaterNetwork
from aquatwin.scenarios.catalog import SCENARIOS, ScenarioContext
from aquatwin.scenarios.engine import ScenarioEngine, compose_episode
from aquatwin.sensors.registry import SensorRegistry, place_sensors
from aquatwin.simulation import SimulationRun, StepRecord


class RowBuilder:
    """Vectorised conversion of StepRecords into tidy tables."""

    def __init__(self, network: WaterNetwork, sensors: SensorRegistry):
        self.net = network
        self.sensors = sensors
        nodes, pipes = network.nodes, network.pipes
        nidx = network.node_index
        self.s_idx = np.array([nidx[n] for n in pipes.start_node])
        self.e_idx = np.array([nidx[n] for n in pipes.end_node])
        reg = sensors.table
        sid_pos = {s: i for i, s in enumerate(reg.sensor_id)}
        node_sensor = {t: sid_pos[s] for s, t, k in zip(reg.sensor_id, reg.target, reg.kind) if k in ("pressure", "level")}
        flow_sensor = {t: sid_pos[s] for s, t, k in zip(reg.sensor_id, reg.target, reg.kind) if k == "flow"}
        D = network.distance_matrix
        mon = [n for n in sensors.monitored_nodes if n in D.index]
        assoc = np.full((len(pipes), 3), -1, int)
        for i, (pid, a, b) in enumerate(zip(pipes.index, pipes.start_node, pipes.end_node)):
            cand = [node_sensor.get(a, -1), node_sensor.get(b, -1), flow_sensor.get(pid, -1)]
            if max(cand) < 0:
                near = min(mon, key=lambda n: min(D.at[a, n], D.at[b, n]))
                cand[0] = node_sensor[near]
            assoc[i] = cand
        self.assoc = assoc
        self.area = np.pi * (pipes.diameter_mm.to_numpy(float) / 1000) ** 2 / 4
        self.monitored = np.zeros(len(nodes), bool)
        for n in node_sensor:
            self.monitored[nidx[n]] = True

    def pipe_block(self, recs: list[StepRecord]) -> dict[str, np.ndarray]:
        T = len(recs)
        get = lambda f: np.array([f(r) for r in recs])
        op = get(lambda r: r.obs_pressure)
        ep = get(lambda r: r.expected["pressure"])
        tp = get(lambda r: r.truth.pressure)
        mask = self.assoc >= 0
        idx = np.where(mask, self.assoc, 0)

        def agg(arr, fn, fill):
            vals = arr[:, idx]                       # T x P x 3
            vals = np.where(mask[None], vals, fill)
            return fn(vals, axis=2)

        health = agg(get(lambda r: r.obs.health), np.min, 1.0)
        lost = agg(get(lambda r: r.obs.lost.astype(float)), np.max, 0.0)
        delay = agg(np.nan_to_num(get(lambda r: r.obs.delay_s), nan=0.0), np.max, 0.0)
        flow = get(lambda r: r.obs_flow)
        return {
            "pressure": op[:, self.s_idx], "pressure_end": op[:, self.e_idx],
            "expected_pressure": ep[:, self.s_idx], "expected_pressure_end": ep[:, self.e_idx],
            "flow": flow, "expected_flow": get(lambda r: r.expected["flow"]),
            "velocity": flow / 1000.0 / self.area,
            "demand": get(lambda r: r.obs_demand)[:, self.s_idx],
            "sensor_health": health, "packet_loss": lost, "delay_seconds": delay,
            "temperature": np.repeat(get(lambda r: r.truth.temperature_c)[:, None], len(self.s_idx), 1),
            "rainfall": np.repeat(get(lambda r: r.truth.rainfall_mm_h)[:, None], len(self.s_idx), 1),
            "ground_movement": np.repeat(get(lambda r: r.truth.ground_movement)[:, None], len(self.s_idx), 1),
            "true_pressure": tp[:, self.s_idx], "true_flow": get(lambda r: r.truth.flow),
            "true_velocity": get(lambda r: r.truth.velocity), "true_pipe_health": get(lambda r: r.truth.health),
            "true_corrosion": get(lambda r: r.truth.corrosion),
            "water_loss": get(lambda r: r.truth.leak_flow), "leak_size": get(lambda r: r.truth.leak_size_mm),
            "leak_location": get(lambda r: r.truth.leak_distance_m), "leak_frac": get(lambda r: r.truth.leak_frac),
            "failure_risk": get(lambda r: r.truth.failure_risk),
        }

    def pipes_frame(self, recs: list[StepRecord], meta: dict) -> pd.DataFrame:
        blk = self.pipe_block(recs)
        T, P = len(recs), len(self.s_idx)
        pipes = self.net.pipes
        age_off = np.array([r.cond.age_offset_years for r in recs])
        df = pd.DataFrame({
            "timestamp": np.repeat([r.cond.timestamp for r in recs], P),
            "step": np.repeat(np.arange(T), P),
            "pipe_id": np.tile(pipes.index.to_numpy(), T),
            "node_id": np.tile(pipes.start_node.to_numpy(), T),
            "end_node": np.tile(pipes.end_node.to_numpy(), T),
            "dma": np.tile(pipes.dma.to_numpy(), T),
            "pipe_age": (np.tile(pipes.age_years.to_numpy(float), T) + np.repeat(age_off, P)),
            "material": np.tile(pipes.material.to_numpy(), T),
            "diameter_mm": np.tile(pipes.diameter_mm.to_numpy(float), T),
            "length_m": np.tile(pipes.length_m.to_numpy(float), T),
            "corrosion": np.tile(pipes.corrosion_index.to_numpy(float), T),
            "repair_count": np.tile(pipes.repair_count.to_numpy(int), T),
            "pipe_health": np.tile(pipes.health_score.to_numpy(float), T),
        })
        for k, v in blk.items():
            df[k] = v.reshape(-1)
        df["is_leak"] = (df.water_loss > 1e-6).astype(int)
        for k, v in meta.items():
            df[k] = v
        return df

    def nodes_frame(self, recs: list[StepRecord], meta: dict) -> pd.DataFrame:
        T = len(recs)
        nodes = self.net.nodes
        N = len(nodes)
        get = lambda f: np.array([f(r) for r in recs]).reshape(-1)
        df = pd.DataFrame({
            "timestamp": np.repeat([r.cond.timestamp for r in recs], N),
            "step": np.repeat(np.arange(T), N),
            "node_id": np.tile(nodes.index.to_numpy(), T),
            "node_type": np.tile(nodes.node_type.to_numpy(), T),
            "dma": np.tile(nodes.dma.to_numpy(), T),
            "elevation": np.tile(nodes.elevation.to_numpy(float), T),
            "monitored": np.tile(self.monitored.astype(int), T),
            "observed_pressure": get(lambda r: r.obs_pressure),
            "expected_pressure": get(lambda r: r.expected["pressure"]),
            "sensor_health": get(lambda r: r.obs_health),
            "observed_demand": get(lambda r: r.obs_demand),
            "twin_demand": get(lambda r: r.twin_demand),
            "true_pressure": get(lambda r: r.truth.pressure),
            "true_demand": get(lambda r: r.truth.demand_served),
            "required_demand": get(lambda r: r.truth.demand_required),
            "surge": get(lambda r: r.truth.surge),
            "isolated": get(lambda r: r.truth.isolated.astype(int)),
        })
        for k, v in meta.items():
            df[k] = v
        return df

    @staticmethod
    def system_frame(recs: list[StepRecord], meta: dict, tank_ids: list[str]) -> pd.DataFrame:
        rows = []
        for r in recs:
            t = r.truth
            row = dict(timestamp=r.cond.timestamp, step=r.cond.step, pump_on=int(t.pump_on), pump_flow=t.pump_flow,
                       observed_pump_flow=r.obs_pump_flow, total_demand=float(t.demand_served.sum()),
                       required_demand=float(t.demand_required.sum()), total_loss=t.total_loss_ls,
                       n_leaks=len(t.leaks), temperature=t.temperature_c, rainfall=t.rainfall_mm_h,
                       ground_movement=t.ground_movement, max_surge=float(t.surge.max()),
                       converged=int(t.converged), mass_balance_error=t.mass_balance_error,
                       isolated_nodes=int(t.isolated.sum()), active_events=",".join(r.cond.active_events))
            for i, tid in enumerate(tank_ids):
                row[f"level_{tid}"] = float(t.tank_levels[i])
            rows.append(row)
        df = pd.DataFrame(rows)
        for k, v in meta.items():
            df[k] = v
        return df


@dataclass
class EpisodeResult:
    pipes: pd.DataFrame
    nodes: pd.DataFrame
    system: pd.DataFrame
    events: pd.DataFrame
    telemetry: pd.DataFrame


def run_episode(network: WaterNetwork, sensors: SensorRegistry, cfg: AquaTwinConfig, level: int,
                scenario_names: list[str], seed: int, episode_id: str, start: str | pd.Timestamp,
                n_steps: int | None = None, rows: RowBuilder | None = None) -> EpisodeResult:
    rng = np.random.default_rng(seed)
    sim = cfg.simulation
    n_steps = n_steps or int(sim.episode_hours * 60 / sim.timestep_minutes)
    scen = ScenarioEngine.from_scenarios(network, sensors, scenario_names, start, n_steps, sim.timestep_minutes, rng)
    run = SimulationRun(network, sensors, cfg, level, scen, seed + 1)
    recs = run.run()
    rows = rows or RowBuilder(network, sensors)
    meta = dict(episode_id=episode_id, level=level, scenario="+".join(scenario_names))
    events = run.events_frame()
    events["episode_id"] = episode_id
    tel = pd.DataFrame([p for r in recs for p in r.obs.packets])
    if not tel.empty:
        tel["episode_id"] = episode_id
    return EpisodeResult(rows.pipes_frame(recs, meta), rows.nodes_frame(recs, meta),
                         rows.system_frame(recs, meta, network.tank_ids), events, tel)


def localization_sweep(network: WaterNetwork, sensors: SensorRegistry, cfg: AquaTwinConfig, n_samples: int,
                       rng: np.random.Generator, warmup: int = 8, leak_steps: int = 3, levels=(0, 1, 2, 3, 4, 5)
                       ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Many short physics runs, each with one leak, for GNN training (coverage of every pipe)."""
    rows = RowBuilder(network, sensors)
    sim = cfg.simulation
    node_frames, pipe_frames, labels = [], [], []
    pipes = network.pipes
    cats = ["small", "medium", "large", "burst"]
    for s in range(n_samples):
        level = int(rng.choice(levels))
        ctx = ScenarioContext(network, sensors, warmup + leak_steps, 60 // sim.timestep_minutes, rng)
        pid = str(rng.choice(pipes.index)) if rng.random() < 0.5 else ctx.sample_pipe()
        cat = str(rng.choice(cats, p=[0.1, 0.3, 0.35, 0.25]))
        ev = ctx.leak_event(cat, 0, pid, scenario="sweep")
        ev.start = warmup
        extra = []
        if level >= 3 and rng.random() < 0.4:
            extra += SCENARIOS[str(rng.choice(["sensor_drift", "sensor_bias", "sensor_failure"]))].builder(ctx)
        if level >= 4 and rng.random() < 0.4:
            extra += SCENARIOS["morning_peak"].builder(ctx)
        start = pd.Timestamp(sim.start) + pd.Timedelta(days=int(rng.integers(0, 60)), hours=int(rng.integers(0, 24)))
        scen = ScenarioEngine(network, sensors, [ev] + extra, start, warmup + leak_steps, sim.timestep_minutes)
        run = SimulationRun(network, sensors, cfg, level, scen, int(rng.integers(1e9)))
        run.truth.tank_levels = rng.uniform(1.5, 5.5, len(run.truth.tank_levels))
        recs = run.run()
        sid = f"S{s:04d}"
        meta = dict(sample_id=sid, level=level)
        nf = rows.nodes_frame(recs, meta)
        pf = rows.pipes_frame(recs, meta)
        pf = pf[pf.step >= warmup - 4]          # keep pre-leak context + leak steps
        node_frames.append(nf)
        pipe_frames.append(pf)
        for r in recs[warmup:]:
            for lk in r.truth.leaks:
                labels.append(dict(sample_id=sid, step=r.cond.step, level=level, pipe_id=lk["pipe_id"],
                                   frac=lk["frac"], distance_m=lk["distance_m"], eq_diameter_mm=lk["eq_diameter_mm"],
                                   leak_flow=lk["flow_ls"], category=cat))
    return pd.concat(node_frames, ignore_index=True), pd.concat(pipe_frames, ignore_index=True), pd.DataFrame(labels)


def prepare_world(cfg: AquaTwinConfig) -> tuple[WaterNetwork, SensorRegistry, dict]:
    network = generate_network(cfg.network)
    calib = calibrate_network(network, cfg)
    sensors = place_sensors(network, cfg.sensors)
    return network, sensors, calib


def generate_dataset(cfg: AquaTwinConfig, verbose: bool = True) -> dict:
    t0 = time.time()
    out = Path(cfg.dataset.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    network, sensors, calib = prepare_world(cfg)
    network.save(out / "network")
    sensors.table.to_csv(out / "sensors.csv", index=False)
    rows = RowBuilder(network, sensors)
    rng = np.random.default_rng(cfg.dataset.seed)
    results = []
    levels = cfg.dataset.levels
    for e in range(cfg.dataset.n_episodes):
        level = int(levels[e % len(levels)])
        names = compose_episode(level, rng)
        start = pd.Timestamp(cfg.simulation.start) + pd.Timedelta(days=int(rng.integers(0, 90)))
        res = run_episode(network, sensors, cfg, level, names, int(rng.integers(1e9)), f"E{e:03d}", start, rows=rows)
        results.append(res)
        if verbose:
            print(f"  episode E{e:03d} L{level} {'+'.join(names):60s} leaks={res.pipes[res.pipes.is_leak == 1].pipe_id.nunique():2d} "
                  f"max_loss={res.system.total_loss.max():7.2f} L/s")
    pipes_df = pd.concat([r.pipes for r in results], ignore_index=True)
    nodes_df = pd.concat([r.nodes for r in results], ignore_index=True)
    system_df = pd.concat([r.system for r in results], ignore_index=True)
    events_df = pd.concat([r.events for r in results], ignore_index=True)
    tel_df = pd.concat([r.telemetry for r in results if not r.telemetry.empty], ignore_index=True)

    pipes_df.to_parquet(out / "pipes_timeseries.parquet", index=False)
    nodes_df.to_parquet(out / "nodes_timeseries.parquet", index=False)
    system_df.to_parquet(out / "system_timeseries.parquet", index=False)
    ev = events_df.copy()
    ev["params"] = ev["params"].apply(lambda p: json.dumps(p, default=float))
    ev.to_parquet(out / "events.parquet", index=False)
    tel_df.to_parquet(out / "telemetry_raw.parquet", index=False)
    if cfg.dataset.write_csv:
        pipes_df.to_csv(out / "pipes_timeseries.csv", index=False)
        ev.to_csv(out / "events.csv", index=False)

    if verbose:
        print(f"  localisation sweep: {cfg.dataset.localization_samples} samples ...")
    ln, lp, ll = localization_sweep(network, sensors, cfg, cfg.dataset.localization_samples, rng)
    ln.to_parquet(out / "localization_nodes.parquet", index=False)
    lp.to_parquet(out / "localization_pipes.parquet", index=False)
    ll.to_parquet(out / "localization_labels.parquet", index=False)

    assets = build_asset_history(network, cfg.dataset.history_years, rng)
    assets.to_parquet(out / "asset_history.parquet", index=False)

    with open(out / "schema.json", "w", encoding="utf-8") as fh:
        json.dump({"pipes_timeseries": {k: dict(source=s, unit=u, description=d) for k, (s, u, d) in PIPE_SCHEMA.items()},
                   "nodes_timeseries": {k: dict(source=s, unit=u, description=d) for k, (s, u, d) in NODE_SCHEMA.items()}},
                  fh, indent=2)
    manifest = dict(
        records_pipes=len(pipes_df), records_nodes=len(nodes_df), episodes=cfg.dataset.n_episodes,
        telemetry_packets=len(tel_df), localization_samples=int(ll.sample_id.nunique()) if len(ll) else 0,
        asset_rows=len(assets), leak_rows=int(pipes_df.is_leak.sum()), calibration=calib,
        network=dict(nodes=len(network.nodes), pipes=len(network.pipes), sensors=sensors.table.kind.value_counts().to_dict()),
        max_mass_balance_error=float(system_df.mass_balance_error.max()),
        solver_convergence=float(system_df.converged.mean()), seconds=round(time.time() - t0, 1),
        config=cfg.to_dict())
    with open(out / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    cfg.save(out / "config.yaml")
    return manifest
