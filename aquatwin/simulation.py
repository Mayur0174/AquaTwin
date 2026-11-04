"""Simulation orchestration: CAUSE -> PHYSICS -> OBSERVATION, one step at a time.

``SimulationRun`` wires the Scenario Engine (cause), the Hydraulic Truth
Engine (physics), the Sensor Reality Engine (observation) and the Twin
Baseline (operator expectation).  It is used identically by the offline
dataset generator, the interactive digital-twin session and the API.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd

from aquatwin.config import AquaTwinConfig
from aquatwin.network.model import WaterNetwork
from aquatwin.scenarios.engine import REALISM_LEVELS, ScenarioEngine
from aquatwin.scenarios.events import Conditions, Event
from aquatwin.sensors.reality import Observation, RealityEngine
from aquatwin.sensors.registry import SensorRegistry
from aquatwin.truth.engine import TruthEngine, TruthSnapshot, TwinBaseline


@dataclass
class StepRecord:
    cond: Conditions
    truth: TruthSnapshot
    obs: Observation
    expected: dict[str, np.ndarray]
    obs_pressure: np.ndarray       # per node (NaN if not monitored / missing)
    obs_health: np.ndarray         # per node pressure-sensor health (NaN if none)
    obs_demand: np.ndarray         # per node AMI demand (NaN if missing)
    obs_flow: np.ndarray           # per pipe (NaN if not metered / missing)
    obs_pump_flow: float
    obs_tank_levels: np.ndarray
    twin_demand: np.ndarray        # per node demand used by the twin (AMI, gap-filled)


class SimulationRun:
    def __init__(self, network: WaterNetwork, sensors: SensorRegistry, cfg: AquaTwinConfig, level: int,
                 scenario: ScenarioEngine, seed: int):
        self.net = network
        self.sensors = sensors
        self.cfg = cfg
        self.level = level
        self.profile = REALISM_LEVELS[level]
        self.scenario = scenario
        ss = np.random.SeedSequence(seed)
        env_seed, sensor_seed = ss.spawn(2)
        self.truth = TruthEngine(network, cfg.simulation, np.random.default_rng(env_seed),
                                 demand_noise=self.profile.demand_noise)
        self.reality = RealityEngine(network, sensors, cfg.sensors, self.profile, np.random.default_rng(sensor_seed),
                                     cfg.simulation.timestep_minutes)
        self.baseline = TwinBaseline(network, cfg.simulation)
        self.step_idx = 0
        self.emergent_events: list[Event] = []
        reg = sensors.table
        nidx, pidx = network.node_index, network.pipe_index
        self._p_sensor = [(reg.index.get_loc(s), nidx[t]) for s, t in zip(reg.sensor_id, reg.target)
                          if reg.at[s, "kind"] == "pressure"]
        self._l_sensor = [(reg.index.get_loc(s), network.tank_ids.index(t), nidx[t])
                          for s, t in zip(reg.sensor_id, reg.target) if reg.at[s, "kind"] == "level"]
        self._f_sensor = [(reg.index.get_loc(s), pidx[t]) for s, t in zip(reg.sensor_id, reg.target)
                          if reg.at[s, "kind"] == "flow" and t in pidx]
        self._pump_sensor = [reg.index.get_loc(s) for s, t in zip(reg.sensor_id, reg.target)
                             if reg.at[s, "kind"] == "flow" and t not in pidx]
        self._a_sensor = [(reg.index.get_loc(s), nidx[t]) for s, t in zip(reg.sensor_id, reg.target)
                          if reg.at[s, "kind"] == "ami"]
        self._last_tank = self.truth.tank_levels.copy()
        self._last_demand = None

    @property
    def n_steps(self) -> int:
        return self.scenario.n_steps

    def step(self) -> StepRecord:
        cond = self.scenario.conditions(self.step_idx)
        truth = self.truth.step(cond)
        self.emergent_events.extend(truth.new_events)
        obs = self.reality.observe(truth, cond)
        n_nodes, n_pipes = len(self.net.nodes), len(self.net.pipes)
        op = np.full(n_nodes, np.nan)
        oh = np.full(n_nodes, np.nan)
        for si, ni in self._p_sensor:
            op[ni] = obs.value[si]
            oh[ni] = obs.health[si]
        tank = self._last_tank.copy()
        for si, ti, ni in self._l_sensor:
            if not np.isnan(obs.value[si]):
                tank[ti] = obs.value[si]
            op[ni] = obs.value[si]
            oh[ni] = obs.health[si]
        self._last_tank = tank
        od = np.full(n_nodes, np.nan)
        for si, ni in self._a_sensor:
            od[ni] = obs.value[si]
        of = np.full(n_pipes, np.nan)
        for si, pi in self._f_sensor:
            of[pi] = obs.value[si]
        opf = float(obs.value[self._pump_sensor[0]]) if self._pump_sensor else np.nan

        # twin expectation from observed consumption (fallback: last value, then pattern forecast)
        dem = od.copy()
        if self._last_demand is not None:
            dem = np.where(np.isnan(dem), self._last_demand, dem)
        if np.isnan(dem).any():
            neutral = dataclasses.replace(cond, demand_mult=1.0, demand_mult_landuse={}, demand_mult_dma={})
            forecast = self.truth.required_demand_ls(neutral, noise=False)
            dem = np.where(np.isnan(dem), forecast, dem)
        self._last_demand = dem
        expected = self.baseline.expected(dem, tank, truth.pump_on, cond.known_closures | truth.tank_valves_closed,
                                          cond.age_offset_years)
        self.step_idx += 1
        return StepRecord(cond, truth, obs, expected, op, oh, od, of, opf, tank, dem)

    def run(self) -> list[StepRecord]:
        return [self.step() for _ in range(self.n_steps - self.step_idx)]

    def events_frame(self) -> pd.DataFrame:
        df = self.scenario.events_frame()
        if self.emergent_events:
            em = pd.DataFrame([e.to_dict() for e in self.emergent_events])
            em["start_time"] = [self.scenario.start + pd.Timedelta(minutes=self.scenario.step_minutes * e.start)
                                for e in self.emergent_events]
            df = pd.concat([df, em], ignore_index=True)
        return df
