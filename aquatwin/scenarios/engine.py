"""Scenario Engine: event timeline -> per-step physical :class:`Conditions`.

Also defines the six *realism levels* used for training-data diversity and
for the evaluation framework, and composes episodes for each level.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aquatwin.network.model import WaterNetwork
from aquatwin.physics.demand import diurnal_temperature
from aquatwin.scenarios.catalog import SCENARIOS, ScenarioContext
from aquatwin.scenarios.events import Conditions, Event, LeakSpec
from aquatwin.sensors.registry import SensorRegistry


@dataclass(frozen=True)
class RealismProfile:
    level: int
    name: str
    sensor_noise: bool
    comms: bool
    degradation: bool
    compound: bool
    environment: bool
    demand_noise: bool
    background_faults: float     # share of sensors with background faults

    @property
    def description(self) -> str:
        return REALISM_DESCRIPTIONS[self.level]


REALISM_DESCRIPTIONS = {
    0: "Perfect environment: exact sensors, perfect communications, nominal assets",
    1: "Sensor noise: Gaussian noise + quantisation on every instrument",
    2: "Communication noise: bursty packet loss, latency, duplicates, gateway outages",
    3: "Infrastructure degradation: drift/bias, aging, blockage, valve faults, water hammer, consumer variability",
    4: "Compound events: concurrent leaks, leak + sensor failure, burst + isolation, demand surges",
    5: "Realistic urban operations: all of the above + weather, settlement, construction, traffic, weekends",
}

REALISM_LEVELS = {
    0: RealismProfile(0, "Perfect", False, False, False, False, False, False, 0.0),
    1: RealismProfile(1, "Sensor Noise", True, False, False, False, False, False, 0.0),
    2: RealismProfile(2, "Communication Noise", True, True, False, False, False, False, 0.0),
    3: RealismProfile(3, "Infrastructure Degradation", True, True, True, False, False, True, 0.03),
    4: RealismProfile(4, "Compound Events", True, True, True, True, False, True, 0.04),
    5: RealismProfile(5, "Realistic Urban Operations", True, True, True, True, True, True, 0.06),
}

POOLS = {
    "normal": ["normal", "morning_peak", "evening_peak", "night_low", "industrial"],
    "leak": ["small_leak", "medium_leak", "large_leak", "catastrophic_burst", "progressive_leak"],
    "sensor": ["sensor_drift", "sensor_failure", "sensor_bias", "frozen_sensor"],
    "communication": ["packet_loss", "comm_delay", "duplicate_packets", "missing_readings"],
    "infrastructure": ["pipe_corrosion", "pipe_aging", "blockage", "water_hammer", "valve_malfunction"],
    "environment": ["heavy_rainfall", "ground_settlement", "construction_damage", "traffic_stress"],
    "complex": ["leak_high_demand", "leak_sensor_failure", "dual_leaks", "burst_valve_shutdown",
                "leak_corrosion_demand_surge"],
}


def compose_episode(level: int, rng: np.random.Generator, leak_prob: float = 0.8) -> list[str]:
    """Pick the scenarios that make up one episode at a given realism level."""
    names = [str(rng.choice(POOLS["normal"]))]
    if level >= 2 and rng.random() < 0.5:
        names.append(str(rng.choice(POOLS["communication"])))
    if level >= 3:
        if rng.random() < 0.6:
            names.append(str(rng.choice([s for s in POOLS["infrastructure"] if s != "pipe_corrosion"])))
        if rng.random() < 0.4:
            names.append(str(rng.choice(POOLS["sensor"])))
    if level >= 5:
        if rng.random() < 0.7:
            names.append(str(rng.choice(POOLS["environment"])))
        if rng.random() < 0.3:
            names.append("weekend")
    has_leak = any(SCENARIOS[n].has_leak for n in names)
    if not has_leak and rng.random() < leak_prob:
        pool = list(POOLS["leak"])
        if level >= 3:
            pool.append("pipe_corrosion")
        if level >= 4 and rng.random() < 0.6:
            pool = POOLS["complex"]
        names.append(str(rng.choice(pool)))
    return names


class ScenarioEngine:
    def __init__(self, network: WaterNetwork, sensors: SensorRegistry, events: list[Event], start: str | pd.Timestamp,
                 n_steps: int, step_minutes: int = 15, daily_mean_temp: float | None = None):
        self.net = network
        self.sensors = sensors
        self.events = list(events)
        self.start = pd.Timestamp(start)
        self.n_steps = n_steps
        self.step_minutes = step_minutes
        self.sph = 60 // step_minutes
        doy = self.start.dayofyear
        self.daily_mean_temp = daily_mean_temp if daily_mean_temp is not None else \
            27.0 + 4.5 * np.cos(2 * np.pi * (doy - 130) / 365.0)
        p = network.pipes
        n = network.nodes
        self.pipe_xy = 0.5 * (n.loc[p.start_node, ["x", "y"]].to_numpy(float) + n.loc[p.end_node, ["x", "y"]].to_numpy(float))
        self.pipe_ids = list(p.index)
        self.major = p.traffic_load.to_numpy(float) > 0.6

    # ------------------------------------------------------------------ factory
    @classmethod
    def from_scenarios(cls, network, sensors, names: list[str], start, n_steps, step_minutes, rng) -> "ScenarioEngine":
        ctx = ScenarioContext(network, sensors, n_steps, 60 // step_minutes, rng)
        events: list[Event] = []
        for name in names:
            events.extend(SCENARIOS[name].builder(ctx))
        return cls(network, sensors, events, start, n_steps, step_minutes)

    def add_event(self, event: Event) -> None:
        self.events.append(event)

    # ------------------------------------------------------------------ conditions
    def _ramp(self, ev: Event, step: int) -> float:
        """Cosine taper (1 h) so demand changes are smooth, as in reality."""
        ramp = self.sph
        a = np.clip((step - ev.start + 1) / ramp, 0, 1)
        b = 1.0 if ev.end is None else np.clip((ev.end - step) / ramp, 0, 1)
        return float(0.5 - 0.5 * np.cos(np.pi * min(a, b)))

    def conditions(self, step: int) -> Conditions:
        ts = self.start + pd.Timedelta(minutes=self.step_minutes * step)
        hour = ts.hour + ts.minute / 60.0
        weekend = ts.dayofweek >= 5 or any(e.kind == "weekend" for e in self.events)
        c = Conditions(step=step, timestamp=ts, hour=hour, weekend=weekend)
        temp = diurnal_temperature(hour, self.daily_mean_temp)
        rain = 0.0
        cum_rain = 0.0
        dt_h = self.step_minutes / 60.0
        for ev in self.events:
            if ev.kind == "rainfall":
                end = self.n_steps if ev.end is None else ev.end
                cum_rain += max(0, min(step, end) - ev.start) * dt_h * ev.params["mm_h"]
        for ev in self.events:
            if not ev.active(step):
                continue
            c.active_events.append(ev.event_id)
            k, P = ev.kind, ev.params
            if k == "demand_scale":
                f = 1.0 + (P["factor"] - 1.0) * self._ramp(ev, step)
                if "land_use" in P:
                    c.demand_mult_landuse[P["land_use"]] = c.demand_mult_landuse.get(P["land_use"], 1.0) * f
                elif "dma" in P:
                    c.demand_mult_dma[P["dma"]] = c.demand_mult_dma.get(P["dma"], 1.0) * f
                else:
                    c.demand_mult *= f
            elif k == "leak":
                hours = (step - ev.start) * dt_h
                mult = min(P.get("max_mult", 1.0), (1.0 + P.get("growth_per_hour", 0.0)) ** hours) \
                    if P.get("growth_per_hour", 0.0) > 0 else 1.0
                c.leaks.append(LeakSpec(ev.event_id, ev.target, P["frac"], P["area_m2"] * mult,
                                        P.get("favad_slope", 0.0) * mult, category=P.get("category", "medium"),
                                        origin=P.get("origin", "scenario")))
            elif k == "close_pipe":
                c.closed_pipes.add(ev.target)
                c.known_closures.add(ev.target)
                if step == ev.start and P.get("rapid"):
                    c.transient_links.append(ev.target)
            elif k == "valve_malfunction":
                c.throttled[ev.target] = P["k"]
            elif k == "blockage":
                c.blockage[ev.target] = P["fraction"]
            elif k == "aging":
                c.age_offset_years += P["years"]
            elif k == "corrosion":
                c.corrosion_boost[ev.target] = c.corrosion_boost.get(ev.target, 0.0) + P["boost"]
            elif k == "pump_trip":
                c.pump_forced_off = True
                if step == ev.start and P.get("rapid"):
                    c.transient_links.append("PU1")
            elif k == "rainfall":
                rain += P["mm_h"]
            elif k == "heat":
                temp += P["offset"]
            elif k == "ground_settlement":
                d = np.hypot(self.pipe_xy[:, 0] - P["x"], self.pipe_xy[:, 1] - P["y"])
                ramp = self._ramp(ev, step)
                for i in np.nonzero(d < P["radius"])[0]:
                    load = P["load_head"] * (1 - d[i] / P["radius"]) * ramp
                    c.external_load[self.pipe_ids[i]] = c.external_load.get(self.pipe_ids[i], 0.0) + load
            elif k == "traffic_stress":
                c.traffic_mult *= P["mult"]
                for i in np.nonzero(self.major)[0]:
                    c.external_load[self.pipe_ids[i]] = c.external_load.get(self.pipe_ids[i], 0.0) + P["load_head"]
            elif k in ("sensor_drift", "sensor_bias", "sensor_frozen", "sensor_failure"):
                c.sensor_faults[ev.target] = dict(type=k, start=ev.start, **P)
            elif k == "packet_loss":
                c.comm_loss_extra += P["extra_loss"]
            elif k == "comm_delay":
                c.comm_delay_mult *= P["mult"]
            elif k == "duplicates":
                c.comm_duplicate_prob = max(c.comm_duplicate_prob, P["prob"])
            elif k == "gateway_outage":
                s = self.sensors.table
                c.comm_outage_sensors |= set(s.sensor_id[(s.dma == ev.target) & (s.kind != "ami")])
        if rain > 0:
            temp -= 4.0
        c.temperature_c = float(temp)
        c.rainfall_mm_h = float(rain)
        c.ground_movement = float(np.clip(cum_rain / 120.0, 0, 0.6))
        return c

    def events_frame(self) -> pd.DataFrame:
        rows = []
        for e in self.events:
            d = e.to_dict()
            d["start_time"] = self.start + pd.Timedelta(minutes=self.step_minutes * e.start)
            rows.append(d)
        cols = ["event_id", "kind", "start", "end", "target", "params", "cause", "scenario", "start_time"]
        return pd.DataFrame(rows, columns=cols)
