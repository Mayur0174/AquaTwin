"""Scenario catalogue.

Every scenario is a *cause*: a list of events that alter the physical world
(or the measurement chain).  The consequences are never scripted -- they are
computed by the hydraulic truth engine and the sensor reality engine.

Where a scenario does not name a specific asset, the asset is selected from
the physics-derived hazard field (break rate x length x environmental
multipliers), so old, corroded, repaired and stressed pipes fail more often
(Rules 3, 4, 6) -- never uniformly at random.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from aquatwin.network.materials import MATERIALS
from aquatwin.network.model import WaterNetwork
from aquatwin.physics.degradation import leak_opening_multiplier
from aquatwin.scenarios.events import LEAK_CATEGORIES, Event, leak_area_m2
from aquatwin.sensors.registry import SensorRegistry


@dataclass
class ScenarioContext:
    network: WaterNetwork
    sensors: SensorRegistry
    n_steps: int
    steps_per_hour: int
    rng: np.random.Generator
    used_pipes: set[str] = field(default_factory=set)

    def step_at(self, hour: float) -> int:
        return int(np.clip(round(hour * self.steps_per_hour), 0, self.n_steps - 1))

    @property
    def hours(self) -> float:
        return self.n_steps / self.steps_per_hour

    def hazard_weights(self, extra: np.ndarray | None = None, mask: np.ndarray | None = None) -> np.ndarray:
        p = self.network.pipes
        w = (p.break_rate * p.length_m / 1000.0).to_numpy() + 1e-6
        if extra is not None:
            w = w * extra
        if mask is not None:
            w = w * mask
        for i, pid in enumerate(p.index):
            if pid in self.used_pipes:
                w[i] = 0.0
        return w / w.sum()

    def sample_pipe(self, extra: np.ndarray | None = None, mask: np.ndarray | None = None) -> str:
        p = self.network.pipes
        pid = str(self.rng.choice(p.index.to_numpy(), p=self.hazard_weights(extra, mask)))
        self.used_pipes.add(pid)
        return pid

    def leak_event(self, category: str, start_h: float, pipe_id: str | None = None,
                   growth_per_hour: float = 0.0, max_mult: float = 1.0, end_h: float | None = None,
                   cause: str = "", scenario: str = "", origin: str = "scenario") -> Event:
        pipe_id = pipe_id or self.sample_pipe()
        self.used_pipes.add(pipe_id)
        pipe = self.network.pipes.loc[pipe_id]
        lo, hi = LEAK_CATEGORIES[category]
        d_eq = self.rng.uniform(lo, hi) * float(leak_opening_multiplier(pipe.corrosion_index))
        d_eq = min(d_eq, 0.8 * pipe.diameter_mm)
        area = leak_area_m2(d_eq)
        slope = MATERIALS[pipe.material].favad_slope * area
        frac = float(self.rng.uniform(0.05, 0.95))
        reason = cause or (f"{category} leak on {pipe_id}: {pipe.material}, {pipe.age_years:.0f} y old, "
                           f"corrosion {pipe.corrosion_index:.2f}, {pipe.repair_count} past repairs "
                           f"(hazard {pipe.break_rate:.2f} breaks/km/yr)")
        return Event("leak", self.step_at(start_h), None if end_h is None else self.step_at(end_h), pipe_id,
                     dict(area_m2=area, favad_slope=slope, frac=frac, category=category,
                          growth_per_hour=growth_per_hour, max_mult=max_mult, origin=origin),
                     cause=reason, scenario=scenario)


@dataclass(frozen=True)
class ScenarioDef:
    name: str
    category: str
    title: str
    rules: tuple[str, ...]
    builder: Callable[[ScenarioContext], list[Event]]
    min_level: int = 0
    has_leak: bool = False


def _rand_h(ctx: ScenarioContext, lo: float, hi: float) -> float:
    return float(ctx.rng.uniform(lo * ctx.hours / 24, hi * ctx.hours / 24))


# ------------------------------------------------------------------ NORMAL OPERATION
def _morning_peak(ctx):
    return [Event("demand_scale", ctx.step_at(6), ctx.step_at(10), params=dict(factor=1.18, land_use="residential"),
                  cause="School/office departure: residential morning peak 18% above typical", scenario="morning_peak")]


def _evening_peak(ctx):
    return [Event("demand_scale", ctx.step_at(18), ctx.step_at(22), params=dict(factor=1.15, land_use="residential"),
                  cause="Cooking/bathing evening peak 15% above typical", scenario="evening_peak")]


def _night_low(ctx):
    return [Event("demand_scale", ctx.step_at(0), ctx.step_at(5), params=dict(factor=0.85),
                  cause="Deep night low: minimal legitimate use, minimum night flow window", scenario="night_low")]


def _weekend(ctx):
    return [Event("weekend", 0, None, cause="Weekend: shifted, flatter residential peaks; commerce/industry reduced",
                  scenario="weekend")]


def _industrial(ctx):
    return [Event("demand_scale", ctx.step_at(_rand_h(ctx, 7, 12)), ctx.step_at(_rand_h(ctx, 15, 20)),
                  params=dict(factor=1.6, land_use="industrial"),
                  cause="Industrial estate batch process: industrial draw +60%", scenario="industrial")]


# ------------------------------------------------------------------ LEAKS
def _leak(category):
    def build(ctx):
        return [ctx.leak_event(category, _rand_h(ctx, 1, 18), scenario=f"{category}_leak")]
    return build


def _progressive(ctx):
    return [ctx.leak_event("small", _rand_h(ctx, 1, 8), growth_per_hour=0.12, max_mult=25.0,
                           scenario="progressive_leak")]


# ------------------------------------------------------------------ SENSOR EVENTS
def _target_sensor(ctx) -> str:
    sensors = ctx.sensors.pressure.sensor_id.tolist()
    return str(ctx.rng.choice(sensors))


def _sensor_event(kind, **params):
    def build(ctx):
        return [Event(kind, ctx.step_at(_rand_h(ctx, 2, 14)), None, _target_sensor(ctx), dict(params),
                      cause=f"Instrument fault: {kind.replace('_', ' ')}", scenario=kind)]
    return build


# ------------------------------------------------------------------ COMMUNICATION EVENTS
def _packet_loss(ctx):
    return [Event("packet_loss", ctx.step_at(_rand_h(ctx, 0, 10)), ctx.step_at(_rand_h(ctx, 14, 24)),
                  params=dict(extra_loss=0.25), cause="GSM/LoRa congestion: +25% packet loss", scenario="packet_loss")]


def _comm_delay(ctx):
    return [Event("comm_delay", ctx.step_at(_rand_h(ctx, 0, 10)), ctx.step_at(_rand_h(ctx, 14, 24)),
                  params=dict(mult=60.0), cause="Backhaul degradation: latency x60 (late readings)", scenario="comm_delay")]


def _duplicates(ctx):
    return [Event("duplicates", 0, None, params=dict(prob=0.08),
                  cause="Gateway retransmission bug: 8% duplicate packets", scenario="duplicate_packets")]


def _missing(ctx):
    dma = str(ctx.rng.choice(sorted(ctx.sensors.pressure.dma.unique())))
    return [Event("gateway_outage", ctx.step_at(_rand_h(ctx, 2, 12)), ctx.step_at(_rand_h(ctx, 14, 20)), dma,
                  cause=f"Gateway power outage: all loggers in {dma} silent", scenario="missing_readings")]


# ------------------------------------------------------------------ INFRASTRUCTURE EVENTS
def _corrosion(ctx):
    p = ctx.network.pipes
    pid = str(p.corrosion_index.idxmax())          # the most corroded main, deterministically
    ctx.used_pipes.add(pid)
    ev = [Event("corrosion", 0, None, pid, dict(boost=0.15),
                cause=f"Accelerated external corrosion on {pid} (aggressive soil {p.at[pid, 'soil_corrosivity']:.2f})",
                scenario="pipe_corrosion")]
    ev.append(ctx.leak_event("small", _rand_h(ctx, 4, 14), pid, growth_per_hour=0.06, max_mult=6.0,
                             cause=f"Corrosion pit perforates wall of {pid}: weeping leak that widens",
                             scenario="pipe_corrosion", origin="corrosion"))
    return ev


def _aging(ctx):
    years = float(ctx.rng.choice([10, 15, 20]))
    return [Event("aging", 0, None, params=dict(years=years),
                  cause=f"Asset fast-forward +{years:.0f} years: tuberculation lowers C, corrosion deepens",
                  scenario="pipe_aging")]


def _blockage(ctx):
    p = ctx.network.pipes
    cand = p[(p.material.isin(["Cast Iron", "Steel"])) | (p.age_years > 35)]
    pid = str(ctx.rng.choice(cand.index)) if len(cand) else str(ctx.rng.choice(p.index))
    return [Event("blockage", ctx.step_at(_rand_h(ctx, 1, 10)), None, pid, dict(fraction=0.55),
                  cause=f"Sediment/tuberculation blockage in {pid} (55% effective area loss)", scenario="blockage")]


def _water_hammer(ctx):
    t = _rand_h(ctx, 6, 18)
    return [Event("pump_trip", ctx.step_at(t), ctx.step_at(t + 0.75), "PU1", dict(rapid=True),
                  cause="Power failure trips the pump: sudden flow stoppage, Joukowsky surge", scenario="water_hammer")]


def _valve_malfunction(ctx):
    p = ctx.network.pipes
    cand = p[p.has_valve & (p.diameter_mm >= 150)]
    pid = str(ctx.rng.choice(cand.index)) if len(cand) else str(ctx.rng.choice(p.index))
    return [Event("valve_malfunction", ctx.step_at(_rand_h(ctx, 1, 12)), None, pid, dict(k=180.0),
                  cause=f"Valve on {pid} stuck ~85% closed (gearbox failure)", scenario="valve_malfunction")]


# ------------------------------------------------------------------ ENVIRONMENTAL EVENTS
def _rainfall(ctx):
    s, e = _rand_h(ctx, 0, 8), _rand_h(ctx, 12, 24)
    return [Event("rainfall", ctx.step_at(s), ctx.step_at(e), params=dict(mm_h=float(ctx.rng.uniform(12, 35))),
                  cause="Monsoon cloudburst: less outdoor use, saturated soil, cooler air", scenario="heavy_rainfall")]


def _settlement(ctx):
    nodes = ctx.network.nodes
    j = nodes[nodes.node_type == "junction"]
    c = j.sample(1, random_state=int(ctx.rng.integers(1e9))).iloc[0]
    return [Event("ground_settlement", ctx.step_at(_rand_h(ctx, 2, 14)), None, None,
                  dict(x=float(c.x), y=float(c.y), radius=350.0, load_head=float(ctx.rng.uniform(90, 160))),
                  cause=f"Ground settlement near {c.node_id}: differential soil movement loads buried mains",
                  scenario="ground_settlement")]


def _construction(ctx):
    p = ctx.network.pipes
    major = (p.traffic_load > 0.6).to_numpy().astype(float)
    pid = ctx.sample_pipe(mask=major + 0.05)
    return [ctx.leak_event("burst", _rand_h(ctx, 9, 17), pid, scenario="construction_damage", origin="construction",
                           cause=f"Excavator strike on {pid} during road works (third-party damage)")]


def _traffic(ctx):
    return [Event("traffic_stress", ctx.step_at(7), ctx.step_at(21), params=dict(mult=1.8, load_head=18.0),
                  cause="Heavy-vehicle diversion: cyclic surface loading on arterial mains", scenario="traffic_stress")]


# ------------------------------------------------------------------ COMPLEX EVENTS
def _leak_high_demand(ctx):
    ev = [Event("demand_scale", ctx.step_at(6), ctx.step_at(22), params=dict(factor=1.3),
                cause="Heatwave demand surge (+30%)", scenario="leak_high_demand"),
          Event("heat", 0, None, params=dict(offset=6.0), cause="Heatwave +6 C", scenario="leak_high_demand")]
    ev.append(ctx.leak_event("medium", _rand_h(ctx, 6, 16), scenario="leak_high_demand"))
    return ev


def _leak_sensor_failure(ctx):
    leak = ctx.leak_event("large", _rand_h(ctx, 4, 16), scenario="leak_sensor_failure")
    sensor = ctx.sensors.nearest_pressure_sensor(ctx.network, leak.target)
    return [leak, Event("sensor_failure", max(0, leak.start - ctx.steps_per_hour), None, sensor,
                        dict(mode="nan"), cause=f"Nearest logger {sensor} to the leak has a dead battery",
                        scenario="leak_sensor_failure")]


def _dual_leaks(ctx):
    return [ctx.leak_event("medium", _rand_h(ctx, 2, 10), scenario="dual_leaks"),
            ctx.leak_event(str(ctx.rng.choice(["small", "medium", "large"])), _rand_h(ctx, 6, 18), scenario="dual_leaks")]


def _burst_shutdown(ctx):
    burst = ctx.leak_event("burst", _rand_h(ctx, 3, 14), scenario="burst_valve_shutdown")
    response_h = float(ctx.rng.uniform(1.0, 2.5))
    return [burst, Event("close_pipe", burst.start + int(response_h * ctx.steps_per_hour), None, burst.target,
                         dict(rapid=False), cause=f"Crew slowly closes valves isolating {burst.target} {response_h:.1f} h after burst",
                         scenario="burst_valve_shutdown")]


def _leak_corrosion_surge(ctx):
    p = ctx.network.pipes
    ranked = p.sort_values("corrosion_index", ascending=False).index[:8]
    pid = str(ctx.rng.choice(ranked))
    return [Event("corrosion", 0, None, pid, dict(boost=0.1), cause=f"Advanced corrosion on {pid}",
                  scenario="leak_corrosion_demand_surge"),
            Event("demand_scale", ctx.step_at(5), ctx.step_at(23), params=dict(factor=1.35),
                  cause="Festival demand surge (+35%)", scenario="leak_corrosion_demand_surge"),
            ctx.leak_event("medium", _rand_h(ctx, 6, 14), pid, growth_per_hour=0.08, max_mult=5.0,
                           scenario="leak_corrosion_demand_surge", origin="corrosion",
                           cause=f"Corroded wall of {pid} fails under surge-time pressure cycling")]


SCENARIOS: dict[str, ScenarioDef] = {s.name: s for s in [
    # normal operation
    ScenarioDef("normal", "normal", "Normal Operation", ("Typical weekday diurnal demand; no faults",), lambda c: []),
    ScenarioDef("morning_peak", "normal", "Morning Demand Peak", (
        "Residential demand x1.18 between 06:00-10:00 on top of the diurnal pattern",
        "Pressure falls network-wide (Rule 7); tanks drain; pump runs continuously"), _morning_peak),
    ScenarioDef("evening_peak", "normal", "Evening Demand Peak", (
        "Residential demand x1.15 between 18:00-22:00", "Pressure falls most at nodes far from tanks"), _evening_peak),
    ScenarioDef("night_low", "normal", "Night Low Demand", (
        "All demand x0.85 between 00:00-05:00", "Pressure rises to its daily maximum; leak flows peak (FAVAD)",
        "Minimum Night Flow window used by the water-balance service"), _night_low),
    ScenarioDef("weekend", "normal", "Weekend Pattern", (
        "Weekend demand curves: residential peak shifts ~2 h later; commercial x0.6; industrial x0.55",), _weekend),
    ScenarioDef("industrial", "normal", "Industrial Consumption", (
        "Industrial estate demand x1.6 for a batch window", "Localised pressure dip in the industrial DMA"),
        _industrial),
    # leaks
    ScenarioDef("small_leak", "leak", "Small Leak", (
        "Orifice 3-6 mm (x corrosion widening), ~0.1-0.4 L/s", "Pipe chosen from hazard field (age/corrosion/repairs)",
        "Pressure signature typically below sensor noise -> detectable mostly through night-flow balance"),
        _leak("small"), has_leak=True),
    ScenarioDef("medium_leak", "leak", "Medium Leak", ("Orifice 8-14 mm, ~1-3 L/s", "Local pressure drop 0.1-1 m"),
                _leak("medium"), has_leak=True),
    ScenarioDef("large_leak", "leak", "Large Leak", ("Orifice 18-30 mm, ~5-15 L/s", "Clear pressure depression around leak"),
                _leak("large"), has_leak=True),
    ScenarioDef("catastrophic_burst", "leak", "Catastrophic Burst", (
        "Orifice 40-90 mm (<=80% of pipe diameter), tens of L/s",
        "Pressure-driven demand: downstream customers lose service (served demand < required)"),
        _leak("burst"), has_leak=True),
    ScenarioDef("progressive_leak", "leak", "Progressive Leak Growth", (
        "Starts small; orifice area grows 12%/h up to x25", "Loss and pressure anomaly grow monotonically (Rules 1-2)"),
        _progressive, has_leak=True),
    # sensors
    ScenarioDef("sensor_drift", "sensor", "Sensor Drift", ("Logger offset grows linearly (~0.6 m/day x5)",
                "Self-diagnostics only partly reveal drift"), _sensor_event("sensor_drift", rate_mult=5.0), min_level=1),
    ScenarioDef("sensor_failure", "sensor", "Sensor Failure", ("Logger stops reporting (NaN)", "sensor_health = 0"),
                _sensor_event("sensor_failure", mode="nan"), min_level=1),
    ScenarioDef("sensor_bias", "sensor", "Sensor Bias", ("Constant +/-1.5 m offset after mis-calibration",
                "Undetectable by self-diagnostics: sensor_health stays 1"), _sensor_event("sensor_bias", offset=1.5), min_level=1),
    ScenarioDef("frozen_sensor", "sensor", "Frozen Sensor", ("Logger repeats its last value (firmware hang)",
                "Flat-line detector lowers sensor_health after 4 identical readings"), _sensor_event("sensor_frozen"), min_level=1),
    # communications
    ScenarioDef("packet_loss", "communication", "Packet Loss", ("Extra 25% Bernoulli loss on top of bursty channel",), _packet_loss, min_level=2),
    ScenarioDef("comm_delay", "communication", "Communication Delay", ("Latency x60; packets later than cutoff are missed",), _comm_delay, min_level=2),
    ScenarioDef("duplicate_packets", "communication", "Duplicate Packets", ("8% of packets duplicated; ingestion de-duplicates by sequence id",), _duplicates, min_level=2),
    ScenarioDef("missing_readings", "communication", "Missing Readings", ("Gateway outage silences one DMA's loggers",), _missing, min_level=2),
    # infrastructure
    ScenarioDef("pipe_corrosion", "infrastructure", "Pipe Corrosion", (
        "Most-corroded main gains +0.15 corrosion index", "Wall perforates -> small leak growing 6%/h",
        "Health and burst strength drop (Rules 4, 9)"), _corrosion, min_level=3, has_leak=True),
    ScenarioDef("pipe_aging", "infrastructure", "Pipe Aging", (
        "All assets aged +10/15/20 y", "Hazen-Williams C decays (tuberculation) -> higher head loss, lower pressure",
        "Corrosion index and failure risk rise (Rule 3)"), _aging, min_level=3),
    ScenarioDef("blockage", "infrastructure", "Blockage", ("Effective area of an old main reduced 55%",
                "Pressure drops downstream, rises upstream; flow re-routes through loops"), _blockage, min_level=3),
    ScenarioDef("water_hammer", "infrastructure", "Water Hammer", (
        "Pump trip -> Joukowsky surge dH = c*dv/g decaying with distance from the pump",
        "Weak pipes face stress-strength failure risk (may burst)", "Tanks supply the network while the pump is off"),
        _water_hammer, min_level=3),
    ScenarioDef("valve_malfunction", "infrastructure", "Valve Malfunction", ("Valve stuck 85% closed (K=180)",
                "Large local head loss; downstream pressure deficit"), _valve_malfunction, min_level=3),
    # environment
    ScenarioDef("heavy_rainfall", "environment", "Heavy Rainfall", (
        "Rain 12-35 mm/h: residential demand -2%/mm/h (cap 12%), temperature -4 C",
        "Soil saturation raises ground-movement index -> higher break hazard"), _rainfall, min_level=5),
    ScenarioDef("ground_settlement", "environment", "Ground Settlement", (
        "Differential settlement in 350 m radius adds 90-160 m head-equivalent bending stress",
        "Stress-strength interference may burst weak pipes (sampled, physics-weighted)"), _settlement, min_level=5),
    ScenarioDef("construction_damage", "environment", "Construction Damage", (
        "Excavator strike on a pipe under a major road -> burst", "Third-party damage, independent of pipe age"),
        _construction, min_level=5, has_leak=True),
    ScenarioDef("traffic_stress", "environment", "Traffic Stress", (
        "Arterial mains get +18 m head-equivalent cyclic load 07:00-21:00", "Failure risk and health deteriorate"),
        _traffic, min_level=5),
    # complex
    ScenarioDef("leak_high_demand", "complex", "Leak + High Demand", (
        "Demand +30% masks leak pressure signature (common-mode drop)", "Detector must use spatial contrast"),
        _leak_high_demand, min_level=4, has_leak=True),
    ScenarioDef("leak_sensor_failure", "complex", "Leak + Sensor Failure", (
        "Large leak while the nearest logger is dead", "Localisation must rely on more distant sensors (Rule 5)"),
        _leak_sensor_failure, min_level=4, has_leak=True),
    ScenarioDef("dual_leaks", "complex", "Dual Leaks", ("Two independent leaks, overlapping in time",
                "Signatures superpose non-linearly through the network"), _dual_leaks, min_level=4, has_leak=True),
    ScenarioDef("burst_valve_shutdown", "complex", "Burst + Valve Shutdown", (
        "Burst followed by isolation 1-2.5 h later", "Isolation stops the loss but may cut supply downstream",
        "Valves are closed slowly (closure time >> 2L/c) to avoid water hammer"), _burst_shutdown, min_level=4, has_leak=True),
    ScenarioDef("leak_corrosion_demand_surge", "complex", "Leak + Corrosion + Demand Surge", (
        "Corroded main + 35% demand surge + growing leak", "All three interact: lower pressure, larger opening"),
        _leak_corrosion_surge, min_level=4, has_leak=True),
]}


def scenarios_by_category() -> dict[str, list[ScenarioDef]]:
    out: dict[str, list[ScenarioDef]] = {}
    for s in SCENARIOS.values():
        out.setdefault(s.category, []).append(s)
    return out
