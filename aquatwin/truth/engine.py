"""Hydraulic Truth Engine: the Ground-Truth World.

``TruthEngine.step(conditions)`` advances the physical world by one time step
(extended-period simulation):

1. demand  <- land-use diurnal pattern x weather x scenario x consumer AR(1)
2. assets  <- age offset, corrosion, roughness (C), blockage
3. control <- pump on/off from tank levels (SCADA-style rule) or forced trip
4. physics <- GGA solve with PDD + FAVAD leak emitters
5. transients <- Joukowsky surge for sudden flow stoppage (pump trip, fast closure)
6. tanks   <- level integration from net inflow
7. failure <- stress-strength interference for event-induced stresses
8. condition <- health / failure-risk update from stress exposure

Nothing here looks at sensors.  The resulting :class:`TruthSnapshot` is the
single source of truth; the Reality Engine only *observes* it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aquatwin.config import GRAVITY, SimulationConfig
from aquatwin.network.materials import MATERIALS
from aquatwin.network.model import WaterNetwork
from aquatwin.physics import degradation as deg
from aquatwin.physics.demand import ConsumerNoise, pattern_value, weather_factor
from aquatwin.physics.hydraulics import solve
from aquatwin.scenarios.events import Conditions, Event, LeakSpec, leak_area_m2
from aquatwin.truth.builder import ModelBuilder, solve_with_tank_rules

SURGE_DECAY_M = 1500.0


@dataclass
class TruthSnapshot:
    step: int
    timestamp: Any
    # nodes (base network order)
    pressure: np.ndarray            # true pressure incl. transient surge, m
    head: np.ndarray
    demand_required: np.ndarray     # L/s
    demand_served: np.ndarray       # L/s
    surge: np.ndarray               # m
    isolated: np.ndarray
    # pipes (network.pipes order)
    flow: np.ndarray                # L/s at the start-side section
    flow_end: np.ndarray            # L/s at the end-side section
    velocity: np.ndarray            # m/s
    leak_flow: np.ndarray           # L/s (true water loss)
    leak_size_mm: np.ndarray        # equivalent orifice diameter
    leak_frac: np.ndarray           # NaN if no leak
    leak_distance_m: np.ndarray     # metres from start node, NaN if none
    health: np.ndarray
    failure_risk: np.ndarray
    corrosion: np.ndarray
    is_open: np.ndarray
    # system
    pump_on: bool
    pump_flow: float
    tank_levels: np.ndarray
    temperature_c: float
    rainfall_mm_h: float
    ground_movement: float
    leaks: list[dict] = field(default_factory=list)
    tank_valves_closed: set = field(default_factory=set)   # altitude valves shut (SCADA digital status)
    new_events: list[Event] = field(default_factory=list)
    converged: bool = True
    mass_balance_error: float = 0.0

    @property
    def total_loss_ls(self) -> float:
        return float(self.leak_flow.sum())


class TruthEngine:
    def __init__(self, network: WaterNetwork, sim: SimulationConfig, env_rng: np.random.Generator,
                 demand_noise: bool = True):
        self.net = network
        self.sim = sim
        self.rng = env_rng
        self.builder = ModelBuilder(network)
        nodes, pipes = network.nodes, network.pipes
        self.dt_h = sim.timestep_minutes / 60.0
        self.tank_levels = nodes.loc[self.builder.tank_ids, "init_level"].to_numpy(float).copy()
        self.tank_min = nodes.loc[self.builder.tank_ids, "min_level"].to_numpy(float)
        self.tank_max = nodes.loc[self.builder.tank_ids, "max_level"].to_numpy(float)
        self.pump_on = True
        self.base_demand = nodes.base_demand_ls.fillna(0.0).to_numpy(float)
        self.land_use = nodes.land_use.fillna("none").to_numpy()
        self.dma = nodes.dma.fillna("").to_numpy()
        self.noise = ConsumerNoise(len(nodes), sim.demand_noise_sigma if demand_noise else 0.0,
                                   sim.demand_noise_rho, env_rng)
        self.material = pipes.material.to_numpy()
        self.age = pipes.age_years.to_numpy(float)
        self.dia_mm = pipes.diameter_mm.to_numpy(float)
        self.length = pipes.length_m.to_numpy(float)
        self.soil = pipes.soil_corrosivity.to_numpy(float)
        self.traffic = pipes.traffic_load.to_numpy(float)
        self.repairs = pipes.repair_count.to_numpy(float)
        self.cycles = pipes.pressure_cycles.to_numpy(float)
        self.health = pipes.health_score.to_numpy(float).copy()
        self.damage = np.zeros(len(pipes))                # accumulated event damage (health points)
        self.wave_speed = np.array([MATERIALS[m].wave_speed_ms for m in self.material])
        self.dynamic_leaks: list[LeakSpec] = []
        self.failed: set[str] = set()
        self.prev_flow = None
        self.prev_head = None
        self.prev_sig = None
        self.cum_loss_m3 = 0.0
        D = network.distance_matrix
        self.node_ids = list(nodes.index)
        self._dist = D.reindex(index=self.node_ids, columns=self.node_ids).fillna(1e6).to_numpy()
        self.pipe_ids = list(pipes.index)
        self._pid = {p: i for i, p in enumerate(self.pipe_ids)}
        xy = nodes[["x", "y"]].to_numpy(float)
        self.pipe_xy = 0.5 * (xy[self.builder.p_start] + xy[self.builder.p_end])

    # ------------------------------------------------------------------ demand
    def required_demand_ls(self, cond: Conditions, noise: bool = True) -> np.ndarray:
        mult = np.ones(len(self.base_demand))
        for i, lu in enumerate(self.land_use):
            if lu == "none":
                mult[i] = 0.0
                continue
            m = pattern_value(lu, cond.hour, cond.weekend) * weather_factor(lu, cond.temperature_c, cond.rainfall_mm_h)
            m *= cond.demand_mult * cond.demand_mult_landuse.get(lu, 1.0) * cond.demand_mult_dma.get(self.dma[i], 1.0)
            mult[i] = m
        d = self.base_demand * mult
        if noise:
            d = d * self.noise.step()
        return d

    # ------------------------------------------------------------------ assets
    def asset_state(self, cond: Conditions) -> tuple[np.ndarray, np.ndarray]:
        age = self.age + cond.age_offset_years
        ci = deg.corrosion_index(self.material, self.dia_mm, age, self.soil)
        for pid, boost in cond.corrosion_boost.items():
            ci[self._pid[pid]] = min(0.98, ci[self._pid[pid]] + boost)
        hw = deg.hazen_williams_c(self.material, age)
        return ci, hw

    # ------------------------------------------------------------------ step
    def step(self, cond: Conditions) -> TruthSnapshot:
        b = self.builder
        demand_ls = self.required_demand_ls(cond)
        ci, hw = self.asset_state(cond)

        # pump control (level-based) unless forced off by an event
        frac = (self.tank_levels - self.tank_min) / (self.tank_max - self.tank_min)
        if self.pump_on and np.all(frac > 0.92):
            self.pump_on = False
        elif not self.pump_on and np.any(frac < 0.35):
            self.pump_on = True
        pump_running = self.pump_on and not cond.pump_forced_off
        speed = np.array([1.0 if pump_running else 0.0])

        leaks = list(cond.leaks) + self.dynamic_leaks
        # EPANET tank rule: an empty tank cannot discharge, a full tank cannot fill.
        warm = (self.prev_sig, self.prev_head, self.prev_flow) if self.prev_sig is not None else None
        model, bmap, res, closed_now = solve_with_tank_rules(b, demand_ls / 1000.0, self.tank_levels, self.tank_min,
                                                 self.tank_max, speed, hw, leaks, cond.closed_pipes, cond.throttled,
                                                 cond.blockage, self.sim, warm)
        self.prev_sig, self.prev_head, self.prev_flow = bmap.signature, res.head, res.flow

        n0, np_ = bmap.n_base_nodes, len(self.pipe_ids)
        flow = res.flow[bmap.first_seg] * 1000.0
        flow_end = res.flow[bmap.last_seg] * 1000.0
        area = np.pi * (self.dia_mm / 1000.0) ** 2 / 4.0
        velocity = flow / 1000.0 / area

        # transients (Joukowsky) for sudden flow stoppage on flagged links
        surge = np.zeros(n0)
        for link in cond.transient_links:
            if link in self._pid:
                i = self._pid[link]
                v_prev = abs(self._last_velocity[i]) if hasattr(self, "_last_velocity") else abs(velocity[i])
                src = b.p_start[i]
                c = self.wave_speed[i]
            else:  # pump trip: surge originates at the pumping station, celerity of the pumping main
                main = int(np.argmax(self.dia_mm))
                v_prev = abs(self._last_velocity[main]) if hasattr(self, "_last_velocity") else 0.0
                src = b.p_start[main] if self.node_ids[b.p_start[main]] == "PS1" else b.p_end[main]
                c = self.wave_speed[main]
            dh = c * v_prev / GRAVITY
            surge = np.maximum(surge, dh * np.exp(-self._dist[src] / SURGE_DECAY_M))
        self._last_velocity = velocity.copy()

        pressure = res.pressure[:n0].copy()
        pressure[b.tank_idx] = self.tank_levels
        pressure = pressure + surge * (~b.fixed[:n0])

        # tank level integration (snapshot reports the levels the hydraulics were solved with)
        levels_now = self.tank_levels.copy()
        net_in = np.zeros(len(b.tank_idx))
        for k, t in enumerate(b.tank_idx):
            inflow = res.flow[model.end == t].sum() - res.flow[model.start == t].sum()
            net_in[k] = inflow
        self.tank_levels = np.clip(self.tank_levels + net_in * self.dt_h * 3600.0 / b.tank_area,
                                   self.tank_min, self.tank_max)

        # leaks -> per pipe truth
        leak_flow = np.zeros(np_)
        leak_size = np.zeros(np_)
        leak_frac = np.full(np_, np.nan)
        leak_rows = []
        for j, lk in enumerate(bmap.leaks):
            i = bmap.leak_pipe_idx[j]
            q = float(res.emitter_flow[j] * 1000.0)
            leak_flow[i] = q
            leak_size[i] = lk.eq_diameter_mm
            leak_frac[i] = lk.frac
            leak_rows.append(dict(leak_id=lk.leak_id, pipe_id=lk.pipe_id, frac=lk.frac,
                                  distance_m=lk.frac * self.length[i], eq_diameter_mm=lk.eq_diameter_mm,
                                  flow_ls=q, category=lk.category, origin=lk.origin))
        self.cum_loss_m3 += leak_flow.sum() / 1000.0 * self.dt_h * 3600.0

        # stress-strength interference for event-induced stresses
        p_ends = np.maximum(pressure[b.p_start], pressure[b.p_end])
        p_static = np.maximum(res.pressure[b.p_start], res.pressure[b.p_end])
        surge_p = np.maximum(surge[b.p_start], surge[b.p_end])
        ext = np.array([cond.external_load.get(pid, 0.0) for pid in self.pipe_ids])
        new_events: list[Event] = []
        if surge_p.max() > 1.0 or ext.max() > 0.0:
            strength = deg.strength_head_m(self.material, self.dia_mm, ci) * np.power(np.clip(self.health / 100, 0.2, 1), 0.3)
            p_evt = np.clip(deg.acute_failure_probability(p_ends + ext, strength)
                            - deg.acute_failure_probability(p_static, strength), 0.0, 1.0) * min(1.0, self.dt_h)
            draws = self.rng.random(np_)
            for i in np.nonzero(draws < p_evt)[0]:
                pid = self.pipe_ids[i]
                if pid in self.failed or any(lk.pipe_id == pid for lk in leaks):
                    continue
                brittle = MATERIALS[self.material[i]].brittle
                d_eq = min(0.6 * self.dia_mm[i], (55.0 if brittle else 16.0) * float(deg.leak_opening_multiplier(ci[i])))
                a = leak_area_m2(d_eq)
                lk = LeakSpec(f"SL-{pid}-{cond.step}", pid, float(self.rng.uniform(0.1, 0.9)), a,
                              MATERIALS[self.material[i]].favad_slope * a,
                              category="burst" if brittle else "medium", origin="stress_induced")
                self.dynamic_leaks.append(lk)
                self.failed.add(pid)
                cause = (f"Stress-strength failure of {pid} ({self.material[i]}, corrosion {ci[i]:.2f}): "
                         f"stress {p_ends[i] + ext[i]:.0f} m vs strength {strength[i]:.0f} m "
                         f"(surge {surge_p[i]:.0f} m, external load {ext[i]:.0f} m)")
                new_events.append(Event("leak", cond.step, None, pid,
                                        dict(area_m2=a, frac=lk.frac, category=lk.category, origin="stress_induced"),
                                        cause=cause, scenario="emergent"))

        # condition dynamics: damage from surge fatigue + external loads, corrosion changes
        self.damage += self.dt_h * (0.02 * ext / 10.0) + 0.05 * surge_p / 10.0
        traffic = np.clip(self.traffic * cond.traffic_mult, 0, 1)
        stress = deg.stress_index(p_static, self.cycles, traffic)
        age = self.age + cond.age_offset_years
        local_gm = np.clip(cond.ground_movement + ext / 200.0, 0, 1)
        self.health = np.clip(deg.health_score(self.material, age, ci, self.repairs, stress) - self.damage, 0, 100)
        rate = deg.break_rate_per_km_yr(self.material, age, self.dia_mm, ci, self.repairs, p_static, self.cycles,
                                        traffic, local_gm)
        risk = deg.annual_failure_probability(rate, self.length)

        is_open = np.array([pid not in cond.closed_pipes for pid in self.pipe_ids])
        return TruthSnapshot(
            step=cond.step, timestamp=cond.timestamp, pressure=pressure, head=res.head[:n0],
            demand_required=demand_ls, demand_served=res.demand_served[:n0] * 1000.0, surge=surge,
            isolated=res.isolated[:n0], flow=flow, flow_end=flow_end, velocity=velocity, leak_flow=leak_flow,
            leak_size_mm=leak_size, leak_frac=leak_frac, leak_distance_m=leak_frac * self.length,
            health=self.health.copy(), failure_risk=risk, corrosion=ci, is_open=is_open,
            pump_on=pump_running, pump_flow=float(res.flow[bmap.pump_links].sum() * 1000.0),
            tank_levels=levels_now, temperature_c=cond.temperature_c,
            rainfall_mm_h=cond.rainfall_mm_h, ground_movement=cond.ground_movement, leaks=leak_rows,
            tank_valves_closed=set(closed_now) - set(cond.closed_pipes),
            new_events=new_events, converged=res.converged, mass_balance_error=res.mass_balance_error)


class TwinBaseline:
    """The digital twin's *expectation*: same physics, but only knowledge an operator has.

    Uses observed AMI consumption, observed tank levels, known pump status and
    known operator actions -- never the hidden leaks, blockages or malfunctions.
    The difference between observations and this baseline is the anomaly
    residual consumed by the AI services.
    """

    def __init__(self, network: WaterNetwork, sim: SimulationConfig):
        self.net = network
        self.sim = sim
        self.builder = ModelBuilder(network)
        self.prev = None
        self.pipe_ids = list(network.pipes.index)
        nodes = network.nodes
        self.tank_min = nodes.loc[self.builder.tank_ids, "min_level"].to_numpy(float)
        self.tank_max = nodes.loc[self.builder.tank_ids, "max_level"].to_numpy(float)

    def expected(self, demand_ls: np.ndarray, tank_levels: np.ndarray, pump_on: bool,
                 known_closures: set[str], age_offset: float = 0.0) -> dict[str, np.ndarray]:
        p = self.net.pipes
        hw = deg.hazen_williams_c(p.material.to_numpy(), p.age_years.to_numpy(float) + age_offset)
        # altitude-valve states are telemetered, so tank closures arrive in known_closures
        reported_levels = np.asarray(tank_levels, float).copy()
        tank_levels = np.clip(tank_levels, self.tank_min + 1e-3, self.tank_max - 1e-3)
        model, bmap, res, _ = solve_with_tank_rules(
            self.builder, np.nan_to_num(demand_ls) / 1000.0, tank_levels, self.tank_min, self.tank_max,
            np.array([1.0 if pump_on else 0.0]), hw, [], known_closures, sim=self.sim,
            warm=((), self.prev[0], self.prev[1]) if self.prev else None, tol_level=1e-6)
        self.prev = (res.head, res.flow)
        n0 = bmap.n_base_nodes
        pressure = res.pressure[:n0].copy()
        pressure[self.builder.tank_idx] = reported_levels
        return {"pressure": pressure, "flow": res.flow[bmap.first_seg] * 1000.0,
                "pump_flow": float(res.flow[bmap.pump_links].sum() * 1000.0)}
