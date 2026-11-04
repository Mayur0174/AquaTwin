"""Sensor Reality Engine: transforms the truth into what SCADA/AMI would see.

Randomness lives here (and only here + environmental uncertainty):

* instrument noise (Gaussian) + quantisation + thermal sensitivity
* instrument faults: drift (linear growth), bias (constant), frozen (stuck
  value), failure (no data)
* communications: Gilbert-Elliott two-state bursty channel, log-normal
  latency (late packets are missed by the as-of snapshot), duplicates,
  gateway outages

Every reading carries its provenance (status, delay, duplicate flag,
sequence number) so the ingestion layer and the AI can reason about data
quality.  The truth is never modified.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from aquatwin.config import SensorConfig
from aquatwin.network.model import WaterNetwork
from aquatwin.scenarios.engine import RealismProfile
from aquatwin.scenarios.events import Conditions
from aquatwin.sensors.registry import SensorRegistry
from aquatwin.truth.engine import TruthSnapshot

FLATLINE_STEPS = 4


@dataclass
class Observation:
    step: int
    timestamp: pd.Timestamp
    sensor_ids: list[str]
    value: np.ndarray            # NaN when missing (lost / late / failed / outage)
    true_value: np.ndarray
    health: np.ndarray           # device self-diagnostic + data-quality score (0..1)
    lost: np.ndarray             # packet lost / outage
    late: np.ndarray             # arrived after cutoff
    delay_s: np.ndarray
    status: np.ndarray
    packets: list[dict] = field(default_factory=list)

    def as_series(self) -> pd.Series:
        return pd.Series(self.value, index=self.sensor_ids)


class RealityEngine:
    def __init__(self, network: WaterNetwork, registry: SensorRegistry, cfg: SensorConfig,
                 profile: RealismProfile, rng: np.random.Generator, step_minutes: int = 15):
        self.net = network
        self.reg = registry
        self.cfg = cfg
        self.profile = profile
        self.rng = rng
        self.dt_days = step_minutes / 1440.0
        t = registry.table
        self.ids = t.sensor_id.tolist()
        self.kind = t.kind.to_numpy()
        self.n = len(self.ids)
        nidx = network.node_index
        pidx = network.pipe_index
        self.target_idx = np.array([nidx.get(tg, pidx.get(tg, -1)) if k in ("pressure", "level", "ami")
                                    else pidx.get(tg, -1) for k, tg in zip(self.kind, t.target)])
        self.tank_pos = {tid: i for i, tid in enumerate(network.tank_ids)}
        self.targets = t.target.tolist()
        self.noise = t.noise.to_numpy(float)
        self.res = t.resolution.to_numpy(float)
        self.sign = np.where(rng.random(self.n) < 0.5, -1.0, 1.0)
        self.calib_age_days = rng.uniform(0, 365, self.n) if profile.degradation else np.zeros(self.n)
        self.channel_bad = np.zeros(self.n, bool)
        self.seq = np.zeros(self.n, int)
        self.frozen_value = np.full(self.n, np.nan)
        self.last_obs = np.full(self.n, np.nan)
        self.flat_count = np.zeros(self.n, int)
        self.fault_start: dict[int, int] = {}
        # background faults (realism level >= 3): physical degradation of instruments
        self.background: dict[int, dict] = {}
        if profile.background_faults > 0:
            cand = np.nonzero(np.isin(self.kind, ["pressure", "flow"]))[0]
            k = int(round(profile.background_faults * len(cand)))
            for i in rng.choice(cand, size=k, replace=False):
                self.background[int(i)] = dict(type=str(rng.choice(["sensor_drift", "sensor_bias", "sensor_frozen",
                                                                     "sensor_failure"], p=[0.45, 0.3, 0.15, 0.1])),
                                               start=int(rng.integers(0, 48)), rate_mult=1.0, offset=0.8, mode="nan")

    # ------------------------------------------------------------------ helpers
    def _true_values(self, truth: TruthSnapshot) -> np.ndarray:
        v = np.full(self.n, np.nan)
        for i, (k, ti) in enumerate(zip(self.kind, self.target_idx)):
            if k == "pressure":
                v[i] = truth.pressure[ti]
            elif k == "level":
                v[i] = truth.tank_levels[self.tank_pos[self.targets[i]]]
            elif k == "flow":
                v[i] = truth.flow[ti] if ti >= 0 else truth.pump_flow
            elif k == "ami":
                v[i] = truth.demand_served[ti]
        return v

    def observe(self, truth: TruthSnapshot, cond: Conditions) -> Observation:
        cfg, prof = self.cfg, self.profile
        true_v = self._true_values(truth)
        v = true_v.copy()
        status = np.array(["ok"] * self.n, dtype=object)
        health = np.ones(self.n)
        if prof.degradation:
            health -= 0.1 * np.minimum(1.0, (self.calib_age_days + cond.step * self.dt_days) / 365.0)

        # --- instrument noise
        if prof.sensor_noise:
            is_p = np.isin(self.kind, ["pressure", "level"])
            is_f = self.kind == "flow"
            is_a = self.kind == "ami"
            v[is_p] += self.rng.normal(0, self.noise[is_p]) + 0.002 * (truth.temperature_c - 25.0)
            sig_f = np.maximum(cfg.flow_noise_floor_ls, self.noise[is_f] * np.abs(v[is_f]))
            v[is_f] += self.rng.normal(0, sig_f)
            v[is_a] *= 1.0 + self.rng.normal(0, self.noise[is_a])
            v = np.round(v / self.res) * self.res

        # --- instrument faults (scenario + background)
        faults = {}
        for sid, f in cond.sensor_faults.items():
            if sid in self.reg.table.index:
                faults[self.ids.index(sid)] = f
        for i, f in self.background.items():
            if cond.step >= f["start"] and i not in faults:
                faults[i] = f
        for i, f in faults.items():
            days = (cond.step - f["start"]) * self.dt_days
            scale = 1.0 if self.kind[i] in ("pressure", "level") else max(abs(true_v[i]), 1.0) * 0.05
            typ = f["type"]
            if typ == "sensor_drift":
                off = self.sign[i] * cfg.drift_rate_m_per_day * f.get("rate_mult", 1.0) * days * scale
                v[i] += off
                status[i] = "drift"
                health[i] -= 0.5 * min(1.0, abs(off) / (8 * max(self.noise[i], 0.05) * scale))
            elif typ == "sensor_bias":
                v[i] += self.sign[i] * f.get("offset", 1.0) * scale
                status[i] = "bias"
            elif typ == "sensor_frozen":
                if np.isnan(self.frozen_value[i]):
                    self.frozen_value[i] = v[i] if np.isnan(self.last_obs[i]) else self.last_obs[i]
                v[i] = self.frozen_value[i]
                status[i] = "frozen"
            elif typ == "sensor_failure":
                v[i] = np.nan if f.get("mode", "nan") == "nan" else 0.0
                status[i] = "failed"
                health[i] = 0.0

        # flat-line detector (data-quality part of sensor health)
        same = np.isclose(v, self.last_obs, atol=1e-9, equal_nan=False) & np.isin(self.kind, ["pressure", "flow"])
        self.flat_count = np.where(same, self.flat_count + 1, 0)
        health = np.where(self.flat_count >= FLATLINE_STEPS, np.minimum(health, 0.3), health)
        self.last_obs = np.where(np.isnan(v), self.last_obs, v)

        # --- communications
        lost = np.zeros(self.n, bool)
        late = np.zeros(self.n, bool)
        delay = np.zeros(self.n)
        dup_p = cond.comm_duplicate_prob
        if prof.comms:
            to_bad = self.rng.random(self.n) < cfg.p_good_to_bad
            to_good = self.rng.random(self.n) < cfg.p_bad_to_good
            self.channel_bad = np.where(self.channel_bad, ~to_good, to_bad)
            p_loss = np.where(self.channel_bad, cfg.loss_in_bad, cfg.loss_in_good) + cond.comm_loss_extra
            lost = self.rng.random(self.n) < p_loss
            if cond.comm_outage_sensors:
                lost |= np.array([sid in cond.comm_outage_sensors for sid in self.ids])
            delay = np.exp(np.log(cfg.delay_median_s * cond.comm_delay_mult) + cfg.delay_sigma * self.rng.normal(size=self.n))
            late = (~lost) & (delay > cfg.late_cutoff_s)
            dup_p = max(dup_p, cfg.duplicate_prob)
        delay = np.where(lost, np.nan, delay)
        status = np.where(lost, "lost", np.where(late, "late", status))

        packets = []
        self.seq += 1
        dups = self.rng.random(self.n) < dup_p if dup_p > 0 else np.zeros(self.n, bool)
        for i in np.nonzero(~lost & ~np.isnan(v))[0]:
            rec = dict(sensor_id=self.ids[i], seq=int(self.seq[i]), timestamp=cond.timestamp,
                       received_at=cond.timestamp + pd.Timedelta(seconds=float(delay[i])), value=float(v[i]),
                       health=float(health[i]), duplicate=False)
            packets.append(rec)
            if dups[i]:
                d2 = dict(rec)
                d2["duplicate"] = True
                d2["received_at"] = rec["received_at"] + pd.Timedelta(seconds=float(self.rng.uniform(1, 30)))
                packets.append(d2)

        snapshot = np.where(lost | late, np.nan, v)
        return Observation(step=cond.step, timestamp=cond.timestamp, sensor_ids=self.ids, value=snapshot,
                           true_value=true_v, health=np.clip(health, 0, 1), lost=lost, late=late,
                           delay_s=delay, status=status, packets=packets)
