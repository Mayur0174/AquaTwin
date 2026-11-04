"""Digital Twin Control Panel backend: a live, steppable twin with AI in the loop.

The session runs the *same* CAUSE -> PHYSICS -> OBSERVATION chain as the
dataset generator, but interactively: operators inject events, the world
evolves step by step, and the AI services make decisions on observations
only.  Truth is kept alongside for scoring and for the "ground truth" layer
in the dashboard (which would simply be absent with real sensors).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from aquatwin.config import AquaTwinConfig
from aquatwin.data.generator import RowBuilder
from aquatwin.models.detection import LeakDetector
from aquatwin.models.failure import FailurePredictor, current_asset_frame
from aquatwin.models.features import detection_features
from aquatwin.pipeline import load_world
from aquatwin.scenarios.catalog import SCENARIOS, ScenarioContext
from aquatwin.scenarios.engine import ScenarioEngine
from aquatwin.scenarios.events import Event
from aquatwin.services.water_loss import dma_water_balance
from aquatwin.simulation import SimulationRun, StepRecord

WINDOW = 12        # steps of history used for streaming detection features
LOC_WINDOW = 24    # longer history so the frozen pre-alarm reference stays available
LOC_STEPS = 3      # localisation snapshots ensembled after onset (the GNN's training range)
CLEAR_STEPS = 8    # latched alarm: clears only after 2 h of normal readings (hysteresis)


@dataclass
class Decision:
    step: int
    timestamp: str
    kind: str                     # alarm | localisation | clear
    text: str
    payload: dict = field(default_factory=dict)


class DigitalTwinSession:
    def __init__(self, cfg: AquaTwinConfig | None = None, level: int = 3, seed: int = 0,
                 start: str | None = None, hours: int = 48, load_models: bool = True):
        self.cfg = cfg or AquaTwinConfig()
        d, md = Path(self.cfg.dataset.output_dir), Path(self.cfg.training.model_dir)
        self.net, self.sensors, self.ctx = load_world(d)
        self.level, self.seed = level, seed
        self.rows = RowBuilder(self.net, self.sensors)
        self.rng = np.random.default_rng(seed)
        sim = self.cfg.simulation
        self.n_steps = int(hours * 60 / sim.timestep_minutes)
        self.scenario = ScenarioEngine(self.net, self.sensors, [], start or sim.start, self.n_steps, sim.timestep_minutes)
        self.ctx_s = ScenarioContext(self.net, self.sensors, self.n_steps, 60 // sim.timestep_minutes, self.rng)
        self.run = SimulationRun(self.net, self.sensors, self.cfg, level, self.scenario, seed)
        self.records: list[StepRecord] = []
        self.pipe_frames: list[pd.DataFrame] = []
        self.node_frames: list[pd.DataFrame] = []
        self.decisions: list[Decision] = []
        self.probabilities: list[pd.Series] = []
        self.alarm_scores: list[float] = []
        self.alarm_streak = 0
        self.low_streak = 0
        self.loc_acc: list[tuple[np.ndarray, np.ndarray]] = []
        self.alarm_active = False
        self.alarm_onset: int | None = None
        self.localization: pd.DataFrame | None = None
        self.detector = self.failure = self.gnn = self.baseline = self.explainer = None
        if load_models:
            self._load_models(md)

    # ------------------------------------------------------------------ models
    def _load_models(self, md: Path) -> None:
        try:
            self.detector = LeakDetector.load(md / "detection")
            from aquatwin.xai.explain import TreeExplainer
            self.explainer = TreeExplainer(self.detector.model, self.detector.features)
        except Exception:
            self.detector = None
        try:
            self.failure = FailurePredictor.load(md / "failure")
        except Exception:
            self.failure = None
        try:
            from aquatwin.models.localization import GNNLocalizer, SensitivityLocalizer
            best = json.loads((md / "localization" / "best.json").read_text())["arch"]
            self.gnn = GNNLocalizer.load(md / "localization", self.ctx, best)
            self.baseline = SensitivityLocalizer.load(md / "localization", self.ctx)
        except Exception:
            self.gnn = None

    # ------------------------------------------------------------------ control panel actions
    @property
    def step_idx(self) -> int:
        return self.run.step_idx

    def _add(self, events: list[Event]) -> list[dict]:
        for e in events:
            self.scenario.add_event(e)
        return [e.to_dict() for e in events]

    def inject_leak(self, pipe_id: str | None = None, category: str = "medium", position: float | None = None,
                    growth_per_hour: float = 0.0) -> dict:
        now_h = self.step_idx / self.ctx_s.steps_per_hour
        ev = self.ctx_s.leak_event(category, now_h, pipe_id, growth_per_hour=growth_per_hour, max_mult=20.0,
                                   scenario="operator_injection")
        ev.start = self.step_idx
        if position is not None:
            ev.params["frac"] = float(np.clip(position, 0.02, 0.98))
        return self._add([ev])[0]

    def inject_burst(self, pipe_id: str | None = None) -> dict:
        return self.inject_leak(pipe_id, "burst")

    def inject_sensor_failure(self, sensor_id: str | None = None, mode: str = "sensor_failure") -> dict:
        sid = sensor_id or str(self.rng.choice(self.sensors.pressure.sensor_id))
        return self._add([Event(mode, self.step_idx, None, sid, dict(mode="nan", offset=1.5, rate_mult=5.0),
                                cause=f"Operator-injected {mode.replace('_', ' ')} on {sid}", scenario="operator_injection")])[0]

    def set_demand(self, factor: float, hours: float = 6.0, land_use: str | None = None) -> dict:
        p = dict(factor=float(factor))
        if land_use:
            p["land_use"] = land_use
        end = self.step_idx + int(hours * self.ctx_s.steps_per_hour)
        return self._add([Event("demand_scale", self.step_idx, end, params=p,
                                cause=f"Operator demand change x{factor:.2f} for {hours:.0f} h", scenario="operator")])[0]

    def set_weather(self, temperature_offset: float = 0.0, rainfall_mm_h: float = 0.0, hours: float = 6.0) -> list[dict]:
        end = self.step_idx + int(hours * self.ctx_s.steps_per_hour)
        evs = []
        if temperature_offset:
            evs.append(Event("heat", self.step_idx, end, params=dict(offset=float(temperature_offset)),
                             cause=f"Weather: {temperature_offset:+.0f} C", scenario="operator"))
        if rainfall_mm_h:
            evs.append(Event("rainfall", self.step_idx, end, params=dict(mm_h=float(rainfall_mm_h)),
                             cause=f"Weather: rain {rainfall_mm_h:.0f} mm/h", scenario="operator"))
        return self._add(evs)

    def close_pipe(self, pipe_id: str) -> dict:
        return self._add([Event("close_pipe", self.step_idx, None, pipe_id, dict(rapid=False),
                                cause=f"Operator isolates {pipe_id}", scenario="operator")])[0]

    def load_scenario(self, name: str) -> list[dict]:
        """Replay a catalogue scenario starting now (event times are relative to the current step)."""
        ctx = ScenarioContext(self.net, self.sensors, self.n_steps - self.step_idx, self.ctx_s.steps_per_hour, self.rng)
        evs = SCENARIOS[name].builder(ctx)
        for e in evs:
            e.start += self.step_idx
            if e.end is not None:
                e.end += self.step_idx
        return self._add(evs)

    # ------------------------------------------------------------------ simulation loop
    def step(self, n: int = 1) -> list[dict]:
        out = []
        for _ in range(n):
            if self.step_idx >= self.n_steps:
                break
            rec = self.run.step()
            self.records.append(rec)
            meta = dict(episode_id="LIVE", level=self.level, scenario="live")
            self.pipe_frames.append(self.rows.pipes_frame([rec], meta).assign(step=rec.cond.step))
            self.node_frames.append(self.rows.nodes_frame([rec], meta).assign(step=rec.cond.step))
            out.append(self._ai_step(rec))
        return out

    def _window(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return (pd.concat(self.pipe_frames[-WINDOW:], ignore_index=True),
                pd.concat(self.node_frames[-WINDOW:], ignore_index=True))

    def _ai_step(self, rec: StepRecord) -> dict:
        state = dict(step=rec.cond.step, timestamp=str(rec.cond.timestamp), true_loss_ls=rec.truth.total_loss_ls,
                     alarm=False, top_pipe=None, max_probability=None)
        if self.detector is None:
            return state
        P, N = self._window()
        F = detection_features(P, N, self.ctx)
        cur = F[F.step == rec.cond.step].copy()
        cur["leak_probability"] = self.detector.predict_proba(cur)
        cur["alarm_score"] = self.detector.alarm_score(cur, cur.leak_probability.to_numpy())
        self.probabilities.append(cur.set_index("pipe_id").leak_probability)
        self.alarm_scores.append(float(cur.alarm_score.max()))
        top = cur.sort_values("alarm_score", ascending=False).iloc[0]
        above = top.alarm_score >= self.detector.alarm_threshold
        self.alarm_streak = self.alarm_streak + 1 if above else 0
        low = top.alarm_score < 0.5 * self.detector.alarm_threshold
        self.low_streak = self.low_streak + 1 if low else 0
        ts = str(rec.cond.timestamp)
        if self.alarm_streak >= self.detector.alarm_persistence and not self.alarm_active:
            self.alarm_active = True
            self.alarm_onset = rec.cond.step - self.detector.alarm_persistence + 1
            row = cur[cur.pipe_id == top.pipe_id]
            from aquatwin.xai.explain import explain_detection
            ex = explain_detection(self.detector, self.explainer, row, self.detector.alarm_threshold)
            self.decisions.append(Decision(rec.cond.step, ts, "alarm", "LEAK ALARM - " + ex["text"],
                                           dict(pipe_id=top.pipe_id, probability=float(top.leak_probability),
                                                contributions=ex["contributions"].to_dict("records"))))
            self.loc_acc = []
            self._localize(rec, ts)
        elif self.alarm_active and self.low_streak >= CLEAR_STEPS:
            self.alarm_active = False
            self.decisions.append(Decision(rec.cond.step, ts, "clear",
                                           f"Alarm cleared: residuals normal for {CLEAR_STEPS * 15} min"))
        elif self.alarm_active and 0 < len(self.loc_acc) < LOC_STEPS:
            self._localize(rec, ts)   # temporal ensemble over the first steps after onset
        state.update(alarm=self.alarm_active, top_pipe=str(top.pipe_id), max_probability=float(top.leak_probability),
                     alarm_score=float(top.alarm_score))
        return state

    def _localize(self, rec: StepRecord, ts: str) -> None:
        if self.gnn is None:
            return
        from aquatwin.models.localization import _slog, predictions_table
        from aquatwin.models.features import dma_balance, node_matrices, snapshot_features
        P = pd.concat(self.pipe_frames[-LOC_WINDOW:], ignore_index=True)
        N = pd.concat(self.node_frames[-LOC_WINDOW:], ignore_index=True)
        m = node_matrices(N, self.ctx, "episode_id")["LIVE"]
        steps = list(m["steps"])
        t = steps.index(rec.cond.step)
        ref = steps.index(self.alarm_onset) if self.alarm_onset in steps else max(0, t - 1)
        g = P[P.step == rec.cond.step].set_index("pipe_id").loc[self.ctx.pipe_ids]
        flow, exp = g.flow.to_numpy(float), g.expected_flow.to_numpy(float)
        ufw, _ = dma_balance(flow[None], exp[None], m["demand"][t][None], self.ctx)
        x, ea = snapshot_features(m, t, flow, exp, self.ctx, ufw[0], self.baseline.S, ref_step=ref)
        x[:, :3] = _slog(x[:, :3])
        ea[:, 4] = _slog(ea[:, 4])
        from aquatwin.models.localization import Snapshot
        snap = Snapshot(x, ea, np.zeros(len(self.ctx.pipe_ids), np.float32), np.full(len(self.ctx.pipe_ids), np.nan),
                        "LIVE", rec.cond.step, self.level)
        prob, pos = self.gnn.predict([snap])
        self.loc_acc.append((prob[0], pos[0]))
        P_ = np.stack([a for a, _ in self.loc_acc])
        Q_ = np.stack([b for _, b in self.loc_acc])
        # geometric-mean ensemble: on a 112-leak benchmark (L2-L4) it lowered mean error from 317 m
        # (single onset snapshot) to 253 m and raised the <=250 m share from 64% to 70%
        lp_ = np.log(P_ + 1e-9).mean(0)
        prob_m = np.exp(lp_ - lp_.max())
        prob_m = prob_m / prob_m.sum()
        pos_m = (P_ * Q_).sum(0) / np.maximum(P_.sum(0), 1e-12)
        table = predictions_table(self.ctx, prob_m[None], pos_m[None], k=5)[0]
        sal = self.gnn.saliency(snap).head(5)
        self.localization = table
        best = table.iloc[0]
        k = len(self.loc_acc)
        if 1 < k < LOC_STEPS:
            return   # keep accumulating; publish the first estimate and the final ensemble
        label = "localises" if k == 1 else f"refines (ensemble of {k} snapshots)"
        text = (f"GNN ({self.gnn.arch}) {label} the leak on {best.pipe_id} at {best.distance_from_start_m:.0f} m "
                f"from {best.start_node} (p={best.probability:.0%}); most influential sensors: "
                + ", ".join(sal.index[:3]))
        self.decisions.append(Decision(rec.cond.step, ts, "localisation", text,
                                       dict(candidates=table.to_dict("records"), saliency=sal.round(4).to_dict())))

    # ------------------------------------------------------------------ views
    def pipes_df(self) -> pd.DataFrame:
        return pd.concat(self.pipe_frames, ignore_index=True) if self.pipe_frames else pd.DataFrame()

    def nodes_df(self) -> pd.DataFrame:
        return pd.concat(self.node_frames, ignore_index=True) if self.node_frames else pd.DataFrame()

    def water_balance(self) -> pd.DataFrame:
        if not self.pipe_frames:
            return pd.DataFrame()
        return dma_water_balance(self.pipes_df(), self.nodes_df(), self.ctx)

    def asset_risk(self) -> pd.DataFrame:
        cur = current_asset_frame(self.net)
        if self.failure is None:
            out = cur.assign(pred_failure_risk=self.net.pipes.failure_risk, pred_remaining_useful_life=self.net.pipes.rul_years)
        else:
            out = pd.concat([cur, self.failure.predict(cur).add_prefix("pred_")], axis=1)
        return out.reset_index(drop=True)

    def truth_leaks(self) -> list[dict]:
        return self.records[-1].truth.leaks if self.records else []

    def events(self) -> pd.DataFrame:
        return self.run.events_frame()

    def state(self) -> dict:
        if not self.records:
            return dict(step=0, started=False)
        r = self.records[-1]
        return dict(step=r.cond.step, timestamp=str(r.cond.timestamp), pump_on=r.truth.pump_on,
                    tank_levels=dict(zip(self.net.tank_ids, map(float, r.truth.tank_levels))),
                    total_demand_ls=float(r.truth.demand_served.sum()), true_loss_ls=r.truth.total_loss_ls,
                    temperature_c=r.truth.temperature_c, rainfall_mm_h=r.truth.rainfall_mm_h,
                    alarm=self.alarm_active, n_decisions=len(self.decisions),
                    min_pressure_m=float(np.nanmin(r.truth.pressure)))
