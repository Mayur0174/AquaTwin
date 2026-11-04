"""AquaTwin AI REST API (FastAPI).

Run:  ``uvicorn aquatwin.api.main:app --port 8000``  ->  docs at ``/docs``.

The API exposes the digital-twin control panel, the AI services and a
telemetry ingestion endpoint.  The ingestion contract is identical to the
simulated Reality Engine's packet format, so real SCADA/MQTT/AMI feeds can be
connected without changing the AI services.
"""
from __future__ import annotations

import threading
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from aquatwin import __version__
from aquatwin.config import AquaTwinConfig
from aquatwin.scenarios.catalog import SCENARIOS
from aquatwin.sensors.ingest import align_snapshot, quality_report

app = FastAPI(title="AquaTwin AI", version=__version__,
              description="Physics-informed digital twin for leak detection, localisation and predictive maintenance")
_lock = threading.Lock()
_state: dict[str, Any] = {"session": None, "telemetry": []}


def _session():
    if _state["session"] is None:
        raise HTTPException(409, "No simulation running: POST /simulation/start first")
    return _state["session"]


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    return obj


# ---------------------------------------------------------------------- schemas
class StartRequest(BaseModel):
    level: int = Field(3, ge=0, le=5, description="Realism level 0-5")
    seed: int = 0
    start: str | None = Field(None, examples=["2026-06-01 00:00"])
    hours: int = Field(48, ge=1, le=24 * 14)
    config_path: str | None = None


class LeakRequest(BaseModel):
    pipe_id: str | None = Field(None, description="Pipe id; omitted -> chosen from the physics hazard field")
    category: Literal["small", "medium", "large", "burst"] = "medium"
    position: float | None = Field(None, ge=0, le=1)
    growth_per_hour: float = 0.0


class SensorFailureRequest(BaseModel):
    sensor_id: str | None = None
    mode: Literal["sensor_failure", "sensor_drift", "sensor_bias", "sensor_frozen"] = "sensor_failure"


class DemandRequest(BaseModel):
    factor: float = Field(1.3, gt=0, le=5)
    hours: float = 6.0
    land_use: Literal["residential", "commercial", "industrial"] | None = None


class WeatherRequest(BaseModel):
    temperature_offset: float = 0.0
    rainfall_mm_h: float = 0.0
    hours: float = 6.0


class Packet(BaseModel):
    sensor_id: str
    seq: int
    timestamp: str
    received_at: str
    value: float
    health: float = 1.0


# ---------------------------------------------------------------------- system
@app.get("/health")
def health():
    return {"status": "ok", "version": __version__, "simulation": _state["session"] is not None}


@app.get("/scenarios")
def scenarios():
    return [dict(name=s.name, category=s.category, title=s.title, rules=list(s.rules), min_level=s.min_level)
            for s in SCENARIOS.values()]


# ---------------------------------------------------------------------- digital twin control
@app.post("/simulation/start")
def start(req: StartRequest):
    from aquatwin.twin.session import DigitalTwinSession
    cfg = AquaTwinConfig.load(req.config_path)
    with _lock:
        _state["session"] = DigitalTwinSession(cfg, req.level, req.seed, req.start, req.hours)
    return {"started": True, "level": req.level, "steps": _state["session"].n_steps}


@app.post("/simulation/step")
def step(n: int = 1):
    with _lock:
        return _clean(_session().step(n))


@app.get("/simulation/state")
def state():
    return _clean(_session().state())


@app.post("/inject/leak")
def inject_leak(req: LeakRequest):
    with _lock:
        return _clean(_session().inject_leak(req.pipe_id, req.category, req.position, req.growth_per_hour))


@app.post("/inject/burst")
def inject_burst(pipe_id: str | None = None):
    with _lock:
        return _clean(_session().inject_burst(pipe_id))


@app.post("/inject/sensor-failure")
def inject_sensor_failure(req: SensorFailureRequest):
    with _lock:
        return _clean(_session().inject_sensor_failure(req.sensor_id, req.mode))


@app.post("/control/demand")
def control_demand(req: DemandRequest):
    with _lock:
        return _clean(_session().set_demand(req.factor, req.hours, req.land_use))


@app.post("/control/weather")
def control_weather(req: WeatherRequest):
    with _lock:
        return _clean(_session().set_weather(req.temperature_offset, req.rainfall_mm_h, req.hours))


@app.post("/control/close-pipe/{pipe_id}")
def close_pipe(pipe_id: str):
    with _lock:
        return _clean(_session().close_pipe(pipe_id))


@app.post("/scenarios/{name}/replay")
def replay(name: str):
    if name not in SCENARIOS:
        raise HTTPException(404, f"unknown scenario {name}")
    with _lock:
        return _clean(_session().load_scenario(name))


@app.get("/events")
def events():
    df = _session().events()
    return _clean(df.astype(object).where(df.notna(), None).to_dict("records"))


# ---------------------------------------------------------------------- AI services
@app.get("/ai/decisions")
def decisions():
    return _clean([d.__dict__ for d in _session().decisions])


@app.get("/ai/detection")
def detection(top: int = 10):
    s = _session()
    if not s.probabilities:
        return []
    p = s.probabilities[-1].sort_values(ascending=False).head(top)
    return [{"pipe_id": k, "leak_probability": float(v)} for k, v in p.items()]


@app.get("/ai/localization")
def localization():
    s = _session()
    return [] if s.localization is None else _clean(s.localization.to_dict("records"))


@app.get("/ai/failure-risk")
def failure_risk(top: int = 20):
    df = _session().asset_risk().sort_values("pred_failure_risk", ascending=False).head(top)
    return _clean(df.to_dict("records"))


@app.get("/ai/water-loss")
def water_loss():
    bal = _session().water_balance()
    if bal.empty:
        return []
    last = bal[bal.step == bal.step.max()]
    return _clean(last[["dma", "unaccounted_ls", "estimated_loss_ls", "consumption_ls"]].to_dict("records"))


@app.get("/network/geojson")
def network_geojson():
    return _clean(_session().net.to_geojson())


@app.get("/network/pipes/{pipe_id}")
def pipe_details(pipe_id: str):
    s = _session()
    if pipe_id not in s.net.pipes.index:
        raise HTTPException(404, "unknown pipe")
    return _clean(s.net.pipes.loc[pipe_id].to_dict())


# ---------------------------------------------------------------------- real-sensor ingestion
@app.post("/ingest/telemetry")
def ingest(packets: list[Packet]):
    """Accept SCADA/AMI packets (same schema as the simulator). De-duplicated and latency-gated."""
    _state["telemetry"].extend(p.model_dump() for p in packets)
    df = pd.DataFrame(_state["telemetry"])
    snap = align_snapshot(df)
    return {"accepted": len(packets), "buffered": len(df), "timestamps": int(len(snap)),
            "sensors": int(snap.shape[1]) if len(snap) else 0}


@app.get("/ingest/quality")
def ingest_quality():
    df = pd.DataFrame(_state["telemetry"])
    if df.empty:
        return []
    df["duplicate"] = df.duplicated(["sensor_id", "seq"])
    n = df.groupby("sensor_id").seq.nunique().max()
    return _clean(quality_report(df, int(n)).reset_index().to_dict("records"))
