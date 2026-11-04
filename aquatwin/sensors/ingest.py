"""Telemetry ingestion: the boundary between *any* data source and the AI.

Real deployments replace the simulator with SCADA/MQTT/AMI head-ends; they
deliver packets with exactly this schema::

    sensor_id, seq, timestamp, received_at, value, health

``align_snapshot`` de-duplicates by (sensor_id, seq), drops packets that
arrive after the latency cutoff and pivots to one as-of value per sensor and
time step -- the same contract the simulated Reality Engine honours.
"""
from __future__ import annotations

import pandas as pd

TELEMETRY_COLUMNS = ["sensor_id", "seq", "timestamp", "received_at", "value", "health"]


def deduplicate(packets: pd.DataFrame) -> pd.DataFrame:
    return packets.sort_values("received_at").drop_duplicates(["sensor_id", "seq"], keep="first")


def align_snapshot(packets: pd.DataFrame, cutoff_s: float = 600.0) -> pd.DataFrame:
    """Return a (timestamp x sensor_id) table of values usable by the AI services."""
    if packets.empty:
        return pd.DataFrame()
    p = deduplicate(packets)
    latency = (pd.to_datetime(p.received_at) - pd.to_datetime(p.timestamp)).dt.total_seconds()
    p = p[latency <= cutoff_s]
    return p.pivot_table(index="timestamp", columns="sensor_id", values="value", aggfunc="first")


def quality_report(packets: pd.DataFrame, expected_per_sensor: int) -> pd.DataFrame:
    p = packets.copy()
    p["latency_s"] = (pd.to_datetime(p.received_at) - pd.to_datetime(p.timestamp)).dt.total_seconds()
    g = p.groupby("sensor_id")
    rep = pd.DataFrame({
        "packets": g.size(),
        "duplicates": g["duplicate"].sum() if "duplicate" in p else 0,
        "median_latency_s": g.latency_s.median(),
        "p95_latency_s": g.latency_s.quantile(0.95),
    })
    rep["unique"] = deduplicate(p).groupby("sensor_id").size()
    rep["completeness"] = rep["unique"] / expected_per_sensor
    return rep
