"""Data schema of the synthetic dataset (also exported to ``schema.json``).

Provenance classes
------------------
observation : what SCADA / AMI / loggers deliver (noisy, may be NaN)
twin        : the digital twin's expectation computed from observations
asset       : GIS / CMMS asset register (static per episode)
environment : weather & ground conditions (measured, low noise)
truth       : hidden ground truth -- NEVER used as a model input
label       : supervised learning targets derived from truth
meta        : bookkeeping
"""
from __future__ import annotations

PIPE_SCHEMA: dict[str, tuple[str, str, str]] = {
    "timestamp": ("meta", "datetime", "Measurement time (15-min resolution)"),
    "episode_id": ("meta", "str", "Simulation episode identifier"),
    "level": ("meta", "int", "Realism level 0-5"),
    "scenario": ("meta", "str", "Scenario names composing the episode"),
    "pipe_id": ("asset", "str", "Pipeline segment id"),
    "node_id": ("asset", "str", "Start junction of the pipe (monitoring reference node)"),
    "end_node": ("asset", "str", "End junction of the pipe"),
    "dma": ("asset", "str", "District Metered Area"),
    "pressure": ("observation", "m", "Observed pressure at node_id (NaN: unmonitored/missing)"),
    "pressure_end": ("observation", "m", "Observed pressure at end_node"),
    "expected_pressure": ("twin", "m", "Twin baseline pressure at node_id"),
    "expected_pressure_end": ("twin", "m", "Twin baseline pressure at end_node"),
    "flow": ("observation", "L/s", "Observed pipe flow (NaN: unmetered/missing)"),
    "expected_flow": ("twin", "L/s", "Twin baseline pipe flow"),
    "velocity": ("observation", "m/s", "Observed velocity = flow / area"),
    "demand": ("observation", "L/s", "AMI consumption at node_id"),
    "pipe_age": ("asset", "years", "Age at simulation time"),
    "material": ("asset", "str", "PVC / HDPE / Steel / Cast Iron / Ductile Iron"),
    "diameter_mm": ("asset", "mm", "Nominal diameter"),
    "length_m": ("asset", "m", "Segment length"),
    "corrosion": ("asset", "0-1", "Corrosion index from last condition assessment"),
    "repair_count": ("asset", "count", "Repairs on record"),
    "pipe_health": ("asset", "0-100", "Condition score in asset register"),
    "sensor_health": ("observation", "0-1", "Worst self-diagnostic score of sensors associated with the pipe"),
    "temperature": ("environment", "C", "Air temperature"),
    "rainfall": ("environment", "mm/h", "Rain intensity"),
    "ground_movement": ("environment", "0-1", "Ground movement index (soil saturation, settlement)"),
    "packet_loss": ("observation", "0/1", "Any associated telemetry packet lost this step"),
    "delay_seconds": ("observation", "s", "Max latency of associated packets"),
    "true_pressure": ("truth", "m", "True pressure at node_id (incl. transients)"),
    "true_flow": ("truth", "L/s", "True flow at pipe start"),
    "true_velocity": ("truth", "m/s", "True velocity"),
    "true_pipe_health": ("truth", "0-100", "Dynamic true condition"),
    "true_corrosion": ("truth", "0-1", "True corrosion index"),
    "water_loss": ("label", "L/s", "True leak outflow on this pipe"),
    "leak_size": ("label", "mm", "Equivalent leak orifice diameter (0 = none)"),
    "leak_location": ("label", "m", "Leak distance from node_id along the pipe (NaN = none)"),
    "leak_frac": ("label", "0-1", "Leak position as fraction of length"),
    "is_leak": ("label", "0/1", "Pipe has an active leak"),
    "failure_risk": ("label", "prob/yr", "True annual failure probability under current stress"),
}

NODE_SCHEMA: dict[str, tuple[str, str, str]] = {
    "timestamp": ("meta", "datetime", ""), "episode_id": ("meta", "str", ""), "step": ("meta", "int", ""),
    "node_id": ("asset", "str", ""), "node_type": ("asset", "str", ""), "dma": ("asset", "str", ""),
    "elevation": ("asset", "m", ""), "monitored": ("asset", "0/1", "Has a pressure/level logger"),
    "observed_pressure": ("observation", "m", ""), "expected_pressure": ("twin", "m", ""),
    "sensor_health": ("observation", "0-1", ""), "observed_demand": ("observation", "L/s", "AMI"),
    "true_pressure": ("truth", "m", ""), "true_demand": ("truth", "L/s", "Served demand"),
    "required_demand": ("truth", "L/s", "Demand before pressure deficiency"),
    "surge": ("truth", "m", "Transient surge head"), "isolated": ("truth", "0/1", "No supply path"),
}

LEAKAGE_COLUMNS = [c for c, (src, _, _) in PIPE_SCHEMA.items() if src in ("truth", "label")]
