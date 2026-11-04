"""Event and condition data structures shared by the Scenario and Truth engines."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_event_counter = itertools.count(1)

# leak size categories: equivalent orifice diameter range (mm)
LEAK_CATEGORIES = {
    "small": (3.0, 6.0),
    "medium": (8.0, 14.0),
    "large": (18.0, 30.0),
    "burst": (40.0, 90.0),
}
DISCHARGE_COEFF = 0.6


def leak_area_m2(eq_diameter_mm: float) -> float:
    return float(np.pi * (eq_diameter_mm / 1000.0) ** 2 / 4.0)


def eq_diameter_mm(area_m2: float) -> float:
    return float(np.sqrt(4.0 * max(area_m2, 0.0) / np.pi) * 1000.0)


@dataclass
class Event:
    kind: str
    start: int                       # step index (inclusive)
    end: int | None = None           # step index (exclusive); None = until end of episode
    target: str | None = None        # pipe_id / sensor_id / dma / node_id
    params: dict[str, Any] = field(default_factory=dict)
    cause: str = ""                  # human-readable causal explanation
    scenario: str = ""
    event_id: str = field(default_factory=lambda: f"EV{next(_event_counter):05d}")

    def active(self, step: int) -> bool:
        return step >= self.start and (self.end is None or step < self.end)

    def to_dict(self) -> dict[str, Any]:
        return {"event_id": self.event_id, "kind": self.kind, "start": self.start, "end": self.end,
                "target": self.target, "params": dict(self.params), "cause": self.cause,
                "scenario": self.scenario}


@dataclass
class LeakSpec:
    leak_id: str
    pipe_id: str
    frac: float                      # position along pipe from start node (0..1)
    area_m2: float                   # current initial orifice area A0
    favad_slope: float               # dA/dh (m^2 per m head)
    cd: float = DISCHARGE_COEFF
    category: str = "medium"
    origin: str = "scenario"         # scenario | stress_induced | construction | corrosion

    @property
    def eq_diameter_mm(self) -> float:
        return eq_diameter_mm(self.area_m2)


@dataclass
class Conditions:
    """Everything that defines the physical world at one time step."""
    step: int
    timestamp: Any
    hour: float
    weekend: bool
    temperature_c: float = 28.0
    rainfall_mm_h: float = 0.0
    ground_movement: float = 0.0                       # 0..1 network-wide index
    demand_mult: float = 1.0
    demand_mult_landuse: dict[str, float] = field(default_factory=dict)
    demand_mult_dma: dict[str, float] = field(default_factory=dict)
    leaks: list[LeakSpec] = field(default_factory=list)
    closed_pipes: set[str] = field(default_factory=set)
    throttled: dict[str, float] = field(default_factory=dict)     # pipe -> minor loss K
    blockage: dict[str, float] = field(default_factory=dict)      # pipe -> fraction
    age_offset_years: float = 0.0
    corrosion_boost: dict[str, float] = field(default_factory=dict)
    pump_forced_off: bool = False
    transient_links: list[str] = field(default_factory=list)      # links with sudden flow stoppage this step
    known_closures: set[str] = field(default_factory=set)         # operator actions the twin knows about
    external_load: dict[str, float] = field(default_factory=dict)  # pipe -> head-equivalent stress
    traffic_mult: float = 1.0
    sensor_faults: dict[str, dict[str, Any]] = field(default_factory=dict)
    comm_loss_extra: float = 0.0
    comm_delay_mult: float = 1.0
    comm_duplicate_prob: float = 0.0
    comm_outage_sensors: set[str] = field(default_factory=set)
    active_events: list[str] = field(default_factory=list)
