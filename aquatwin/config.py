"""Central configuration for AquaTwin AI.

Every tunable number lives here so that experiments are reproducible from a
single YAML file (see ``configs/default.yaml``).  Nothing in the simulator is
hard-coded "magic"; physical constants and engineering design values are
documented where they are defined.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

GRAVITY = 9.81            # m/s^2
WATER_DENSITY = 1000.0    # kg/m^3


@dataclass
class NetworkConfig:
    seed: int = 42
    grid_x: int = 12                    # street intersections (east-west)
    grid_y: int = 10                    # street intersections (north-south)
    spacing_m: float = 220.0            # mean block length
    jitter: float = 0.18                # positional jitter as fraction of spacing
    edge_drop_frac: float = 0.10        # missing street segments (irregular blocks)
    diagonal_frac: float = 0.04         # arterial diagonal roads
    center_lat: float = 18.5204         # map anchor (any city; Pune by default)
    center_lon: float = 73.8567
    base_elevation_m: float = 552.0
    hill_height_m: float = 22.0
    dma_split: tuple[int, int] = (2, 2)  # District Metered Areas (cols, rows)
    population_density_km2: float = 16000.0
    lpcd: float = 135.0                 # litres / capita / day (CPHEEO norm)
    current_year: int = 2026
    n_tanks: int = 2
    tank_staging_m: float = 18.0        # elevated service reservoir staging height
    valve_fraction: float = 0.35        # share of pipes with isolation valves
    peak_factor: float = 2.5            # design peak factor for pipe sizing
    design_velocity_ms: float = 1.0     # pipe sizing rule


@dataclass
class SimulationConfig:
    timestep_minutes: int = 15
    episode_hours: int = 24
    start: str = "2026-06-01 00:00"
    pdd_pmin_m: float = 0.5             # pressure below which no demand is served
    pdd_preq_m: float = 12.0            # pressure at which full demand is served
    solver_tol: float = 1e-6
    solver_max_iter: int = 60
    demand_noise_sigma: float = 0.05    # consumer behavioural uncertainty (AR(1))
    demand_noise_rho: float = 0.85


@dataclass
class SensorConfig:
    pressure_coverage: float = 0.5      # share of junctions with pressure loggers
    flow_coverage: float = 0.20         # share of pipes with flow meters (+ mandatory)
    pressure_noise_m: float = 0.06      # 1-sigma random noise of a high-resolution logger
    pressure_resolution_m: float = 0.05
    flow_noise_frac: float = 0.015      # 1.5% of reading (electromagnetic meter)
    flow_noise_floor_ls: float = 0.05
    flow_resolution_ls: float = 0.01
    ami_noise_frac: float = 0.03        # customer meter (AMI) demand reading noise
    drift_rate_m_per_day: float = 0.6   # pressure drift when a sensor is drifting
    background_fault_rate: float = 0.03 # share of sensors with background faults (L3+)
    # communications (Gilbert-Elliott bursty channel)
    p_good_to_bad: float = 0.02
    p_bad_to_good: float = 0.25
    loss_in_bad: float = 0.6
    loss_in_good: float = 0.005
    delay_median_s: float = 3.0
    delay_sigma: float = 1.2
    duplicate_prob: float = 0.01
    late_cutoff_s: float = 600.0        # readings arriving later than this are missed


@dataclass
class DatasetConfig:
    output_dir: str = "data/generated"
    n_episodes: int = 24
    levels: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    localization_samples: int = 900
    localization_steps_per_sample: int = 3
    history_years: int = 25             # asset-history panel for failure model
    write_csv: bool = True
    seed: int = 7


@dataclass
class TrainingConfig:
    model_dir: str = "artifacts/models"
    report_dir: str = "reports"
    test_fraction: float = 0.25
    detection_max_neg_ratio: float = 25.0
    gnn_epochs: int = 60
    gnn_hidden: int = 64
    gnn_layers: int = 3
    gnn_lr: float = 3e-3
    gnn_architectures: list[str] = field(default_factory=lambda: ["gcn", "sage", "gat"])
    seed: int = 1


@dataclass
class AquaTwinConfig:
    network: NetworkConfig = field(default_factory=NetworkConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: str | Path | None = None) -> "AquaTwinConfig":
        cfg = cls()
        path = path or os.environ.get("AQUATWIN_CONFIG")
        if not path:
            return cfg
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        _merge(cfg, raw)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)


def _merge(obj: Any, raw: dict[str, Any]) -> None:
    names = {f.name for f in fields(obj)}
    for key, value in raw.items():
        if key not in names:
            raise KeyError(f"Unknown config key '{key}' for {type(obj).__name__}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        elif isinstance(current, tuple) and isinstance(value, list):
            setattr(obj, key, tuple(value))
        else:
            setattr(obj, key, value)
