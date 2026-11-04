"""Pipe material engineering properties.

Values are representative literature figures used throughout the water
industry.  Sources (paraphrased):

* Hazen-Williams C and its decay with age: AWWA M32, Sharp & Walski (1988)
  tuberculation studies for unlined cast iron.
* Break-rate growth (Shamir & Howard 1979 exponential model
  ``N(t) = N0 * exp(A * age)``) with N0/A in the range reported by
  Kleiner & Rajani (2001) for North American / Indian utilities.
* External corrosion pit growth ``d = k * age**n`` (Rossum 1969 /
  Romanoff soil studies), ``n`` ~ 0.3-0.6.
* FAVAD leak-area/pressure slope (May 1994; Van Zyl & Cassa 2014): rigid metal
  pipes have fixed-area leaks (N1 ~ 0.5), plastic pipes have expanding leaks
  (N1 up to 1.5).
* Pressure-wave celerity for Joukowsky surges (Wylie & Streeter).
* Tensile strength for Barlow hoop-stress capacity.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    design_life_yr: float
    hw_c_new: float              # Hazen-Williams C when new
    hw_c_decay_per_yr: float     # roughness growth (tuberculation)
    hw_c_min: float
    corrosion_k: float           # pit-depth coefficient (mm / yr**n) at unit soil corrosivity
    corrosion_n: float           # pit-depth exponent
    wall_mm_per_100mm: float     # wall thickness per 100 mm of diameter
    wall_min_mm: float
    break_n0: float              # breaks / km / yr at age 0 (Shamir-Howard)
    break_a: float               # exponential growth rate (1/yr)
    favad_slope: float           # leak-area expansion m^2 per m head per m^2 of initial area
    wave_speed_ms: float         # pressure wave celerity
    tensile_mpa: float           # hoop strength for Barlow formula
    brittle: bool                # brittle materials fail catastrophically (bursts)


MATERIALS: dict[str, Material] = {
    "PVC": Material("PVC", 60, 150, 0.05, 140, 0.015, 0.40, 4.5, 4.0,
                    0.030, 0.012, 0.020, 420, 45, True),
    "HDPE": Material("HDPE", 80, 150, 0.03, 140, 0.008, 0.40, 6.0, 5.0,
                     0.020, 0.010, 0.035, 300, 22, False),
    "Steel": Material("Steel", 50, 130, 0.55, 80, 0.80, 0.45, 3.0, 5.0,
                      0.035, 0.035, 0.0, 1100, 250, False),
    "Cast Iron": Material("Cast Iron", 75, 130, 0.85, 60, 0.95, 0.45, 6.0, 9.0,
                          0.040, 0.040, 0.0, 1200, 150, True),
    "Ductile Iron": Material("Ductile Iron", 90, 140, 0.30, 100, 0.35, 0.45, 4.0, 6.0,
                             0.025, 0.025, 0.0, 1100, 300, False),
}

MATERIAL_NAMES = list(MATERIALS)

# Commercially available nominal diameters (mm) used by the pipe-sizing rule.
COMMERCIAL_DIAMETERS_MM = [100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700]


def wall_thickness_mm(material: str, diameter_mm: float) -> float:
    m = MATERIALS[material]
    return max(m.wall_min_mm, m.wall_mm_per_100mm * diameter_mm / 100.0)
