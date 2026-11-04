"""Asset deterioration physics: corrosion, roughness, break hazard, health, RUL.

All functions are vectorised (numpy) and deterministic.  The only place where
randomness touches asset condition is the *realisation* of failures from a
hazard rate (``sample_*`` helpers take an explicit ``rng``) -- the hazard
itself is a deterministic function of age, material, corrosion, stress and
repair history, which guarantees the immutable rules:

* Rule 3  older pipes -> higher hazard (Shamir-Howard exponential growth)
* Rule 4  corrosion -> higher hazard and larger leak openings
* Rule 6  repairs -> higher hazard / lower health (self-exciting break history)
* Rule 9  health decreases with age and stress
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from aquatwin.config import GRAVITY, WATER_DENSITY
from aquatwin.network.materials import MATERIALS, wall_thickness_mm

# Cox proportional-hazard coefficients (dimensionless multipliers on log-hazard)
BETA_CORROSION = 1.3
BETA_REPAIR = 0.15
BETA_PRESSURE = 0.025      # per metre above 30 m
BETA_CYCLES = 0.02         # per daily pressure cycle
BETA_TRAFFIC = 0.5
BETA_GROUND = 0.9
DIAMETER_REF_MM = 150.0
DIAMETER_EXP = 0.6

HEALTH_FAILURE_THRESHOLD = 35.0   # health below which a pipe is "end of life"
# residual-strength / stress-concentration factor (graphitised cast iron is the weakest)
STRENGTH_KNOCKDOWN = {"PVC": 0.6, "HDPE": 0.6, "Steel": 0.4, "Ductile Iron": 0.4, "Cast Iron": 0.28}
STRENGTH_COV = 0.22               # coefficient of variation of in-situ strength


def _mat(materials, attr: str) -> np.ndarray:
    return np.array([getattr(MATERIALS[m], attr) for m in np.atleast_1d(materials)], dtype=float)


def hazen_williams_c(materials, age, blockage=0.0) -> np.ndarray:
    """Roughness coefficient decays linearly with age down to a floor."""
    c_new = _mat(materials, "hw_c_new")
    decay = _mat(materials, "hw_c_decay_per_yr")
    c_min = _mat(materials, "hw_c_min")
    c = np.maximum(c_min, c_new - decay * np.asarray(age, float))
    return c * (1.0 - 0.5 * np.asarray(blockage, float))


def corrosion_depth_mm(materials, age, soil_corrosivity) -> np.ndarray:
    k = _mat(materials, "corrosion_k")
    n = _mat(materials, "corrosion_n")
    return k * np.asarray(soil_corrosivity, float) * np.power(np.maximum(age, 0.0), n)


def corrosion_index(materials, diameter_mm, age, soil_corrosivity) -> np.ndarray:
    """Fraction of wall lost to corrosion / degradation (0 = new, 1 = perforated)."""
    mats = np.atleast_1d(materials)
    wall = np.array([wall_thickness_mm(m, d) for m, d in zip(mats, np.broadcast_to(diameter_mm, mats.shape))])
    return np.clip(corrosion_depth_mm(mats, age, soil_corrosivity) / wall, 0.0, 0.98)


def stress_index(pressure_m, cycles_per_day, traffic) -> np.ndarray:
    p = np.asarray(pressure_m, float)
    return np.clip(0.4 * (p - 30.0) / 30.0 + 0.3 * np.asarray(cycles_per_day, float) / 30.0
                   + 0.3 * np.asarray(traffic, float), 0.0, 1.0)


def break_rate_per_km_yr(materials, age, diameter_mm, corrosion, repairs, pressure_m,
                         cycles_per_day, traffic, ground_movement) -> np.ndarray:
    """Shamir-Howard baseline x Cox covariates -> breaks / km / year."""
    n0 = _mat(materials, "break_n0")
    a = _mat(materials, "break_a")
    base = n0 * np.exp(a * np.asarray(age, float))
    cov = (BETA_CORROSION * np.asarray(corrosion, float)
           + BETA_REPAIR * np.minimum(np.asarray(repairs, float), 8.0)
           + BETA_PRESSURE * np.maximum(np.asarray(pressure_m, float) - 30.0, -15.0)
           + BETA_CYCLES * np.asarray(cycles_per_day, float)
           + BETA_TRAFFIC * np.asarray(traffic, float)
           + BETA_GROUND * np.asarray(ground_movement, float))
    dia = np.power(DIAMETER_REF_MM / np.asarray(diameter_mm, float), DIAMETER_EXP)
    return base * np.exp(cov) * dia


def annual_failure_probability(rate_per_km_yr, length_m) -> np.ndarray:
    return 1.0 - np.exp(-np.asarray(rate_per_km_yr) * np.asarray(length_m) / 1000.0)


def health_score(materials, age, corrosion, repairs, stress) -> np.ndarray:
    """0-100 condition score (100 = as new)."""
    life = _mat(materials, "design_life_yr")
    ageing = 1.0 - 0.45 * np.power(np.minimum(np.asarray(age, float) / life, 1.5), 1.6)
    h = (100.0 * np.power(1.0 - np.asarray(corrosion, float), 0.7) * np.clip(ageing, 0.05, 1.0)
         * np.power(0.92, np.asarray(repairs, float)) * (1.0 - 0.15 * np.asarray(stress, float)))
    return np.clip(h, 0.0, 100.0)


def remaining_useful_life(materials, diameter_mm, age, soil, repairs, stress,
                          horizon_yr: int = 120) -> np.ndarray:
    """Years until health drops below the end-of-life threshold (forward projection)."""
    mats = np.atleast_1d(materials)
    age = np.asarray(age, float)
    years = np.arange(0, horizon_yr + 1, dtype=float)
    rul = np.full(len(mats), float(horizon_yr))
    for i in range(len(mats)):
        a = age[i] + years
        ci = corrosion_index(np.repeat(mats[i], len(years)), np.repeat(np.atleast_1d(diameter_mm)[i], len(years)),
                             a, np.repeat(np.atleast_1d(soil)[i], len(years)))
        h = health_score(np.repeat(mats[i], len(years)), a, ci, np.atleast_1d(repairs)[i], np.atleast_1d(stress)[i])
        below = np.nonzero(h < HEALTH_FAILURE_THRESHOLD)[0]
        if len(below):
            rul[i] = years[below[0]]
    return rul


def strength_head_m(materials, diameter_mm, corrosion) -> np.ndarray:
    """In-situ burst capacity (metres of head): Barlow hoop stress with residual strength."""
    mats = np.atleast_1d(materials)
    d = np.broadcast_to(np.asarray(diameter_mm, float), mats.shape)
    wall = np.array([wall_thickness_mm(m, di) for m, di in zip(mats, d)])
    sigma = _mat(mats, "tensile_mpa") * 1e6
    p_burst = 2.0 * sigma * (wall / 1000.0) / (d / 1000.0)
    head = p_burst / (WATER_DENSITY * GRAVITY)
    kd = np.array([STRENGTH_KNOCKDOWN[m] for m in mats])
    return head * kd * np.power(1.0 - np.asarray(corrosion, float), 1.2)


def acute_failure_probability(stress_head, strength_head) -> np.ndarray:
    """Stress-strength interference: P(strength < stress) with uncertain in-situ strength."""
    s = np.asarray(strength_head, float)
    return norm.cdf((np.asarray(stress_head, float) - s) / (STRENGTH_COV * s))


def leak_opening_multiplier(corrosion) -> np.ndarray:
    """Corroded walls tear open wider for the same initiating defect (Rule 4)."""
    return 1.0 + 1.5 * np.asarray(corrosion, float)
