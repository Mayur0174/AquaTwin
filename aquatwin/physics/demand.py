"""Water demand physics: diurnal patterns, weekly cycle and weather response.

Rule 8 (demand follows time-of-day behaviour) is encoded as smooth hourly
pattern curves per land-use class.  Demand also responds causally to weather
(hot days -> more use, rain -> less outdoor use).  The only stochastic part is
a small, temporally-correlated consumer-behaviour term (environmental
uncertainty), which never changes the shape of the underlying pattern.
"""
from __future__ import annotations

import numpy as np

# Hourly multipliers (mean ~1.0).  Residential: morning (6-9h) and evening
# (18-21h) peaks typical of Indian intermittent-to-24x7 transition cities.
RESIDENTIAL = np.array([0.38, 0.32, 0.30, 0.30, 0.42, 0.85, 1.55, 1.95, 1.80, 1.40,
                        1.10, 0.98, 0.95, 0.90, 0.85, 0.88, 1.00, 1.25, 1.62, 1.75,
                        1.48, 1.10, 0.75, 0.50])
RESIDENTIAL_WEEKEND = np.array([0.42, 0.35, 0.32, 0.30, 0.34, 0.50, 0.90, 1.35, 1.75, 1.85,
                                1.60, 1.30, 1.15, 1.05, 0.95, 0.95, 1.05, 1.25, 1.50, 1.60,
                                1.42, 1.12, 0.80, 0.55])
COMMERCIAL = np.array([0.25, 0.20, 0.20, 0.20, 0.22, 0.30, 0.55, 0.95, 1.45, 1.70,
                       1.75, 1.70, 1.60, 1.65, 1.70, 1.65, 1.55, 1.45, 1.30, 1.10,
                       0.80, 0.55, 0.38, 0.30])
COMMERCIAL_WEEKEND = COMMERCIAL * 0.6 + 0.15
INDUSTRIAL = np.array([0.80, 0.80, 0.80, 0.80, 0.80, 0.85, 1.15, 1.20, 1.20, 1.20,
                       1.20, 1.15, 1.05, 1.15, 1.20, 1.20, 1.20, 1.15, 1.05, 0.95,
                       0.90, 0.85, 0.80, 0.80])
INDUSTRIAL_WEEKEND = INDUSTRIAL * 0.55

PATTERNS = {
    ("residential", False): RESIDENTIAL / RESIDENTIAL.mean(),
    ("residential", True): RESIDENTIAL_WEEKEND / RESIDENTIAL_WEEKEND.mean(),
    ("commercial", False): COMMERCIAL / COMMERCIAL.mean(),
    ("commercial", True): COMMERCIAL_WEEKEND / COMMERCIAL.mean(),
    ("industrial", False): INDUSTRIAL / INDUSTRIAL.mean(),
    ("industrial", True): INDUSTRIAL_WEEKEND / INDUSTRIAL.mean(),
}
LAND_USES = ("residential", "commercial", "industrial")


def pattern_value(land_use: str, hour_of_day: float, weekend: bool) -> float:
    """Smooth (periodic linear interpolation) hourly multiplier."""
    pat = PATTERNS[(land_use, weekend)]
    h0 = int(np.floor(hour_of_day)) % 24
    frac = hour_of_day - np.floor(hour_of_day)
    return float(pat[h0] * (1 - frac) + pat[(h0 + 1) % 24] * frac)


def weather_factor(land_use: str, temperature_c: float, rainfall_mm_h: float) -> float:
    """Hot weather raises residential/commercial use; rainfall suppresses outdoor use."""
    if land_use == "industrial":
        temp = 1.0 + 0.004 * (temperature_c - 27.0)          # cooling-water load
    else:
        temp = 1.0 + 0.015 * (temperature_c - 27.0)
    rain = 1.0 - min(0.12, 0.02 * rainfall_mm_h) if land_use == "residential" else 1.0
    return max(0.5, temp * rain)


def diurnal_temperature(hour_of_day: float, daily_mean_c: float = 28.0, amplitude_c: float = 5.5) -> float:
    """Temperature peaks ~15:00, minimum ~05:00."""
    return daily_mean_c + amplitude_c * np.sin(2 * np.pi * (hour_of_day - 9.0) / 24.0)


class ConsumerNoise:
    """AR(1) multiplicative noise per consumer node: correlated behavioural uncertainty."""

    def __init__(self, n_nodes: int, sigma: float, rho: float, rng: np.random.Generator):
        self.sigma = sigma
        self.rho = rho
        self.rng = rng
        self.state = rng.normal(0, sigma, n_nodes) if sigma > 0 else np.zeros(n_nodes)

    def step(self) -> np.ndarray:
        if self.sigma <= 0:
            return np.ones_like(self.state)
        innov = self.rng.normal(0, self.sigma * np.sqrt(1 - self.rho ** 2), self.state.shape)
        self.state = self.rho * self.state + innov
        return np.clip(1.0 + self.state, 0.6, 1.4)
