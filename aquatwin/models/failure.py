"""Failure Prediction Service (predictive maintenance).

Two XGBoost regressors trained on the asset-history panel:

* ``failure_risk``          annual failure probability (trained on logit scale)
* ``remaining_useful_life`` years until the condition threshold is crossed

Validation uses GroupKFold by pipe so that no pipe appears in both train and
test (otherwise year-to-year autocorrelation inflates the scores).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from aquatwin.network.materials import MATERIAL_NAMES

FAILURE_FEATURES = ["pipe_age", "material_code", "repair_count", "corrosion", "pressure_cycles", "historical_leaks",
                    "diameter_mm", "length_m", "mean_pressure"]
PARAMS = dict(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
              min_child_weight=2, tree_method="hist", n_jobs=4)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["material_code"] = d.material.map({m: i for i, m in enumerate(MATERIAL_NAMES)}).astype(int)
    return d


def regression_metrics(y, p) -> dict[str, float]:
    return dict(mae=float(mean_absolute_error(y, p)), rmse=float(np.sqrt(mean_squared_error(y, p))),
                r2=float(r2_score(y, p)))


def _logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


@dataclass
class FailurePredictor:
    features: list[str] = field(default_factory=lambda: list(FAILURE_FEATURES))
    risk_model: xgb.XGBRegressor | None = None
    rul_model: xgb.XGBRegressor | None = None

    def fit(self, panel: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> dict:
        d = prepare(panel)
        X, groups = d[self.features], d.pipe_id
        y_r, y_l = _logit(d.failure_risk.to_numpy()), d.remaining_useful_life.to_numpy()
        oof_r, oof_l = np.zeros(len(d)), np.zeros(len(d))
        for tr, te in GroupKFold(n_splits=n_splits).split(X, groups=groups):
            m1 = xgb.XGBRegressor(**PARAMS, random_state=seed).fit(X.iloc[tr], y_r[tr])
            m2 = xgb.XGBRegressor(**PARAMS, random_state=seed).fit(X.iloc[tr], y_l[tr])
            oof_r[te], oof_l[te] = m1.predict(X.iloc[te]), m2.predict(X.iloc[te])
        self.risk_model = xgb.XGBRegressor(**PARAMS, random_state=seed).fit(X, y_r)
        self.rul_model = xgb.XGBRegressor(**PARAMS, random_state=seed).fit(X, y_l)
        risk_pred = _sigmoid(oof_r)
        report = dict(failure_risk=regression_metrics(d.failure_risk, risk_pred),
                      remaining_useful_life=regression_metrics(y_l, np.clip(oof_l, 0, None)),
                      n_rows=len(d), n_pipes=int(d.pipe_id.nunique()))
        # does predicted risk rank the pipes that actually broke the following year?
        if d.breaks_next_year.sum() > 0:
            report["break_auc"] = float(roc_auc_score((d.breaks_next_year > 0).astype(int), risk_pred))
            top = d.assign(p=risk_pred).sort_values("p", ascending=False)
            k = max(1, int(0.1 * len(top)))
            report["breaks_captured_top10pct"] = float(top.breaks_next_year.iloc[:k].sum() / d.breaks_next_year.sum())
        return report

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        d = prepare(df)
        X = d[self.features]
        return pd.DataFrame({"failure_risk": _sigmoid(self.risk_model.predict(X)),
                             "remaining_useful_life": np.clip(self.rul_model.predict(X), 0, None)}, index=df.index)

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.risk_model.save_model(d / "failure_risk.json")
        self.rul_model.save_model(d / "rul.json")
        (d / "failure_meta.json").write_text(json.dumps({"features": self.features}))

    @classmethod
    def load(cls, directory: str | Path) -> "FailurePredictor":
        d = Path(directory)
        r, l = xgb.XGBRegressor(), xgb.XGBRegressor()
        r.load_model(d / "failure_risk.json")
        l.load_model(d / "rul.json")
        return cls(features=json.loads((d / "failure_meta.json").read_text())["features"], risk_model=r, rul_model=l)


def current_asset_frame(network) -> pd.DataFrame:
    """Asset register (today) in the failure-model input schema."""
    p = network.pipes
    return pd.DataFrame({
        "pipe_id": p.pipe_id, "pipe_age": p.age_years, "material": p.material, "repair_count": p.repair_count,
        "corrosion": p.corrosion_index, "pressure_cycles": p.pressure_cycles, "historical_leaks": p.historical_leaks,
        "diameter_mm": p.diameter_mm, "length_m": p.length_m,
        "mean_pressure": [network.graph.edges[network.edge_of(pid)].get("est_pressure_m", 40.0) for pid in p.index],
    }, index=p.index)
