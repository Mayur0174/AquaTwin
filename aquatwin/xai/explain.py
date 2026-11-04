"""Explainable AI layer (SHAP + GNN saliency + causal event trace).

* Detection: TreeSHAP contributions per prediction -> "why was a leak
  detected on P-042?" with plain-language reasons.
* Failure risk: TreeSHAP on the risk model -> "why is this pipe high risk?"
* Localisation: gradient x input saliency -> which sensors drove the GNN.
* Causality: every simulated anomaly carries the scenario event that caused
  it, so explanations can be audited against ground truth in the twin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

FEATURE_TEXT = {
    "nbr_min": "lowest pressure residual among nearby sensors",
    "nbr_res": "pressure residual of nearby sensors",
    "nbr_contrast": "local pressure drop relative to the rest of the network",
    "contrast_rank": "how anomalous this area is compared with all others",
    "nbr_min_rank": "rank of the nearby minimum residual",
    "nbr_delta": "change of local residual in the last 1-2 h",
    "nbr_roll4": "1-hour average local residual",
    "res_start": "pressure residual at the start junction",
    "res_end": "pressure residual at the end junction",
    "res_local": "worst residual at this pipe's junctions",
    "flow_res": "flow excess over the twin's expectation",
    "flow_res_rel": "relative flow excess over expectation",
    "flow_res_rank": "rank of flow excess",
    "dma_ufw": "DMA unaccounted-for flow (water balance)",
    "dma_ufw_rel": "DMA unaccounted flow / consumption",
    "dma_ufw_roll4": "1-hour DMA unaccounted flow",
    "gap_to_min": "distance to the most anomalous area",
    "net_min_contrast": "strongest anomaly anywhere in the network",
    "net_median_res": "network-wide (common-mode) residual",
    "corrosion": "corrosion index", "pipe_health": "pipe condition score", "pipe_age": "pipe age",
    "repair_count": "number of past repairs", "historical_leaks": "number of past leaks",
    "sensor_health": "sensor health", "pressure": "observed pressure", "flow": "observed flow",
    "demand": "consumer demand", "length_m": "pipe length", "diameter_mm": "pipe diameter",
    "material_code": "pipe material", "pressure_cycles": "daily pressure cycles", "mean_pressure": "operating pressure",
    "hour_sin": "time of day", "hour_cos": "time of day", "is_night": "night period", "nbr_count": "sensor coverage",
}


class TreeExplainer:
    def __init__(self, model, features: list[str]):
        self.features = list(features)
        self.explainer = shap.TreeExplainer(model)

    def shap_values(self, X: pd.DataFrame) -> pd.DataFrame:
        vals = self.explainer.shap_values(X[self.features])
        if isinstance(vals, list):
            vals = vals[-1]
        return pd.DataFrame(vals, columns=self.features, index=X.index)

    @property
    def base_value(self) -> float:
        b = self.explainer.expected_value
        return float(np.atleast_1d(b)[-1])

    def global_importance(self, X: pd.DataFrame, n: int = 4000) -> pd.Series:
        Xs = X.sample(min(n, len(X)), random_state=0)
        return self.shap_values(Xs).abs().mean().sort_values(ascending=False)

    def explain_row(self, row: pd.DataFrame, top: int = 6) -> pd.DataFrame:
        sv = self.shap_values(row).iloc[0]
        df = pd.DataFrame({"feature": sv.index, "value": row[self.features].iloc[0].values, "shap": sv.values})
        df["abs"] = df.shap.abs()
        df["meaning"] = df.feature.map(FEATURE_TEXT).fillna(df.feature)
        return df.sort_values("abs", ascending=False).head(top).drop(columns="abs").reset_index(drop=True)


def narrate(contrib: pd.DataFrame, subject: str, verdict: str, n: int = 3) -> str:
    """Turn SHAP contributions into a sentence a control-room operator can act on."""
    up = contrib[contrib.shap > 0].head(n)
    down = contrib[contrib.shap < 0].head(1)
    parts = [f"{r.meaning} = {r.value:.3g}" if isinstance(r.value, (int, float, np.floating)) and not np.isnan(r.value)
             else f"{r.meaning} unavailable" for r in up.itertuples()]
    text = f"{subject}: {verdict}"
    if parts:
        text += " mainly because " + "; ".join(parts)
    if len(down):
        r = down.iloc[0]
        text += f". Counter-evidence: {r.meaning}"
    return text + "."


def explain_detection(detector, explainer: TreeExplainer, row: pd.DataFrame, threshold: float | None = None) -> dict:
    p = float(detector.predict_proba(row)[0])
    contrib = explainer.explain_row(row)
    thr = detector.threshold if threshold is None else threshold
    verdict = f"leak probability {p:.0%} ({'above' if p >= thr else 'below'} alarm threshold {thr:.0%})"
    return dict(probability=p, contributions=contrib, text=narrate(contrib, f"Pipe {row.pipe_id.iloc[0]}", verdict))


def explain_risk(failure_model, explainer: TreeExplainer, row: pd.DataFrame) -> dict:
    from aquatwin.models.failure import prepare
    pred = failure_model.predict(row).iloc[0]
    contrib = explainer.explain_row(prepare(row))
    verdict = f"annual failure risk {pred.failure_risk:.1%}, remaining life {pred.remaining_useful_life:.0f} y"
    return dict(prediction=pred.to_dict(), contributions=contrib,
                text=narrate(contrib, f"Pipe {row.pipe_id.iloc[0]}", verdict))
