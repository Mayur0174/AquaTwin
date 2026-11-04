"""Leak Detection Service: XGBoost classifier -> leak_probability per pipe & time.

Pipeline: feature engineering (``features.detection_features``) -> feature
selection (gain-based, cumulative 98%) -> class balancing (negative
under-sampling + ``scale_pos_weight``) -> training with episode-grouped
validation (no temporal/episode leakage) -> F1-optimal threshold ->
network-level alarm logic with persistence (false-alarm suppression).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

from aquatwin.data.schema import LEAKAGE_COLUMNS
from aquatwin.models.features import DETECTION_FEATURES

XGB_PARAMS = dict(n_estimators=400, max_depth=6, learning_rate=0.06, subsample=0.85, colsample_bytree=0.85,
                  min_child_weight=3, reg_lambda=1.0, tree_method="hist", eval_metric="aucpr", n_jobs=4)


def classification_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict[str, float]:
    yhat = (p >= thr).astype(int)
    out = dict(accuracy=accuracy_score(y, yhat), precision=precision_score(y, yhat, zero_division=0),
               recall=recall_score(y, yhat, zero_division=0), f1=f1_score(y, yhat, zero_division=0),
               threshold=float(thr), positives=int(y.sum()), n=int(len(y)),
               false_positives=int(((yhat == 1) & (y == 0)).sum()), false_negatives=int(((yhat == 0) & (y == 1)).sum()))
    if 0 < y.sum() < len(y):
        out["roc_auc"] = roc_auc_score(y, p)
        out["pr_auc"] = average_precision_score(y, p)
    return {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in out.items()}


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    i = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[i]) if len(thr) else 0.5


@dataclass
class LeakDetector:
    features: list[str] = field(default_factory=lambda: list(DETECTION_FEATURES))
    threshold: float = 0.5
    alarm_threshold: float = 0.5
    alarm_persistence: int = 2
    model: xgb.XGBClassifier | None = None
    selection_report: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ training
    def fit(self, df: pd.DataFrame, max_neg_ratio: float = 25.0, seed: int = 0, select: bool = True,
            n_folds: int = 3) -> dict:
        assert not set(self.features) & set(LEAKAGE_COLUMNS), "truth/label column used as a feature"
        rng = np.random.default_rng(seed)
        if select:
            bal = self._balance(df, max_neg_ratio, rng)
            m0 = self._new(bal, seed)
            m0.fit(bal[self.features], bal.is_leak)
            gain = pd.Series(m0.get_booster().get_score(importance_type="total_gain")).reindex(self.features).fillna(0)
            gain = gain.sort_values(ascending=False)
            cum = gain.cumsum() / gain.sum()
            keep = cum.index[: max(10, int((cum < 0.98).sum()) + 1)].tolist()
            self.selection_report = dict(ranked_gain=(gain / gain.sum()).round(4).to_dict(), kept=keep,
                                         dropped=[f for f in self.features if f not in keep])
            self.features = keep
        # out-of-fold predictions grouped by episode -> robust threshold selection
        episodes = np.array(sorted(df.episode_id.unique()))
        rng.shuffle(episodes)
        folds = np.array_split(episodes, min(n_folds, len(episodes)))
        oof = np.zeros(len(df))
        for f in folds:
            te = df.episode_id.isin(set(f)).to_numpy()
            tr_bal = self._balance(df[~te], max_neg_ratio, rng)
            m = self._new(tr_bal, seed)
            m.fit(tr_bal[self.features], tr_bal.is_leak, verbose=False)
            oof[te] = m.predict_proba(df.loc[te, self.features])[:, 1]
        y = df.is_leak.to_numpy()
        self.threshold = best_threshold(y, oof)
        self.alarm_threshold = self._tune_alarm(df, oof)
        full = self._balance(df, max_neg_ratio, rng)
        self.model = self._new(full, seed)
        self.model.fit(full[self.features], full.is_leak, verbose=False)
        return dict(validation=classification_metrics(y, oof, self.threshold),
                    validation_event=network_event_metrics(df.assign(leak_probability=oof), self.alarm_threshold,
                                                           self.alarm_persistence),
                    selection=self.selection_report, train_rows=len(full),
                    class_balance=dict(pos=int(full.is_leak.sum()), neg=int((full.is_leak == 0).sum())))

    def _new(self, d: pd.DataFrame, seed: int) -> xgb.XGBClassifier:
        spw = max(1.0, (d.is_leak == 0).sum() / max(1, d.is_leak.sum()) / 4.0)
        return xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw, random_state=seed)

    @staticmethod
    def _balance(d: pd.DataFrame, ratio: float, rng) -> pd.DataFrame:
        pos = d[d.is_leak == 1]
        neg = d[d.is_leak == 0]
        n = min(len(neg), int(max(len(pos), 50) * ratio))
        return pd.concat([pos, neg.iloc[rng.choice(len(neg), n, replace=False)]]).sample(frac=1, random_state=0)

    def _tune_alarm(self, va: pd.DataFrame, p: np.ndarray) -> float:
        best, best_f = 0.5, -1
        full_days = va.groupby("episode_id").step.transform("nunique") >= 48     # tune on operational episodes
        if full_days.any():
            va, p = va[full_days.to_numpy()], p[full_days.to_numpy()]
        for thr in np.linspace(0.2, 0.95, 16):
            m = network_event_metrics(va.assign(leak_probability=p), thr, self.alarm_persistence)
            if m["event_f1"] > best_f:
                best, best_f = float(thr), m["event_f1"]
        return best

    # ------------------------------------------------------------------ inference
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(df[self.features])[:, 1]

    def evaluate(self, df: pd.DataFrame) -> dict:
        p = self.predict_proba(df)
        out = dict(row=classification_metrics(df.is_leak.to_numpy(), p, self.threshold))
        out["event"] = network_event_metrics(df.assign(leak_probability=p), self.alarm_threshold, self.alarm_persistence)
        # pipe ranking quality during leak steps: is a true leaking pipe among the top-k scored pipes?
        d = df.assign(leak_probability=p)
        hits = {1: [], 5: [], 10: []}
        for _, g in d[d.groupby(["episode_id", "step"]).is_leak.transform("max") == 1].groupby(["episode_id", "step"]):
            order = g.sort_values("leak_probability", ascending=False).is_leak.to_numpy()
            for k in hits:
                hits[k].append(order[:k].max())
        out["pipe_hit_at_k"] = {f"hit@{k}": float(np.mean(v)) if v else float("nan") for k, v in hits.items()}
        if "leak_size" in df:
            by = {}
            size = df.leak_size.to_numpy()
            for cat, (lo, hi) in {"small(<7mm)": (0, 7), "medium(7-16)": (7, 16), "large(16-35)": (16, 35),
                                  "burst(>35)": (35, 1e9)}.items():
                sel = (df.is_leak == 1).to_numpy() & (size >= lo) & (size < hi)
                if sel.any():
                    by[cat] = float((p[sel] >= self.threshold).mean())
            out["recall_by_size"] = by
        return out

    # ------------------------------------------------------------------ io
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.model.save_model(d / "detector.json")
        meta = dict(features=self.features, threshold=self.threshold, alarm_threshold=self.alarm_threshold,
                    alarm_persistence=self.alarm_persistence, selection=self.selection_report)
        (d / "detector_meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> "LeakDetector":
        d = Path(directory)
        meta = json.loads((d / "detector_meta.json").read_text())
        m = xgb.XGBClassifier()
        m.load_model(d / "detector.json")
        return cls(features=meta["features"], threshold=meta["threshold"], alarm_threshold=meta["alarm_threshold"],
                   alarm_persistence=meta["alarm_persistence"], model=m, selection_report=meta.get("selection", {}))


def network_alarms(df: pd.DataFrame, thr: float, persistence: int, group: str = "episode_id") -> pd.DataFrame:
    """Network-level alarm per time step: max pipe probability above thr for `persistence` steps."""
    g = df.groupby([group, "step"]).agg(pmax=("leak_probability", "max"), truth=("is_leak", "max"),
                                        top_pipe=("leak_probability", lambda s: df.loc[s.idxmax(), "pipe_id"]))
    g = g.reset_index().sort_values([group, "step"])
    above = (g.pmax >= thr).astype(int)
    g["alarm"] = above.groupby(g[group]).transform(lambda s: s.rolling(persistence, min_periods=persistence).min()).fillna(0).astype(int)
    return g


def network_event_metrics(df: pd.DataFrame, thr: float, persistence: int, group: str = "episode_id") -> dict:
    g = network_alarms(df, thr, persistence, group)
    tp = int(((g.alarm == 1) & (g.truth == 1)).sum())
    fp = int(((g.alarm == 1) & (g.truth == 0)).sum())
    fn = int(((g.alarm == 0) & (g.truth == 1)).sum())
    tn = int(((g.alarm == 0) & (g.truth == 0)).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    # detection delay per leak onset (first true step -> first alarm), in steps
    delays, missed = [], 0
    for _, e in g.groupby(group):
        on = e[e.truth == 1]
        if on.empty:
            continue
        first = on.step.min()
        hit = e[(e.step >= first) & (e.alarm == 1)]
        if hit.empty:
            missed += 1
        else:
            delays.append(int(hit.step.min() - first))
    auc = float(roc_auc_score(g.truth, g.pmax)) if 0 < g.truth.sum() < len(g) else float("nan")
    return dict(event_roc_auc=auc, event_precision=prec, event_recall=rec, event_f1=2 * prec * rec / max(1e-12, prec + rec),
                event_accuracy=(tp + tn) / max(1, len(g)), false_alarm_steps=fp, missed_steps=fn,
                mean_detection_delay_steps=float(np.mean(delays)) if delays else float("nan"),
                leak_episodes_missed=missed, leak_episodes=missed + len(delays))
