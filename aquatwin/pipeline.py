"""End-to-end training pipeline: dataset -> features -> models -> reports -> registry."""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from aquatwin.config import AquaTwinConfig
from aquatwin.mlops.registry import ModelRegistry
from aquatwin.models.detection import LeakDetector
from aquatwin.models.failure import FailurePredictor, current_asset_frame
from aquatwin.models.features import NetworkContext, detection_features
from aquatwin.network.model import WaterNetwork
from aquatwin.sensors.registry import SensorRegistry
from aquatwin.services.water_loss import dma_water_balance, loss_estimation_metrics

warnings.filterwarnings("ignore", category=UserWarning)


def load_world(data_dir: str | Path) -> tuple[WaterNetwork, SensorRegistry, NetworkContext]:
    d = Path(data_dir)
    net = WaterNetwork.load(d / "network")
    sensors = SensorRegistry(pd.read_csv(d / "sensors.csv").set_index("sensor_id", drop=False))
    return net, sensors, NetworkContext.build(net, sensors)


def split_episodes(episodes: list[str], test_fraction: float) -> tuple[list[str], list[str]]:
    """Levels cycle 0..5 over episodes, so the last block is a level-stratified test set."""
    eps = sorted(episodes)
    n_test = max(1, int(round(test_fraction * len(eps))))
    return eps[:-n_test], eps[-n_test:]


def train_all(cfg: AquaTwinConfig, verbose: bool = True) -> dict:
    t0 = time.time()
    d = Path(cfg.dataset.output_dir)
    md, rd = Path(cfg.training.model_dir), Path(cfg.training.report_dir)
    md.mkdir(parents=True, exist_ok=True)
    rd.mkdir(parents=True, exist_ok=True)
    log = print if verbose else (lambda *a, **k: None)
    net, sensors, ctx = load_world(d)
    registry = ModelRegistry(md)
    report: dict = {}

    # ------------------------------------------------------------------ detection
    log("[1/4] leak detection (XGBoost)")
    P = pd.read_parquet(d / "pipes_timeseries.parquet")
    N = pd.read_parquet(d / "nodes_timeseries.parquet")
    train_eps, test_eps = split_episodes(P.episode_id.unique().tolist(), cfg.training.test_fraction)
    F = detection_features(P, N, ctx)
    SP = pd.read_parquet(d / "localization_pipes.parquet")
    SN = pd.read_parquet(d / "localization_nodes.parquet")
    SP["episode_id"], SN["episode_id"] = SP.sample_id, SN.sample_id
    FS = detection_features(SP, SN, ctx)
    train_df = pd.concat([F[F.episode_id.isin(train_eps)], FS], ignore_index=True)
    test_df = F[F.episode_id.isin(test_eps)]
    det = LeakDetector()
    fit_rep = det.fit(train_df, cfg.training.detection_max_neg_ratio, cfg.training.seed)
    det_test = det.evaluate(test_df)
    det.save(md / "detection")
    report["detection"] = dict(validation_oof=fit_rep["validation"], validation_event=fit_rep["validation_event"],
                               test=det_test, selected_features=det.features,
                               feature_selection=fit_rep["selection"], class_balance=fit_rep["class_balance"],
                               train_episodes=train_eps, test_episodes=test_eps)
    registry.register("leak_detector", det_test, d / "manifest.json", dict(features=det.features))
    log(f"      test ROC-AUC {det_test['row'].get('roc_auc', float('nan')):.3f} | event F1 {det_test['event']['event_f1']:.3f}")

    # ------------------------------------------------------------------ failure prediction
    log("[2/4] failure prediction (XGBoost regressors)")
    panel = pd.read_parquet(d / "asset_history.parquet")
    fp = FailurePredictor()
    fail_rep = fp.fit(panel, seed=cfg.training.seed)
    fp.save(md / "failure")
    current = current_asset_frame(net)
    pred = fp.predict(current)
    pd.concat([current, pred.add_prefix("pred_")], axis=1).to_csv(rd / "asset_risk_ranking.csv", index=False)
    report["failure"] = fail_rep
    registry.register("failure_predictor", fail_rep, d / "manifest.json")
    log(f"      risk R2 {fail_rep['failure_risk']['r2']:.3f} | RUL R2 {fail_rep['remaining_useful_life']['r2']:.3f}")

    # ------------------------------------------------------------------ localisation
    report["localization"] = {}
    try:
        from aquatwin.models.localization import GNNLocalizer, SensitivityLocalizer, build_snapshots, localization_metrics
        log("[3/4] leak localisation (GNN: GCN / GraphSAGE / GAT vs sensitivity baseline)")
        base = SensitivityLocalizer(ctx)
        base.save(md / "localization")
        S = build_snapshots(SN, SP, ctx, "sample_id", sensitivity=base.S)
        E = build_snapshots(N[N.episode_id.isin(test_eps)], P[P.episode_id.isin(test_eps)], ctx, "episode_id",
                            sensitivity=base.S)
        rng = np.random.default_rng(cfg.training.seed)
        ids = sorted({s.group for s in S})
        rng.shuffle(ids)
        val_ids = set(ids[: max(1, len(ids) // 5)])
        tr = [s for s in S if s.group not in val_ids]
        va = [s for s in S if s.group in val_ids]
        pb, qb = base.predict(va)
        loc = {"sensitivity_baseline": dict(validation=localization_metrics(ctx, va, pb, qb))}
        if E:
            pb, qb = base.predict(E)
            loc["sensitivity_baseline"]["test"] = localization_metrics(ctx, E, pb, qb)
        best, best_err = None, np.inf
        for arch in cfg.training.gnn_architectures:
            g = GNNLocalizer(ctx, arch, cfg.training.gnn_hidden, cfg.training.gnn_layers, cfg.training.seed)
            ts = time.time()
            g.fit(tr, va, cfg.training.gnn_epochs, cfg.training.gnn_lr, verbose=verbose)
            entry = dict(validation=g.evaluate(va), train_seconds=round(time.time() - ts, 1),
                         params=int(sum(p.numel() for p in g.model.parameters())))
            if E:
                entry["test"] = g.evaluate(E)
            loc[arch] = entry
            g.save(md / "localization")
            log(f"      {arch:5s} val err {entry['validation']['mean_error_m']:.0f} m | "
                f"test err {entry.get('test', {}).get('mean_error_m', float('nan')):.0f} m")
            if entry["validation"]["mean_error_m"] < best_err:
                best, best_err = arch, entry["validation"]["mean_error_m"]
        loc["best_architecture"] = best
        loc["n_train_snapshots"], loc["n_val_snapshots"], loc["n_test_snapshots"] = len(tr), len(va), len(E)
        (md / "localization" / "best.json").write_text(json.dumps({"arch": best}))
        report["localization"] = loc
        registry.register("leak_localizer", loc[best], d / "manifest.json", dict(architecture=best))
    except ImportError as exc:  # pragma: no cover
        log(f"      skipped GNN ({exc})")

    # ------------------------------------------------------------------ water loss + XAI
    log("[4/4] water-loss estimation & explainability")
    bal = dma_water_balance(P[P.episode_id.isin(test_eps)], N[N.episode_id.isin(test_eps)], ctx)
    report["water_loss"] = loss_estimation_metrics(bal)
    from aquatwin.xai.explain import TreeExplainer
    from aquatwin.models.failure import prepare
    det_imp = TreeExplainer(det.model, det.features).global_importance(test_df)
    fail_imp = TreeExplainer(fp.risk_model, fp.features).global_importance(prepare(panel))
    report["explainability"] = dict(detection_global_shap=det_imp.round(4).to_dict(),
                                    failure_global_shap=fail_imp.round(4).to_dict())
    report["seconds"] = round(time.time() - t0, 1)
    (rd / "training_report.json").write_text(json.dumps(report, indent=2, default=float))
    (rd / "training_report.md").write_text(render_training_report(report))
    log(f"done in {report['seconds']} s -> {rd / 'training_report.md'}")
    return report


def _fmt(v, nd=3):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def render_training_report(r: dict) -> str:
    det = r["detection"]["test"]
    row, ev = det["row"], det["event"]
    lines = ["# AquaTwin AI - Training Report", "",
             "Held-out test set: unseen episodes covering realism levels 0-5.", "",
             "## Leak detection (XGBoost)", "",
             "| Granularity | Accuracy | Precision | Recall | F1 | ROC-AUC |", "|---|---|---|---|---|---|",
             f"| Network alarm (per time step) | {_fmt(ev['event_accuracy'])} | {_fmt(ev['event_precision'])} | "
             f"{_fmt(ev['event_recall'])} | {_fmt(ev['event_f1'])} | {_fmt(ev.get('event_roc_auc'))} |",
             f"| Pipe x time step | {_fmt(row['accuracy'])} | {_fmt(row['precision'])} | {_fmt(row['recall'])} | "
             f"{_fmt(row['f1'])} | {_fmt(row.get('roc_auc'))} |", "",
             f"Mean detection delay: {_fmt(ev['mean_detection_delay_steps'], 2)} steps (15 min each); "
             f"leak episodes missed: {ev['leak_episodes_missed']}/{ev['leak_episodes']}.", "",
             "Pipe ranking during leaks: " + ", ".join(f"{k} = {_fmt(v)}" for k, v in det["pipe_hit_at_k"].items()), "",
             "Recall by leak size (pipe level): " + ", ".join(f"{k}: {_fmt(v)}" for k, v in det.get("recall_by_size", {}).items()), "",
             f"Selected features ({len(r['detection']['selected_features'])}): `" + "`, `".join(r["detection"]["selected_features"]) + "`", ""]
    loc = r.get("localization", {})
    if loc:
        lines += ["## Leak localisation (GNN)", "",
                  "| Model | Split | Top-1 | Top-5 | Mean error (m) | Median error (m) | <=100 m | <=250 m |",
                  "|---|---|---|---|---|---|---|---|"]
        for name in ["sensitivity_baseline"] + [a for a in ("gcn", "sage", "gat") if a in loc]:
            for split in ("validation", "test"):
                m = loc[name].get(split)
                if m:
                    lines.append(f"| {name} | {split} | {_fmt(m['top1'])} | {_fmt(m['top5'])} | {_fmt(m['mean_error_m'], 0)} | "
                                 f"{_fmt(m['median_error_m'], 0)} | {_fmt(m['within_100m'])} | {_fmt(m['within_250m'])} |")
        lines += ["", f"Best architecture (validation): **{loc.get('best_architecture')}**", ""]
    f = r["failure"]
    lines += ["## Failure prediction (XGBoost regressors, GroupKFold by pipe)", "",
              "| Target | MAE | RMSE | R2 |", "|---|---|---|---|",
              f"| failure_risk (annual prob.) | {_fmt(f['failure_risk']['mae'], 4)} | {_fmt(f['failure_risk']['rmse'], 4)} | {_fmt(f['failure_risk']['r2'])} |",
              f"| remaining_useful_life (y) | {_fmt(f['remaining_useful_life']['mae'], 2)} | {_fmt(f['remaining_useful_life']['rmse'], 2)} | {_fmt(f['remaining_useful_life']['r2'])} |",
              "", f"Next-year break AUC: {_fmt(f.get('break_auc'))}; breaks captured by top-10% risk list: {_fmt(f.get('breaks_captured_top10pct'))}", "",
              "## Water-loss estimation (DMA balance, test episodes)", "",
              ", ".join(f"{k} = {_fmt(v)}" for k, v in r["water_loss"].items()), "",
              "## Global explanations (mean |SHAP|)", "", "Detection: " +
              ", ".join(f"{k} ({v:.3f})" for k, v in list(r["explainability"]["detection_global_shap"].items())[:8]), "",
              "Failure risk: " + ", ".join(f"{k} ({v:.3f})" for k, v in list(r["explainability"]["failure_global_shap"].items())[:8]), ""]
    return "\n".join(lines)
