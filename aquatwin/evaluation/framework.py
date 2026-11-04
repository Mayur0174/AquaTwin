"""Evaluation framework: system performance under increasing realism.

For every realism level 0-5 we simulate *fresh* episodes (new seeds, never
seen in training) and run the full AI stack exactly as in production:

detection (row + network alarm) -> localisation at alarm onsets -> DMA water
balance -> failure prediction on an asset register whose data quality
degrades with the level (inspection noise, lost repair records).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from aquatwin.config import AquaTwinConfig
from aquatwin.data.assets import build_asset_history
from aquatwin.data.generator import RowBuilder, run_episode
from aquatwin.models.detection import LeakDetector, classification_metrics, network_event_metrics
from aquatwin.models.failure import FailurePredictor
from aquatwin.models.features import detection_features
from aquatwin.pipeline import load_world
from aquatwin.scenarios.engine import REALISM_LEVELS, compose_episode
from aquatwin.services.water_loss import dma_water_balance, loss_estimation_metrics

INSPECTION_NOISE = {0: 0.0, 1: 0.03, 2: 0.04, 3: 0.07, 4: 0.09, 5: 0.12}
RECORD_LOSS = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.05, 4: 0.08, 5: 0.15}


def evaluate_levels(cfg: AquaTwinConfig, episodes_per_level: int = 3, seed: int = 2024,
                    verbose: bool = True, loc_samples: int = 60) -> pd.DataFrame:
    t0 = time.time()
    d, md = Path(cfg.dataset.output_dir), Path(cfg.training.model_dir)
    net, sensors, ctx = load_world(d)
    det = LeakDetector.load(md / "detection")
    fp = FailurePredictor.load(md / "failure")
    gnn = base = None
    try:
        from aquatwin.models.localization import GNNLocalizer, SensitivityLocalizer, build_snapshots, localization_metrics
        best = json.loads((md / "localization" / "best.json").read_text())["arch"]
        gnn = GNNLocalizer.load(md / "localization", ctx, best)
        base = SensitivityLocalizer.load(md / "localization", ctx)
    except Exception:  # pragma: no cover
        best = None
    rows = RowBuilder(net, sensors)
    rng = np.random.default_rng(seed)
    # failure prediction is scored on an UNSEEN city (different seed): no pipe overlap with training
    from dataclasses import replace
    from aquatwin.network.calibration import calibrate_network
    from aquatwin.network.generator import generate_network
    city2 = generate_network(replace(cfg.network, seed=cfg.network.seed + 101))
    calibrate_network(city2, cfg)
    results = []
    for level in range(6):
        P_all, N_all = [], []
        for k in range(episodes_per_level):
            names = compose_episode(level, rng, leak_prob=0.85)
            start = pd.Timestamp(cfg.simulation.start) + pd.Timedelta(days=int(rng.integers(100, 300)))
            res = run_episode(net, sensors, cfg, level, names, int(rng.integers(1e9)), f"V{level}{k:02d}", start, rows=rows)
            P_all.append(res.pipes)
            N_all.append(res.nodes)
        P, N = pd.concat(P_all, ignore_index=True), pd.concat(N_all, ignore_index=True)
        F = detection_features(P, N, ctx)
        p = det.predict_proba(F)
        row = classification_metrics(F.is_leak.to_numpy(), p, det.threshold)
        ev = network_event_metrics(F.assign(leak_probability=p, alarm_score=det.alarm_score(F, p)),
                                   det.alarm_threshold, det.alarm_persistence)
        # detectable-leak view: drop time steps whose only leaks are below 1 L/s (physical sensing floor)
        step_loss = F.groupby(["episode_id", "step"]).water_loss.transform("sum")
        keep = ~((step_loss > 0) & (step_loss < 1.0))
        Fd = F[keep.to_numpy()]
        evd = network_event_metrics(Fd.assign(leak_probability=p[keep.to_numpy()],
                                              alarm_score=det.alarm_score(Fd, p[keep.to_numpy()])),
                                    det.alarm_threshold, det.alarm_persistence)
        out = dict(level=level, name=REALISM_LEVELS[level].name, episodes=episodes_per_level,
                   leaks_detected=f"{ev['leak_episodes'] - ev['leak_episodes_missed']}/{ev['leak_episodes']}",
                   detectable_f1=evd["event_f1"], detectable_recall=evd["event_recall"],
                   detection_accuracy=ev["event_accuracy"], detection_precision=ev["event_precision"],
                   detection_recall=ev["event_recall"], detection_f1=ev["event_f1"],
                   detection_roc_auc=ev.get("event_roc_auc"), pipe_roc_auc=row.get("roc_auc"),
                   false_positives=ev["false_alarm_steps"], false_negatives=ev["missed_steps"],
                   detection_delay_min=ev["mean_detection_delay_steps"] * cfg.simulation.timestep_minutes)
        if gnn is not None:
            snaps = build_snapshots(N, P, ctx, "episode_id", sensitivity=base.S)
            if snaps:
                prob, pos = gnn.predict(snaps)
                lm = localization_metrics(ctx, snaps, prob, pos)
                pb, qb = base.predict(snaps)
                bm = localization_metrics(ctx, snaps, pb, qb)
                out.update(episode_loc_error_m=lm["mean_error_m"], episode_loc_snapshots=lm["n"])
            # controlled benchmark: fresh single-leak physics runs at this realism level (statistically meaningful n)
            from aquatwin.data.generator import localization_sweep
            ln, lp, _ = localization_sweep(net, sensors, cfg, loc_samples, np.random.default_rng(seed + 50 + level),
                                           levels=(level,))
            bsnaps = build_snapshots(ln, lp, ctx, "sample_id", sensitivity=base.S)
            prob, pos = gnn.predict(bsnaps)
            lm = localization_metrics(ctx, bsnaps, prob, pos)
            pb, qb = base.predict(bsnaps)
            bm = localization_metrics(ctx, bsnaps, pb, qb)
            out.update(localization_error_m=lm["mean_error_m"], localization_median_m=lm["median_error_m"],
                       localization_top1=lm["top1"], localization_top5=lm["top5"],
                       localization_within_250m=lm["within_250m"], baseline_error_m=bm["mean_error_m"],
                       baseline_median_m=bm["median_error_m"], baseline_top1=bm["top1"], localization_snapshots=lm["n"])
        bal = dma_water_balance(P, N, ctx)
        wl = loss_estimation_metrics(bal)
        out.update(water_loss_mae_ls=wl["mae_ls"], water_loss_volume_error_pct=wl["volume_error_pct"])
        # failure prediction with degraded asset data
        panel = build_asset_history(city2, cfg.dataset.history_years, np.random.default_rng(seed + level),
                                    inspection_noise=max(0.06, INSPECTION_NOISE[level] + 0.03))
        if RECORD_LOSS[level] > 0:
            lost = rng.random(len(panel)) < RECORD_LOSS[level]
            panel.loc[lost, "repair_count"] = (panel.loc[lost, "repair_count"] * 0.5).round()
        panel = panel[panel.year == panel.year.max()]
        pred = fp.predict(panel)
        from sklearn.metrics import r2_score
        out.update(failure_r2=float(r2_score(panel.failure_risk, pred.failure_risk)),
                   failure_mae=float((panel.failure_risk - pred.failure_risk).abs().mean()),
                   rul_mae_years=float((panel.remaining_useful_life - pred.remaining_useful_life).abs().mean()))
        results.append(out)
        if verbose:
            print(f"  L{level} {out['name']:28s} leaks {out['leaks_detected']}  det F1 {out['detection_f1']:.3f} "
                  f"(detectable {out['detectable_f1']:.3f})  FP {out['false_positives']:3d}  "
                  f"FN {out['false_negatives']:3d}  loc err {out.get('localization_error_m', float('nan')):6.0f} m  "
                  f"fail R2 {out['failure_r2']:.3f}")
    df = pd.DataFrame(results)
    rd = Path(cfg.training.report_dir)
    rd.mkdir(parents=True, exist_ok=True)
    df.to_csv(rd / "evaluation_levels.csv", index=False)
    (rd / "evaluation_levels.json").write_text(df.to_json(orient="records", indent=2))
    (rd / "evaluation_levels.md").write_text(render_evaluation(df, time.time() - t0))
    return df


def render_evaluation(df: pd.DataFrame, seconds: float) -> str:
    cols = [("level", "Level"), ("name", "Environment"), ("leaks_detected", "Leak episodes detected"),
            ("detection_accuracy", "Detection acc."), ("detection_f1", "Detection F1"),
            ("detectable_f1", "F1 (leaks ≥1 L/s)"), ("false_positives", "False pos. (steps)"),
            ("false_negatives", "False neg. (steps)"), ("detection_delay_min", "Delay (min)"),
            ("localization_error_m", "Loc. error (m)"), ("localization_median_m", "Loc. median (m)"),
            ("localization_top1", "Loc. top-1"),
            ("localization_within_250m", "Loc. ≤250 m"), ("baseline_error_m", "Classical loc. (m)"),
            ("baseline_median_m", "Classical median (m)"), ("failure_r2", "Failure R2"),
            ("water_loss_mae_ls", "Loss MAE (L/s)")]
    cols = [(c, h) for c, h in cols if c in df]
    lines = ["# AquaTwin AI - Evaluation under increasing realism", "",
             "Fresh, unseen episodes per level; full production pipeline.", "",
             "| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        vals = []
        for c, _ in cols:
            v = r[c]
            vals.append(f"{v:.3f}" if isinstance(v, float) and abs(v) < 10 else (f"{v:.0f}" if isinstance(v, float) else str(v)))
        lines.append("| " + " | ".join(vals) + " |")
    lines += ["", "Localisation columns come from a controlled benchmark of fresh single-leak physics runs "
              "(≈60 leaks, ~180 onset snapshots per level, leaks ≥ 1 L/s); the classical column is the "
              "sensitivity-matrix correlation method on the same snapshots. Detection columns come from full-day "
              "operational episodes. *F1 (leaks ≥1 L/s)* excludes time steps whose only leaks are below the "
              "physical sensing floor.", "", f"_Generated in {seconds:.0f} s._", "",
              "Level definitions: " + "; ".join(f"L{k}: {v.description}" for k, v in REALISM_LEVELS.items())]
    return "\n".join(lines)
