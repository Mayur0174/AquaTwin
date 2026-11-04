"""Export real twin runs + model reports into a self-contained HTML explorer (``python -m aquatwin ui``).

Every number in the explorer comes from this export: scenarios are simulated by the
truth/reality engines and decided on by the trained models, exactly like the live twin.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aquatwin.config import AquaTwinConfig
from aquatwin.scenarios.catalog import SCENARIOS
from aquatwin.scenarios.engine import REALISM_LEVELS

TEMPLATE = Path(__file__).with_name("template.html")


def _r(a, nd=2):
    return [None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), nd) for v in a]


def _run_scenario(cfg: AquaTwinConfig, spec: dict) -> dict:
    from aquatwin.twin.session import DigitalTwinSession
    tw = DigitalTwinSession(cfg, level=spec["level"], seed=spec["seed"], start=spec["start"], hours=spec["hours"])
    actions = []
    for step in range(tw.n_steps):
        for act in spec["actions"]:
            if act["at"] == step:
                if act["do"] == "leak":
                    ev = tw.inject_leak(act.get("pipe"), act.get("size", "large"))
                    actions.append(dict(step=step, text=f"{act.get('size', 'large').title()} leak starts on {ev['target']}",
                                        cause=ev["cause"]))
                    spec["_leak_pipe"] = ev["target"]
                elif act["do"] == "burst":
                    ev = tw.inject_burst(act.get("pipe"))
                    actions.append(dict(step=step, text=f"Burst on {ev['target']}", cause=ev["cause"]))
                    spec["_leak_pipe"] = ev["target"]
                elif act["do"] == "isolate":
                    tw.close_pipe(spec["_leak_pipe"])
                    actions.append(dict(step=step, text=f"Crew isolates {spec['_leak_pipe']}", cause="Operator valve closure"))
                elif act["do"] == "kill_nearest_sensor":
                    sid = tw.sensors.nearest_pressure_sensor(tw.net, spec["_leak_pipe"])
                    tw.inject_sensor_failure(sid, "sensor_failure")
                    actions.append(dict(step=step, text=f"Nearest logger {sid} fails", cause="Dead battery"))
                elif act["do"] == "demand":
                    tw.set_demand(act["factor"], act["hours"])
                    actions.append(dict(step=step, text=f"Demand surge x{act['factor']}", cause="Heatwave"))
        tw.step(1)

    net = tw.net
    frames = []
    for i, rec in enumerate(tw.records):
        t = rec.truth
        prob = tw.probabilities[i].reindex(net.pipes.index).to_numpy() if i < len(tw.probabilities) else np.zeros(len(net.pipes))
        frames.append(dict(
            t=str(rec.cond.timestamp)[5:16], p=_r(t.pressure, 1), r=_r(rec.obs_pressure - rec.expected["pressure"], 2),
            q=_r(t.flow, 1), lp=_r(prob, 3), loss=round(t.total_loss_ls, 2),
            score=round(tw.alarm_scores[i], 3) if i < len(tw.alarm_scores) else 0.0,
            leaks=[dict(pipe=l["pipe_id"], frac=round(l["frac"], 3), q=round(l["flow_ls"], 2), d=round(l["eq_diameter_mm"], 1))
                   for l in t.leaks],
            demand=round(float(t.demand_served.sum()), 1), pump=bool(t.pump_on),
            tanks=_r(t.tank_levels, 2), temp=round(t.temperature_c, 1)))
    bal = tw.water_balance()
    est = bal.groupby("step").estimated_loss_ls.sum().reindex(range(len(frames))).fillna(0).to_numpy() if not bal.empty else []
    for f, e in zip(frames, est):
        f["est"] = round(float(e), 2)
    decisions = []
    step_of = {str(r.cond.timestamp): i for i, r in enumerate(tw.records)}
    for d in tw.decisions:
        item = dict(step=step_of.get(d.timestamp, 0), kind=d.kind, text=d.text)
        if d.kind == "alarm":
            item["pipe"] = d.payload["pipe_id"]
            item["shap"] = [dict(f=c["meaning"], v=round(float(c["shap"]), 3)) for c in d.payload["contributions"]]
        if d.kind == "localisation":
            item["cands"] = [dict(pipe=c["pipe_id"], p=round(c["probability"], 3), frac=round(c["position_frac"], 3),
                                  dist=round(c["distance_from_start_m"], 0), start=c["start_node"])
                             for c in d.payload["candidates"]]
            truth = tw.records[item["step"]].truth.leaks
            if truth:
                c0 = item["cands"][0]
                item["error_m"] = round(min(net.point_distance(c0["pipe"], c0["frac"], l["pipe_id"], l["frac"]) for l in truth))
        decisions.append(item)
    return dict(id=spec["id"], title=spec["title"], level=spec["level"], level_name=REALISM_LEVELS[spec["level"]].name,
                story=spec["story"], actions=actions, frames=frames, decisions=decisions,
                threshold=float(tw.detector.alarm_threshold) if tw.detector else None)


SCENARIO_SPECS = [
    dict(id="evening-leak", title="Large leak at evening peak", level=3, seed=11, start="2026-06-10 16:00", hours=9,
         story="A corroded main opens during the evening demand peak, under realistic sensor drift and noise.",
         actions=[dict(at=6, do="leak", size="large")]),
    dict(id="burst-isolation", title="Burst, then valve isolation", level=4, seed=5, start="2026-06-12 05:00", hours=9,
         story="A burst during the morning ramp-up; the crew isolates the pipe two hours later.",
         actions=[dict(at=6, do="burst"), dict(at=14, do="isolate")]),
    dict(id="leak-dead-logger", title="Leak with the nearest logger dead", level=4, seed=21, start="2026-06-15 21:00", hours=9,
         story="A large night-time leak while the closest pressure logger has stopped reporting.",
         actions=[dict(at=6, do="leak", size="large"), dict(at=6, do="kill_nearest_sensor")]),
    dict(id="heatwave-leak", title="Leak hidden by a heatwave surge", level=5, seed=8, start="2026-05-20 10:00", hours=9,
         story="Demand jumps 30% across the city while a medium leak starts; the common-mode drop masks it.",
         actions=[dict(at=4, do="demand", factor=1.3, hours=8), dict(at=7, do="leak", size="medium")]),
]


def export_ui_data(cfg: AquaTwinConfig, specs: list[dict] | None = None, verbose: bool = True) -> dict:
    from aquatwin.twin.session import DigitalTwinSession
    rd = Path(cfg.training.report_dir)
    base = DigitalTwinSession(cfg, level=0, seed=0, hours=1)
    net = base.net
    risk = base.asset_risk().set_index("pipe_id")
    n, p = net.nodes, net.pipes
    monitored = set(base.sensors.monitored_nodes)
    metered = set(base.sensors.metered_pipes)
    nodes = [dict(id=i, type=r.node_type, x=round(r.x, 1), y=round(r.y, 1), z=round(r.elevation, 1), dma=r.dma,
                  lu=r.land_use, mon=i in monitored) for i, r in n.iterrows()]
    pipes = [dict(id=i, s=r.start_node, e=r.end_node, mat=r.material, d=int(r.diameter_mm), L=round(r.length_m),
                  age=int(r.age_years), yr=int(r.installation_year), c=round(r.corrosion_index, 2), rep=int(r.repair_count),
                  h=round(r.health_score), risk=round(float(risk.at[i, "pred_failure_risk"]), 4),
                  rul=round(float(risk.at[i, "pred_remaining_useful_life"]), 1), dma=r.dma, valve=bool(r.has_valve),
                  meter=i in metered, brk=round(r.break_rate, 3)) for i, r in p.iterrows()]
    scen = []
    for spec in specs or SCENARIO_SPECS:
        if verbose:
            print(f"  simulating '{spec['title']}' (L{spec['level']}) ...")
        scen.append(_run_scenario(cfg, dict(spec)))
    training = json.loads((rd / "training_report.json").read_text()) if (rd / "training_report.json").exists() else {}
    levels = json.loads((rd / "evaluation_levels.json").read_text()) if (rd / "evaluation_levels.json").exists() else []
    manifest_path = Path(cfg.dataset.output_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    det = training.get("detection", {}).get("test", {})
    loc = training.get("localization", {})
    fail = training.get("failure", {})
    summary = dict(
        detection=dict(event=det.get("event", {}), row=det.get("row", {}), hit=det.get("pipe_hit_at_k", {})),
        localization={k: v.get("validation") for k, v in loc.items() if isinstance(v, dict) and "validation" in v},
        best_arch=loc.get("best_architecture"), failure=fail, water_loss=training.get("water_loss", {}),
        shap_detection=training.get("explainability", {}).get("detection_global_shap", {}),
        shap_failure=training.get("explainability", {}).get("failure_global_shap", {}),
        dataset=dict(records=manifest.get("records_pipes"), packets=manifest.get("telemetry_packets"),
                     sweep=manifest.get("localization_samples"), mbe=manifest.get("max_mass_balance_error")))
    catalog = [dict(name=s.name, cat=s.category, title=s.title, rules=list(s.rules)) for s in SCENARIOS.values()]
    return dict(network=dict(nodes=nodes, pipes=pipes, width=net.graph.graph.get("width_m"),
                             height=net.graph.graph.get("height_m")),
                scenarios=scen, summary=summary, levels=levels, catalog=catalog)


def build_ui(cfg: AquaTwinConfig, out: str | Path = "ui/aquatwin_explorer.html", verbose: bool = True) -> Path:
    data = export_ui_data(cfg, verbose=verbose)
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__AQUATWIN_DATA__*/null",
                                                       json.dumps(data, separators=(",", ":"), default=float))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    if verbose:
        print(f"  wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    return out
