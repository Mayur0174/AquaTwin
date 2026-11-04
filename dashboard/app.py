"""AquaTwin AI - GIS Dashboard & Digital Twin Control Panel (Streamlit + Folium).

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aquatwin.config import AquaTwinConfig  # noqa: E402
from aquatwin.scenarios.catalog import SCENARIOS, scenarios_by_category  # noqa: E402
from aquatwin.scenarios.engine import REALISM_LEVELS  # noqa: E402

# ---------------------------------------------------------------- palette (dataviz reference instance)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_ORANGE = ["#fde3d6", "#f9bfa2", "#f39a70", "#eb6834", "#c94f20", "#9c3a14", "#6e280c"]
DIVERGE = ("#2a78d6", "#e34948")
GRID, AXIS, MUTED = "#e1e0d9", "#c3c2b7", "#898781"

st.set_page_config(page_title="AquaTwin AI", page_icon="💧", layout="wide")


def style(fig: go.Figure, title: str = "", height: int = 320, y_title: str = "") -> go.Figure:
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=14)), height=height, margin=dict(l=8, r=8, t=40, b=8),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=MUTED),
                      legend=dict(orientation="h", y=-0.18), hovermode="x unified")
    fig.update_xaxes(showgrid=False, linecolor=AXIS)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title=y_title)
    return fig


def ramp(values: np.ndarray, colors: list[str], lo: float | None = None, hi: float | None = None) -> list[str]:
    v = np.asarray(values, float)
    lo = np.nanmin(v) if lo is None else lo
    hi = np.nanmax(v) if hi is None else hi
    t = np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)
    idx = np.nan_to_num(t * (len(colors) - 1)).round().astype(int)
    return [colors[i] for i in idx]


def health_status(h: float) -> str:
    return "good" if h >= 70 else "warning" if h >= 50 else "serious" if h >= 35 else "critical"


def risk_status(r: float) -> str:
    return "good" if r < 0.02 else "warning" if r < 0.05 else "serious" if r < 0.15 else "critical"


# ---------------------------------------------------------------- session bootstrap
cfg = AquaTwinConfig.load(st.session_state.get("config_path"))
data_dir, model_dir, report_dir = Path(cfg.dataset.output_dir), Path(cfg.training.model_dir), Path(cfg.training.report_dir)
if not (data_dir / "network" / "network.json").exists():
    st.title("💧 AquaTwin AI")
    st.error("No generated dataset found. Run `python -m aquatwin all` (or `make all`) first, then reload.")
    st.stop()


def new_session(level: int, seed: int, start: str, hours: int):
    from aquatwin.twin.session import DigitalTwinSession
    return DigitalTwinSession(cfg, level=level, seed=seed, start=start, hours=hours)


if "twin" not in st.session_state:
    with st.spinner("Booting digital twin (network, sensors, AI models)..."):
        st.session_state.twin = new_session(3, 0, cfg.simulation.start, 48)
        st.session_state.twin.step(4)
twin = st.session_state.twin
net = twin.net
pipes, nodes = net.pipes, net.nodes

# ---------------------------------------------------------------- sidebar: control panel
with st.sidebar:
    st.header("🎛️ Twin Control Panel")
    with st.expander("Simulation", expanded=True):
        level = st.selectbox("Realism level", list(REALISM_LEVELS), index=3,
                             format_func=lambda k: f"L{k} - {REALISM_LEVELS[k].name}")
        c1, c2 = st.columns(2)
        seed = c1.number_input("Seed", 0, 9999, 0)
        hours = c2.number_input("Horizon (h)", 6, 336, 48)
        start = st.text_input("Start", cfg.simulation.start)
        if st.button("▶ Start new simulation", use_container_width=True):
            st.session_state.twin = new_session(level, int(seed), start, int(hours))
            st.session_state.twin.step(4)
            st.rerun()
        c1, c2, c3 = st.columns(3)
        for col, n, lbl in ((c1, 1, "+15 min"), (c2, 4, "+1 h"), (c3, 24, "+6 h")):
            if col.button(lbl, use_container_width=True):
                with st.spinner("Simulating physics + AI..."):
                    twin.step(n)
                st.rerun()
    with st.expander("Inject events"):
        pipe_choice = st.selectbox("Pipe", ["(physics hazard field)"] + pipes.index.tolist())
        pid = None if pipe_choice.startswith("(") else pipe_choice
        cat = st.select_slider("Leak size", ["small", "medium", "large", "burst"], "large")
        pos = st.slider("Position along pipe", 0.0, 1.0, 0.5)
        c1, c2 = st.columns(2)
        if c1.button("💧 Inject leak", use_container_width=True):
            ev = twin.inject_leak(pid, cat, pos)
            st.toast(f"Leak injected on {ev['target']}")
        if c2.button("💥 Inject burst", use_container_width=True):
            ev = twin.inject_burst(pid)
            st.toast(f"Burst injected on {ev['target']}")
        sensor = st.selectbox("Sensor", twin.sensors.pressure.sensor_id.tolist())
        mode = st.selectbox("Sensor fault", ["sensor_failure", "sensor_drift", "sensor_bias", "sensor_frozen"])
        if st.button("📡 Inject sensor fault", use_container_width=True):
            twin.inject_sensor_failure(sensor, mode)
            st.toast(f"{mode} on {sensor}")
        if pid and st.button(f"🔒 Isolate {pid} (close valves)", use_container_width=True):
            twin.close_pipe(pid)
    with st.expander("Demand & weather"):
        factor = st.slider("Demand factor", 0.5, 2.0, 1.3, 0.05)
        land = st.selectbox("Applies to", ["all", "residential", "commercial", "industrial"])
        dh = st.slider("Duration (h)", 1, 24, 6)
        if st.button("Apply demand change", use_container_width=True):
            twin.set_demand(factor, dh, None if land == "all" else land)
        temp = st.slider("Temperature offset (°C)", -8, 10, 0)
        rain = st.slider("Rainfall (mm/h)", 0, 60, 0)
        if st.button("Apply weather", use_container_width=True):
            twin.set_weather(temp, rain, dh)
    with st.expander("Replay historical / catalogue scenario"):
        scen = st.selectbox("Scenario", list(SCENARIOS), format_func=lambda s: SCENARIOS[s].title)
        st.caption(" · ".join(SCENARIOS[scen].rules))
        if st.button("Load scenario from now", use_container_width=True):
            twin.load_scenario(scen)
            st.toast(f"Scenario '{SCENARIOS[scen].title}' scheduled")
    show_truth = st.toggle("Show ground truth layer", value=True,
                           help="Only exists in simulation - a real deployment has no truth layer")

# ---------------------------------------------------------------- header KPIs
s = twin.state()
st.title("💧 AquaTwin AI — Water Network Digital Twin")
k = st.columns(6)
k[0].metric("Simulated time", s.get("timestamp", "-")[5:16])
k[1].metric("Demand", f"{s.get('total_demand_ls', 0):.0f} L/s")
k[2].metric("AI status", "🚨 LEAK ALARM" if s.get("alarm") else "✅ Normal")
bal = twin.water_balance()
est_loss = float(bal[bal.step == bal.step.max()].estimated_loss_ls.sum()) if not bal.empty else 0.0
k[3].metric("Estimated loss (DMA balance)", f"{est_loss:.1f} L/s")
k[4].metric("True loss", f"{s.get('true_loss_ls', 0):.1f} L/s" if show_truth else "hidden")
k[5].metric("Pump / Tanks", ("ON" if s.get("pump_on") else "OFF") + " / " +
            ", ".join(f"{v:.1f} m" for v in s.get("tank_levels", {}).values()))

tabs = st.tabs(["🗺️ Network Map", "⏯️ Timeline Playback", "🧠 AI Decisions & XAI", "🏗️ Asset Risk",
                "🚰 Water Loss", "🧪 Evaluation", "📚 Scenarios"])

# ---------------------------------------------------------------- map
def build_map(step_pos: int, layer: str, mat_filter, dma_filter, min_risk: float, age_rng) -> folium.Map:
    rec = twin.records[step_pos] if twin.records else None
    risk = twin.asset_risk().set_index("pipe_id")
    prob = twin.probabilities[step_pos] if twin.probabilities and step_pos < len(twin.probabilities) else None
    center = [nodes.lat.mean(), nodes.lon.mean()]
    m = folium.Map(location=center, zoom_start=15, tiles="cartodbpositron", control_scale=True)
    sel = pipes[pipes.material.isin(mat_filter) & pipes.dma.isin(dma_filter) &
                (risk.loc[pipes.index, "pred_failure_risk"] >= min_risk) &
                pipes.age_years.between(*age_rng)]
    nidx = net.node_index
    if rec is not None:
        p_node = rec.truth.pressure if show_truth else np.nan_to_num(rec.obs_pressure, nan=np.nan)
        pp = np.array([np.nanmean([p_node[nidx[a]], p_node[nidx[b]]]) for a, b in zip(pipes.start_node, pipes.end_node)])
        pressure = pd.Series(pp, index=pipes.index)
        flow = pd.Series(np.abs(rec.truth.flow), index=pipes.index)
    if layer == "Pipe health":
        colors = {pid: STATUS[health_status(h)] for pid, h in pipes.health_score.items()}
    elif layer == "Failure risk (AI)":
        colors = {pid: STATUS[risk_status(r)] for pid, r in risk.pred_failure_risk.items()}
    elif layer == "Pressure" and rec is not None:
        colors = dict(zip(pipes.index, ramp(pressure.values, SEQ_BLUE, 15, 60)))
    elif layer == "Flow" and rec is not None:
        colors = dict(zip(pipes.index, ramp(np.log1p(flow.values), SEQ_BLUE)))
    elif layer == "Leak probability (AI)" and prob is not None:
        colors = dict(zip(prob.index, ramp(prob.values, SEQ_ORANGE, 0, 1)))
    else:
        colors = {pid: "#86b6ef" for pid in pipes.index}
    for pid, r in sel.iterrows():
        a, b = nodes.loc[r.start_node], nodes.loc[r.end_node]
        rk = risk.loc[pid]
        popup = (f"<b>{pid}</b> ({r.material}, {r.diameter_mm:.0f} mm)<br>{r.start_node} → {r.end_node}, {r.length_m:.0f} m<br>"
                 f"Installed {r.installation_year} (age {r.age_years:.0f} y), repairs {r.repair_count}<br>"
                 f"Corrosion {r.corrosion_index:.2f} · Health {r.health_score:.0f}/100 ({health_status(r.health_score)})<br>"
                 f"AI failure risk {rk.pred_failure_risk:.1%}/yr · RUL {rk.pred_remaining_useful_life:.0f} y<br>"
                 f"DMA {r.dma}{' · boundary meter' if r.is_dma_boundary else ''}{' · valve ' + r.valve_id if r.has_valve else ''}")
        if rec is not None:
            popup += f"<br>Pressure ~{pressure[pid]:.1f} m · flow {rec.truth.flow[net.pipe_index[pid]]:.1f} L/s"
        if prob is not None:
            popup += f"<br>Leak probability {prob.get(pid, 0):.1%}"
        w = 2 + 4 * (r.diameter_mm >= 300) + 2 * (r.diameter_mm >= 500)
        folium.PolyLine([[a.lat, a.lon], [b.lat, b.lon]], color=colors.get(pid, "#86b6ef"), weight=w, opacity=0.9,
                        tooltip=pid, popup=folium.Popup(popup, max_width=320)).add_to(m)
    for nid, n in nodes[nodes.node_type != "junction"].iterrows():
        icon = {"reservoir": "tint", "pump_station": "cog", "tank": "database"}[n.node_type]
        folium.Marker([n.lat, n.lon], tooltip=f"{nid} ({n.node_type})",
                      icon=folium.Icon(color="blue", icon=icon, prefix="fa")).add_to(m)
    for nid in twin.sensors.monitored_nodes:
        n = nodes.loc[nid]
        folium.CircleMarker([n.lat, n.lon], radius=3, color="#52514e", fill=True, fill_opacity=0.8,
                            tooltip=f"pressure logger {nid}").add_to(m)
    # predicted failure zones: top-10 AI risk
    for pid in risk.pred_failure_risk.sort_values(ascending=False).index[:10]:
        lat, lon = net.point_on_pipe(pid, 0.5)
        folium.Circle([lat, lon], radius=110, color=STATUS["critical"], fill=True, fill_opacity=0.12, weight=1,
                      tooltip=f"Predicted failure zone: {pid} ({risk.at[pid, 'pred_failure_risk']:.1%}/yr)").add_to(m)
    if twin.localization is not None:
        for i, c in twin.localization.head(3).iterrows():
            folium.Marker([c.lat, c.lon], tooltip=f"AI leak candidate #{i + 1}: {c.pipe_id} @ {c.distance_from_start_m:.0f} m "
                                                  f"from {c.start_node} (p={c.probability:.0%})",
                          icon=folium.Icon(color="orange", icon="crosshairs", prefix="fa")).add_to(m)
    if show_truth and rec is not None:
        for lk in rec.truth.leaks:
            lat, lon = net.point_on_pipe(lk["pipe_id"], lk["frac"])
            folium.Marker([lat, lon], tooltip=f"TRUE leak {lk['pipe_id']} {lk['eq_diameter_mm']:.0f} mm, "
                                              f"{lk['flow_ls']:.1f} L/s @ {lk['distance_m']:.0f} m",
                          icon=folium.Icon(color="red", icon="tint", prefix="fa")).add_to(m)
    return m


with tabs[0]:
    c = st.columns([2, 2, 2, 2, 2])
    layer = c[0].selectbox("Colour pipes by", ["Pipe health", "Failure risk (AI)", "Pressure", "Flow", "Leak probability (AI)"])
    mats = c[1].multiselect("Material", sorted(pipes.material.unique()), sorted(pipes.material.unique()))
    dmas = c[2].multiselect("DMA", sorted(pipes.dma.dropna().unique()), sorted(pipes.dma.dropna().unique()))
    min_risk = c[3].slider("Min AI risk", 0.0, 0.3, 0.0, 0.01)
    age = c[4].slider("Age (y)", 0, int(pipes.age_years.max()) + 1, (0, int(pipes.age_years.max()) + 1))
    pos = len(twin.records) - 1
    st_folium(build_map(pos, layer, mats, dmas, min_risk, age), height=560, use_container_width=True, returned_objects=[])
    if layer in ("Pipe health", "Failure risk (AI)"):
        lab = {"Pipe health": ("≥70", "50–70", "35–50", "<35"), "Failure risk (AI)": ("<2%/y", "2–5%", "5–15%", ">15%")}[layer]
        st.markdown(" ".join(f"<span style='color:{STATUS[s_]}'>━━</span> {s_} {l}" for s_, l in zip(STATUS, lab)),
                    unsafe_allow_html=True)
    st.caption("🔴 true leak (simulation only) · 🟠 AI leak candidates · ○ predicted failure zones · • pressure loggers · "
               "click a pipe for asset details")

# ---------------------------------------------------------------- timeline playback
with tabs[1]:
    if len(twin.records) < 2:
        st.info("Advance the simulation to build a timeline.")
    else:
        t_sel = st.slider("Playback step", 0, len(twin.records) - 1, len(twin.records) - 1)
        rec = twin.records[t_sel]
        st.caption(f"{rec.cond.timestamp} · active events: {', '.join(rec.cond.active_events) or 'none'}")
        cA, cB = st.columns([3, 2])
        with cA:
            st_folium(build_map(t_sel, "Pressure", pipes.material.unique(), pipes.dma.dropna().unique(), 0.0, (0, 999)),
                      height=420, use_container_width=True, returned_objects=[], key="playback_map")
        with cB:
            node = st.selectbox("Pressure logger", twin.sensors.monitored_nodes)
            ND = twin.nodes_df()
            nd = ND[ND.node_id == node]
            fig = go.Figure()
            fig.add_scatter(x=nd.timestamp, y=nd.observed_pressure, name="observed", line=dict(color=SERIES[0], width=2))
            fig.add_scatter(x=nd.timestamp, y=nd.expected_pressure, name="twin expected", line=dict(color=SERIES[1], width=2, dash="dot"))
            if show_truth:
                fig.add_scatter(x=nd.timestamp, y=nd.true_pressure, name="truth", line=dict(color=SERIES[2], width=2))
            st.plotly_chart(style(fig, f"Pressure at {node}", 260, "m"), use_container_width=True, theme=None)
            sysdf = pd.DataFrame([dict(t=r.cond.timestamp, loss=r.truth.total_loss_ls) for r in twin.records])
            pm = pd.DataFrame(dict(t=[r.cond.timestamp for r in twin.records[-len(twin.probabilities):]],
                                   p=[p.max() for p in twin.probabilities]))
            fig = go.Figure()
            fig.add_scatter(x=pm.t, y=pm.p, name="max leak probability", line=dict(color=SERIES[0], width=2))
            if twin.detector is not None:
                fig.add_hline(y=twin.detector.alarm_threshold, line=dict(color=MUTED, dash="dash", width=1),
                              annotation_text="alarm threshold")
            st.plotly_chart(style(fig, "AI leak probability (network max)", 220), use_container_width=True, theme=None)
            if show_truth:
                fig = go.Figure(go.Scatter(x=sysdf.t, y=sysdf.loss, name="true loss", line=dict(color=SERIES[0], width=2),
                                           fill="tozeroy", fillcolor="rgba(42,120,214,0.12)"))
                st.plotly_chart(style(fig, "True water loss", 200, "L/s"), use_container_width=True, theme=None)

# ---------------------------------------------------------------- AI decisions & XAI
with tabs[2]:
    cA, cB = st.columns([3, 2])
    with cA:
        st.subheader("AI decision log")
        if not twin.decisions:
            st.info("No AI decisions yet. Inject a leak and advance time.")
        for d in reversed(twin.decisions[-15:]):
            icon = {"alarm": "🚨", "localisation": "📍", "clear": "✅"}[d.kind]
            st.markdown(f"**{icon} {d.timestamp[5:16]} — {d.kind.upper()}**  \n{d.text}")
        if twin.localization is not None:
            st.subheader("GNN localisation candidates")
            st.dataframe(twin.localization.style.format({"probability": "{:.1%}", "position_frac": "{:.2f}",
                                                         "distance_from_start_m": "{:.0f}", "lat": "{:.5f}", "lon": "{:.5f}"}),
                         use_container_width=True, hide_index=True)
    with cB:
        st.subheader("Why? (SHAP)")
        alarms = [d for d in twin.decisions if d.kind == "alarm"]
        if alarms:
            c = pd.DataFrame(alarms[-1].payload["contributions"])
            fig = go.Figure(go.Bar(x=c.shap, y=c.meaning, orientation="h",
                                   marker=dict(color=[DIVERGE[1] if v > 0 else DIVERGE[0] for v in c.shap]),
                                   hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(style(fig, f"Leak alarm on {alarms[-1].payload['pipe_id']}: feature contributions", 340,
                                  ""), use_container_width=True, theme=None)
            st.caption("Red pushes towards *leak*, blue towards *normal* (log-odds).")
        rep = report_dir / "training_report.json"
        if rep.exists():
            r = json.loads(rep.read_text())
            imp = pd.Series(r["explainability"]["detection_global_shap"]).head(10)
            fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=SERIES[0]))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(style(fig, "Global detector importance (mean |SHAP|)", 320), use_container_width=True, theme=None)

# ---------------------------------------------------------------- asset risk
with tabs[3]:
    ar = twin.asset_risk().merge(pipes[["pipe_id", "health_score", "dma", "installation_year"]], on="pipe_id")
    ar = ar.sort_values("pred_failure_risk", ascending=False)
    cA, cB = st.columns([3, 2])
    with cA:
        st.subheader("Predictive-maintenance priority list")
        st.dataframe(ar.head(25)[["pipe_id", "material", "pipe_age", "corrosion", "repair_count", "health_score",
                                  "pred_failure_risk", "pred_remaining_useful_life", "dma"]]
                     .style.format({"pred_failure_risk": "{:.1%}", "corrosion": "{:.2f}", "health_score": "{:.0f}",
                                    "pred_remaining_useful_life": "{:.0f} y", "pipe_age": "{:.0f}"}),
                     use_container_width=True, hide_index=True, height=420)
    with cB:
        by = ar.groupby("material").pred_failure_risk.mean().sort_values()
        fig = go.Figure(go.Bar(x=by.values, y=by.index, orientation="h", marker_color=SERIES[0],
                               hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
        st.plotly_chart(style(fig, "Mean predicted annual failure risk by material", 260), use_container_width=True, theme=None)
        target = st.selectbox("Explain pipe", ar.pipe_id.head(25))
        if twin.failure is not None:
            from aquatwin.xai.explain import TreeExplainer, explain_risk
            ex = explain_risk(twin.failure, TreeExplainer(twin.failure.risk_model, twin.failure.features),
                              ar[ar.pipe_id == target].drop(columns=["pred_failure_risk", "pred_remaining_useful_life"]))
            st.info(ex["text"])
            c = ex["contributions"]
            fig = go.Figure(go.Bar(x=c.shap, y=c.meaning, orientation="h",
                                   marker=dict(color=[DIVERGE[1] if v > 0 else DIVERGE[0] for v in c.shap])))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(style(fig, "Risk drivers (SHAP, logit)", 260), use_container_width=True, theme=None)

# ---------------------------------------------------------------- water loss
with tabs[4]:
    if bal.empty:
        st.info("Advance the simulation first.")
    else:
        dma = st.selectbox("DMA", sorted(bal.dma.unique()))
        b = bal[bal.dma == dma]
        fig = go.Figure()
        fig.add_scatter(x=b.timestamp, y=b.estimated_loss_ls, name="AI estimate (DMA balance)", line=dict(color=SERIES[0], width=2))
        if show_truth:
            fig.add_scatter(x=b.timestamp, y=b.true_loss_ls, name="true loss", line=dict(color=SERIES[1], width=2))
        st.plotly_chart(style(fig, f"{dma}: real-loss estimate", 320, "L/s"), use_container_width=True, theme=None)
        tot = bal.groupby("dma")[["estimated_loss_ls", "true_loss_ls"]].sum() * 0.9
        tot.columns = ["estimated m³", "true m³"]
        st.dataframe(tot.round(1) if show_truth else tot[["estimated m³"]].round(1), use_container_width=True)

# ---------------------------------------------------------------- evaluation
with tabs[5]:
    ev_path = report_dir / "evaluation_levels.csv"
    if ev_path.exists():
        ev = pd.read_csv(ev_path)
        st.markdown("Fresh, unseen episodes at each realism level, run through the full production pipeline.")
        lab = [f"L{r.level} {r['name']}" for _, r in ev.iterrows()]
        c1, c2, c3 = st.columns(3)
        fig = go.Figure()
        for i, (col, nm) in enumerate([("detection_f1", "F1"), ("detection_accuracy", "accuracy")]):
            fig.add_scatter(x=lab, y=ev[col], name=nm, mode="lines+markers", line=dict(color=SERIES[i], width=2),
                            marker=dict(size=8))
        c1.plotly_chart(style(fig, "Leak detection (network alarm)", 300), use_container_width=True, theme=None)
        if "localization_error_m" in ev:
            fig = go.Figure()
            fig.add_scatter(x=lab, y=ev.localization_error_m, name="GNN", mode="lines+markers", line=dict(color=SERIES[0], width=2), marker=dict(size=8))
            fig.add_scatter(x=lab, y=ev.baseline_error_m, name="classical sensitivity", mode="lines+markers", line=dict(color=SERIES[1], width=2), marker=dict(size=8))
            c2.plotly_chart(style(fig, "Localisation error", 300, "m"), use_container_width=True, theme=None)
        fig = go.Figure()
        fig.add_bar(x=lab, y=ev.false_positives, name="false positives", marker_color=SERIES[0])
        fig.add_bar(x=lab, y=ev.false_negatives, name="false negatives", marker_color=SERIES[1])
        c3.plotly_chart(style(fig, "Alarm errors (time steps)", 300), use_container_width=True, theme=None)
        st.dataframe(ev, use_container_width=True, hide_index=True)
    else:
        st.info("Run `python -m aquatwin evaluate` to populate this tab.")
    tr = report_dir / "training_report.md"
    if tr.exists():
        with st.expander("Training report"):
            st.markdown(tr.read_text())

# ---------------------------------------------------------------- scenarios
with tabs[6]:
    for catname, items in scenarios_by_category().items():
        st.subheader(catname.title())
        for sd in items:
            st.markdown(f"**{sd.title}** (`{sd.name}`, level ≥ {sd.min_level})  \n" + "  \n".join(f"• {r}" for r in sd.rules))
    st.subheader("Events in this session (with causal explanation)")
    evs = twin.events()
    if not evs.empty:
        st.dataframe(evs[["start_time", "kind", "target", "cause"]], use_container_width=True, hide_index=True)
