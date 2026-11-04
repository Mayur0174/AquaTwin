"""Feature engineering shared by detection, localisation and the live twin.

Design principles
-----------------
* **Model-based residuals.** Raw pressure is dominated by elevation and
  demand; the leak signal lives in ``observed - twin_expected``.
* **Spatial contrast (common-mode rejection).** Demand surges and tank level
  changes move *all* residuals; leaks move residuals *locally*.  Subtracting
  the network median residual removes the common mode (Leak + High Demand).
* **Hydraulic proximity.** Pipes inherit evidence from monitored nodes
  weighted by inverse network distance (Rule 5: closer sensors see more).
* **Water balance.** DMA inflow (boundary meters) minus AMI consumption =
  unaccounted flow, the utility-standard top-down leak indicator (MNF).
* **Temporal change.** Residual change versus 1-2 h earlier removes static
  model mismatch and sensor bias.
* NaN is kept as NaN: XGBoost learns default directions for missing data,
  which is exactly the situation with packet loss and failed sensors.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from aquatwin.network.materials import MATERIAL_NAMES
from aquatwin.network.model import WaterNetwork
from aquatwin.sensors.registry import SensorRegistry

NEIGHBOURHOOD_M = 700.0
DETECTION_FEATURES = [
    # spec inputs
    "pressure", "flow", "demand", "corrosion", "pipe_health", "sensor_health",
    # engineered
    "res_start", "res_end", "res_local", "nbr_res", "nbr_min", "nbr_contrast", "nbr_count",
    "nbr_delta", "nbr_roll4", "flow_res", "flow_res_rel", "dma_ufw", "dma_ufw_rel", "dma_ufw_roll4",
    "net_median_res", "hour_sin", "hour_cos", "is_night", "pipe_age", "diameter_mm", "length_m",
    "repair_count", "material_code", "nbr_min_rank", "contrast_rank", "flow_res_rank", "net_min_contrast",
    "gap_to_min",
]


@dataclass
class NetworkContext:
    net: WaterNetwork
    node_ids: list[str]
    pipe_ids: list[str]
    W: np.ndarray                 # pipes x nodes proximity weights (0 beyond radius)
    near: np.ndarray              # pipes x nodes bool within radius
    zone_in: dict[str, list[tuple[int, int]]]  # dma -> [(pipe index, +1/-1 sign of inflow)]
    node_dma: np.ndarray
    dmas: list[str]
    edge_index: np.ndarray        # 2 x E (both directions) on node indices
    s_idx: np.ndarray
    e_idx: np.ndarray

    @classmethod
    def build(cls, net: WaterNetwork, sensors: SensorRegistry | None = None) -> "NetworkContext":
        nodes, pipes = net.nodes, net.pipes
        node_ids, pipe_ids = list(nodes.index), list(pipes.index)
        D = net.distance_matrix.reindex(index=node_ids, columns=node_ids).fillna(1e9).to_numpy()
        nidx = {n: i for i, n in enumerate(node_ids)}
        s = np.array([nidx[n] for n in pipes.start_node])
        e = np.array([nidx[n] for n in pipes.end_node])
        L = pipes.length_m.to_numpy(float)
        dist = np.minimum(D[s] + 0.5 * L[:, None], D[e] + 0.5 * L[:, None])     # pipe midpoint -> node
        near = dist <= NEIGHBOURHOOD_M
        W = np.where(near, 1.0 / (dist + 50.0), 0.0)
        ntype = nodes.node_type.to_numpy()
        zone = np.where(np.isin(ntype, ["tank", "pump_station", "reservoir"]), "SRC", nodes.dma.to_numpy())
        zone_in: dict[str, list[tuple[int, int]]] = {}
        for i in range(len(pipe_ids)):
            za, zb = zone[s[i]], zone[e[i]]
            if za == zb:
                continue
            if zb != "SRC":
                zone_in.setdefault(zb, []).append((i, +1))    # positive flow enters zb
            if za != "SRC":
                zone_in.setdefault(za, []).append((i, -1))
        ei = np.concatenate([np.stack([s, e]), np.stack([e, s])], axis=1)
        return cls(net, node_ids, pipe_ids, W, near, zone_in, zone, sorted(zone_in), ei, s, e)


def _hour(ts: pd.Series) -> np.ndarray:
    t = pd.to_datetime(ts)
    return (t.dt.hour + t.dt.minute / 60.0).to_numpy()


def node_matrices(nodes_df: pd.DataFrame, ctx: NetworkContext, group: str) -> dict[str, dict[str, np.ndarray]]:
    """Per group (episode/sample): T x N arrays of residuals, health, demand."""
    out = {}
    for gid, g in nodes_df.groupby(group, sort=False):
        g = g.sort_values(["step", "node_id"])
        T = g.step.nunique()
        order = pd.Index(sorted(ctx.node_ids))
        perm = np.array([order.get_loc(n) for n in ctx.node_ids])
        def mat(col):
            return g[col].to_numpy(float).reshape(T, len(order))[:, perm]
        obs, exp = mat("observed_pressure"), mat("expected_pressure")
        dcol = "twin_demand" if "twin_demand" in g else "observed_demand"
        out[gid] = dict(res=obs - exp, obs=obs, exp=exp, health=mat("sensor_health"), demand=mat(dcol),
                        steps=np.sort(g.step.unique()), timestamps=g.drop_duplicates("step").timestamp.to_numpy())
    return out


def dma_balance(flow: np.ndarray, exp_flow: np.ndarray, demand: np.ndarray, ctx: NetworkContext
                ) -> tuple[np.ndarray, np.ndarray]:
    """Unaccounted-for flow per DMA (T x n_dma): metered net inflow - AMI consumption."""
    flow = np.where(np.isnan(flow), exp_flow, flow)
    T = flow.shape[0]
    ufw = np.zeros((T, len(ctx.dmas)))
    cons = np.zeros((T, len(ctx.dmas)))
    dem = np.nan_to_num(demand)
    for k, dma in enumerate(ctx.dmas):
        inflow = np.zeros(T)
        for i, sgn in ctx.zone_in[dma]:
            inflow += sgn * flow[:, i]
        c = dem[:, ctx.node_dma == dma].sum(axis=1)
        ufw[:, k] = inflow - c
        cons[:, k] = c
    return ufw, cons


def detection_features(pipes_df: pd.DataFrame, nodes_df: pd.DataFrame, ctx: NetworkContext,
                       group: str = "episode_id") -> pd.DataFrame:
    mats = node_matrices(nodes_df, ctx, group)
    frames = []
    pidx = {p: i for i, p in enumerate(ctx.pipe_ids)}
    mat_code = {m: i for i, m in enumerate(MATERIAL_NAMES)}
    for gid, g in pipes_df.groupby(group, sort=False):
        m = mats[gid]
        g = g.sort_values(["step", "pipe_id"]).reset_index(drop=True)
        T, P = g.step.nunique(), len(ctx.pipe_ids)
        perm = np.array([pidx[p] for p in g.pipe_id.iloc[:P]])
        inv = np.argsort(perm)
        tsel = np.searchsorted(m["steps"], np.sort(g.step.unique()))    # align node steps to pipe steps
        res = m["res"][tsel]
        valid = ~np.isnan(res)
        r0 = np.nan_to_num(res)
        net_med = np.nanmedian(np.where(valid, res, np.nan), axis=1)               # T
        num = r0 @ ctx.W.T
        den = valid.astype(float) @ ctx.W.T
        nbr = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)              # T x P
        cnt = valid.astype(float) @ ctx.near.T.astype(float)
        big = np.where(valid, res, np.inf)
        nbr_min = np.full((T, P), np.nan)
        for i in range(P):
            sel = ctx.near[i]
            if sel.any():
                v = big[:, sel].min(axis=1)
                nbr_min[:, i] = np.where(np.isinf(v), np.nan, v)
        contrast = nbr - net_med[:, None]
        nbr_s = pd.DataFrame(contrast)
        roll4 = nbr_s.rolling(4, min_periods=1).mean().to_numpy()
        delta = contrast - nbr_s.shift(6).rolling(3, min_periods=1).mean().to_numpy()

        def tp(col):
            return g[col].to_numpy(float).reshape(T, P)[:, inv]                     # T x P in ctx order

        flow, exp_flow = tp("flow"), tp("expected_flow")
        # node-level AMI consumption for the DMA balance
        ufw, cons = dma_balance(flow, exp_flow, m["demand"][tsel], ctx)
        pipe_dma = np.array([ctx.node_dma[ctx.s_idx[i]] if ctx.node_dma[ctx.s_idx[i]] != "SRC"
                             else ctx.node_dma[ctx.e_idx[i]] for i in range(P)])
        dk = np.array([ctx.dmas.index(d) if d in ctx.dmas else -1 for d in pipe_dma])
        ufw_p = np.where(dk >= 0, ufw[:, np.maximum(dk, 0)], np.nan)
        cons_p = np.where(dk >= 0, cons[:, np.maximum(dk, 0)], np.nan)
        ufw_roll = pd.DataFrame(ufw_p).rolling(4, min_periods=1).mean().to_numpy()

        feat = {
            "res_start": tp("pressure") - tp("expected_pressure"),
            "res_end": tp("pressure_end") - tp("expected_pressure_end"),
            "nbr_res": nbr, "nbr_min": nbr_min - net_med[:, None], "nbr_contrast": contrast, "nbr_count": cnt,
            "nbr_delta": delta, "nbr_roll4": roll4,
            "flow_res": flow - exp_flow, "flow_res_rel": (flow - exp_flow) / np.maximum(np.abs(exp_flow), 1.0),
            "dma_ufw": ufw_p, "dma_ufw_rel": ufw_p / np.maximum(cons_p, 1.0), "dma_ufw_roll4": ufw_roll,
            "net_median_res": np.repeat(net_med[:, None], P, 1),
        }
        out = g.copy()
        order = perm  # g rows (within step) map to ctx index perm -> back to g order
        for k, v in feat.items():
            out[k] = v[:, perm].reshape(-1)
        out["res_local"] = np.fmin(out.res_start.to_numpy(), out.res_end.to_numpy())
        h = _hour(out.timestamp)
        out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
        out["is_night"] = ((h >= 0) & (h < 5)).astype(int)
        out["material_code"] = out.material.map(mat_code).astype(int)
        # cross-sectional context: is this pipe in the most anomalous part of the network right now?
        grp = out.groupby("step")
        out["nbr_min_rank"] = grp.nbr_min.rank(pct=True)
        out["contrast_rank"] = grp.nbr_contrast.rank(pct=True)
        out["flow_res_rank"] = grp.flow_res_rel.rank(pct=True, ascending=False)
        out["net_min_contrast"] = grp.nbr_contrast.transform("min")
        out["gap_to_min"] = out.nbr_contrast - out.net_min_contrast
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------- GNN snapshot features
NODE_FEATURES = ["res", "res_contrast", "res_delta", "mask", "obs_norm", "elev_norm", "demand_norm",
                 "health", "dma_ufw", "is_junction", "is_tank", "is_source"]
EDGE_FEATURES = ["length_norm", "diameter_norm", "age_norm", "corrosion", "flow_res", "flow_mask"] + \
                [f"mat_{m}" for m in MATERIAL_NAMES] + ["sens_corr", "sens_rank"]


def static_edge_features(ctx: NetworkContext) -> np.ndarray:
    p = ctx.net.pipes
    mats = np.array([[float(m == name) for name in MATERIAL_NAMES] for m in p.material])
    base = np.stack([p.length_m / 300.0, p.diameter_mm / 300.0, p.age_years / 50.0, p.corrosion_index], axis=1)
    return np.concatenate([base.to_numpy(float) if hasattr(base, "to_numpy") else base, mats], axis=1)


def snapshot_features(m: dict[str, np.ndarray], t: int, flow_row: np.ndarray, exp_flow_row: np.ndarray,
                      ctx: NetworkContext, ufw_row: np.ndarray | None = None,
                      sensitivity: np.ndarray | None = None, ref_step: int | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Node (N x F) and edge (P x Fe) features for one time index ``t`` of a group matrix dict."""
    nodes = ctx.net.nodes.loc[ctx.node_ids]
    res = m["res"][t]
    valid = ~np.isnan(res)
    med = np.nanmedian(res[valid]) if valid.any() else 0.0
    r_end = t if ref_step is None else ref_step          # frozen pre-alarm reference window
    past = m["res"][max(0, r_end - 5):max(0, r_end - 1)]
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)    # all-NaN columns = sensor silent before onset
        base = np.nanmean(past, axis=0) if len(past) else np.full_like(res, np.nan)
    delta = np.where(valid & ~np.isnan(base), res - base, 0.0)
    obs = np.nan_to_num(m["obs"][t])
    elev = nodes.elevation.to_numpy(float)
    dem = np.nan_to_num(m["demand"][t])
    ntype = nodes.node_type.to_numpy()
    ufw_node = np.zeros(len(ctx.node_ids))
    if ufw_row is not None:
        for k, d in enumerate(ctx.dmas):
            ufw_node[ctx.node_dma == d] = ufw_row[k] / 10.0
    x = np.stack([
        np.where(valid, res, 0.0), np.where(valid, res - med, 0.0), delta, valid.astype(float),
        np.where(valid, obs / 50.0, 0.0), (elev - elev.mean()) / 10.0, dem / 2.0,
        np.nan_to_num(m["health"][t], nan=0.0), ufw_node,
        (ntype == "junction").astype(float), (ntype == "tank").astype(float),
        np.isin(ntype, ["reservoir", "pump_station"]).astype(float)], axis=1)
    fmask = ~np.isnan(flow_row)
    fres = np.where(fmask, (flow_row - exp_flow_row) / np.maximum(np.abs(exp_flow_row), 1.0), 0.0)
    e_static = static_edge_features(ctx)
    corr = np.zeros(len(ctx.pipe_ids))
    if sensitivity is not None and valid.sum() >= 3:
        corr = signature_correlation(sensitivity, np.where(valid, res - med, 0.0), valid)
    rank = np.argsort(np.argsort(-corr)) / max(1, len(corr) - 1)
    ea = np.concatenate([e_static[:, :4], fres[:, None], fmask[:, None].astype(float), e_static[:, 4:],
                         corr[:, None], (1 - rank)[:, None]], axis=1)
    return x.astype(np.float32), ea.astype(np.float32)


def signature_correlation(S: np.ndarray, residual: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Pearson correlation between the residual vector and each pipe's simulated leak signature."""
    Sv = S[valid]
    rv = residual[valid]
    Sc = Sv - Sv.mean(0)
    rc = rv - rv.mean()
    return (Sc * rc[:, None]).sum(0) / (np.linalg.norm(Sc, axis=0) * np.linalg.norm(rc) + 1e-12)
