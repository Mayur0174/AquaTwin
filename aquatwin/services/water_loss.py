"""Water Loss Estimation Service.

Two complementary, utility-standard estimators:

1. **DMA water balance (top-down):** metered DMA net inflow minus AMI
   consumption = unaccounted-for water (real losses + meter error).
2. **Minimum Night Flow (MNF):** at 02:00-04:00 legitimate use is minimal,
   so night unaccounted flow is the best estimate of leakage; converted to a
   daily volume with the FAVAD night-day factor (pressure is higher at night).

Both are computed only from observations; truth is used only to score them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aquatwin.models.features import NetworkContext, dma_balance, node_matrices

NIGHT_HOURS = (2.0, 4.0)
NIGHT_DAY_FACTOR = 0.85      # FAVAD: daytime pressure lower -> leak flow lower than at night


def dma_water_balance(pipes_df: pd.DataFrame, nodes_df: pd.DataFrame, ctx: NetworkContext,
                      group: str = "episode_id") -> pd.DataFrame:
    mats = node_matrices(nodes_df, ctx, group)
    pidx = {p: i for i, p in enumerate(ctx.pipe_ids)}
    node_dma = ctx.node_dma
    rows = []
    for gid, g in pipes_df.groupby(group, sort=False):
        m = mats[gid]
        g = g.sort_values(["step", "pipe_id"])
        steps = np.sort(g.step.unique())
        T, P = len(steps), len(ctx.pipe_ids)
        inv = np.argsort(np.array([pidx[p] for p in g.pipe_id.iloc[:P]]))
        tp = lambda c: g[c].to_numpy(float).reshape(T, P)[:, inv]
        tsel = np.searchsorted(m["steps"], steps)
        ufw, cons = dma_balance(tp("flow"), tp("expected_flow"), m["demand"][tsel], ctx)
        loss = np.nan_to_num(tp("water_loss"))
        ts = g.drop_duplicates("step").timestamp.to_numpy()
        pipe_dma = np.array([node_dma[ctx.s_idx[i]] if node_dma[ctx.s_idx[i]] != "SRC" else node_dma[ctx.e_idx[i]]
                             for i in range(P)])
        for k, d in enumerate(ctx.dmas):
            true_loss = loss[:, pipe_dma == d].sum(axis=1)
            for t in range(T):
                rows.append(dict(group=gid, step=int(steps[t]), timestamp=ts[t], dma=d, unaccounted_ls=ufw[t, k],
                                 consumption_ls=cons[t, k], true_loss_ls=true_loss[t]))
    df = pd.DataFrame(rows)
    df["estimated_loss_ls"] = (df.groupby(["group", "dma"]).unaccounted_ls
                               .transform(lambda s: s.rolling(4, min_periods=1).mean()).clip(lower=0))
    return df


def minimum_night_flow(balance: pd.DataFrame) -> pd.DataFrame:
    h = pd.to_datetime(balance.timestamp)
    hour = h.dt.hour + h.dt.minute / 60
    night = balance[(hour >= NIGHT_HOURS[0]) & (hour < NIGHT_HOURS[1])]
    g = night.groupby(["group", "dma"]).agg(mnf_unaccounted_ls=("unaccounted_ls", "mean"),
                                            true_night_loss_ls=("true_loss_ls", "mean")).reset_index()
    g["estimated_daily_loss_m3"] = g.mnf_unaccounted_ls.clip(lower=0) * NIGHT_DAY_FACTOR * 86.4
    return g


def loss_estimation_metrics(balance: pd.DataFrame) -> dict:
    err = balance.estimated_loss_ls - balance.true_loss_ls
    vol_true = balance.groupby("group").true_loss_ls.sum() * 0.9   # 15-min steps -> m3
    vol_est = balance.groupby("group").estimated_loss_ls.sum() * 0.9
    return dict(mae_ls=float(err.abs().mean()), bias_ls=float(err.mean()),
                rmse_ls=float(np.sqrt((err ** 2).mean())),
                volume_error_pct=float(((vol_est - vol_true).abs().sum() / max(vol_true.sum(), 1e-9)) * 100),
                corr=float(np.corrcoef(balance.estimated_loss_ls, balance.true_loss_ls)[0, 1])
                if balance.true_loss_ls.std() > 0 else float("nan"))
