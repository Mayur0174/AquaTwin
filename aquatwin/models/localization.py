"""GNN Leak Localisation Service (PyTorch Geometric).

Graph: nodes = junctions/tanks/sources, edges = pipelines (both directions).
Node features: twin residual, spatial contrast, temporal change, observability
mask, pressure, elevation, demand, sensor health, DMA water-balance, type.
Edge features: length, diameter, age, corrosion, material one-hot, flow residual.

Architecture (shared by the three evaluated encoders)::

    x --Linear--> h0 --[Conv + LayerNorm + ReLU + residual] x L--> h
    pipe (u->v):  MLP([h_u, h_v, |h_u-h_v|, edge_emb]) -> (leak logit, position logit)
    softmax over the pipes of each snapshot -> Leak Pipe Probability
    sigmoid(position logit) -> location on pipe -> distance from start junction

* GCN uses a physics-informed edge weight (hydraulic conductance
  D^2.63 / L^0.54 from Hazen-Williams).
* GraphSAGE uses mean aggregation (inductive, robust to missing sensors).
* GAT (GATv2) attends over neighbours using edge features.

A classical model-based baseline (leak sensitivity matrix + correlation,
Perez et al. 2011) is provided for comparison.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from aquatwin.models.features import (EDGE_FEATURES, NODE_FEATURES, NetworkContext, dma_balance, node_matrices,
                                      signature_correlation, snapshot_features)

try:  # torch is optional for the rest of the platform
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv
    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False


@dataclass
class Snapshot:
    x: np.ndarray            # N x Fn
    edge_attr: np.ndarray    # P x Fe
    target: np.ndarray       # P (leak share, sums to 1)
    frac: np.ndarray         # P (NaN where no leak)
    group: str
    step: int
    level: int


def _slog(v):
    return np.sign(v) * np.log1p(np.abs(v))


def leak_onsets(loss: np.ndarray, min_leak_ls: float) -> list[int]:
    """Time indices where a new pipe starts leaking above ``min_leak_ls`` (what an alarm reacts to)."""
    active = loss >= min_leak_ls
    onsets = []
    for t in range(len(loss)):
        new = active[t] & ~(active[t - 1] if t > 0 else np.zeros_like(active[t]))
        if new.any():
            onsets.append(t)
    return onsets


def build_snapshots(nodes_df: pd.DataFrame, pipes_df: pd.DataFrame, ctx: NetworkContext, group: str,
                    min_leak_ls: float = 1.0, per_onset: int = 4, sensitivity: np.ndarray | None = None
                    ) -> list[Snapshot]:
    """Onset-anchored snapshots: localisation runs when the alarm fires, against a pre-onset reference."""
    mats = node_matrices(nodes_df, ctx, group)
    pidx = {p: i for i, p in enumerate(ctx.pipe_ids)}
    out = []
    for gid, g in pipes_df.groupby(group, sort=False):
        m = mats[gid]
        g = g.sort_values(["step", "pipe_id"])
        steps = np.sort(g.step.unique())
        T, P = len(steps), len(ctx.pipe_ids)
        perm = np.array([pidx[p] for p in g.pipe_id.iloc[:P]])
        inv = np.argsort(perm)
        def tp(col, fill=np.nan):
            if col not in g:
                return np.full((T, P), fill)
            return g[col].to_numpy(float).reshape(T, P)[:, inv]
        flow, exp_flow, loss = tp("flow"), tp("expected_flow"), np.nan_to_num(tp("water_loss"))
        frac = tp("leak_frac")
        tsel = np.searchsorted(m["steps"], steps)
        ufw, _ = dma_balance(flow, exp_flow, m["demand"][tsel], ctx)
        level = int(g["level"].iloc[0]) if "level" in g else -1
        node_steps = list(m["steps"])
        for o in leak_onsets(loss, min_leak_ls):
            ref = node_steps.index(steps[o])
            for t in range(o, min(T, o + per_onset)):
                if loss[t].sum() < min_leak_ls:
                    break
                nt = node_steps.index(steps[t])
                x, ea = snapshot_features(m, nt, flow[t], exp_flow[t], ctx, ufw[t], sensitivity, ref_step=ref)
                x[:, :3] = _slog(x[:, :3])
                ea[:, 4] = _slog(ea[:, 4])
                tgt = np.where(loss[t] >= min_leak_ls * 0.5, loss[t], 0.0)
                tgt = tgt / tgt.sum()
                out.append(Snapshot(x, ea, tgt.astype(np.float32), frac[t], str(gid), int(steps[t]), level))
    return out


# ---------------------------------------------------------------------- model
if HAS_TORCH:
    class LeakLocGNN(nn.Module):
        def __init__(self, arch: str, in_dim: int, edge_dim: int, hidden: int = 64, layers: int = 3,
                     dropout: float = 0.1):
            super().__init__()
            self.arch = arch
            self.node_enc = nn.Linear(in_dim, hidden)
            self.edge_enc = nn.Linear(edge_dim, hidden)
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            for _ in range(layers):
                if arch == "gcn":
                    self.convs.append(GCNConv(hidden, hidden, add_self_loops=True, normalize=True))
                elif arch == "sage":
                    self.convs.append(SAGEConv(hidden, hidden, aggr="mean"))
                elif arch == "gat":
                    self.convs.append(GATv2Conv(hidden, hidden // 4, heads=4, edge_dim=hidden, add_self_loops=False))
                else:
                    raise ValueError(arch)
                self.norms.append(nn.LayerNorm(hidden))
            self.dropout = dropout
            self.head = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

        def forward(self, x, edge_index, edge_attr_dir, edge_weight, s_idx, e_idx, edge_attr_pipe):
            h = F.relu(self.node_enc(x))
            e_dir = self.edge_enc(edge_attr_dir)
            for conv, norm in zip(self.convs, self.norms):
                if self.arch == "gcn":
                    z = conv(h, edge_index, edge_weight)
                elif self.arch == "sage":
                    z = conv(h, edge_index)
                else:
                    z = conv(h, edge_index, e_dir)
                h = h + F.dropout(F.relu(norm(z)), self.dropout, self.training)
            hs, he = h[s_idx], h[e_idx]
            out = self.head(torch.cat([hs, he, (hs - he).abs(), self.edge_enc(edge_attr_pipe)], dim=1))
            return out[:, 0], out[:, 1]


class GNNLocalizer:
    def __init__(self, ctx: NetworkContext, arch: str = "sage", hidden: int = 64, layers: int = 3, seed: int = 0):
        if not HAS_TORCH:
            raise ImportError("torch + torch_geometric are required for the GNN localiser")
        torch.manual_seed(seed)
        self.ctx, self.arch, self.hidden, self.layers = ctx, arch, hidden, layers
        self.N, self.P = len(ctx.node_ids), len(ctx.pipe_ids)
        p = ctx.net.pipes
        cond = (p.diameter_mm.to_numpy() / 1000) ** 2.63 / p.length_m.to_numpy() ** 0.54
        self.conductance = cond / cond.max()
        self.model = LeakLocGNN(arch, len(NODE_FEATURES), len(EDGE_FEATURES), hidden, layers)
        self.history: list[dict] = []

    def _batch(self, snaps: list[Snapshot]):
        B, N, P = len(snaps), self.N, self.P
        x = torch.tensor(np.concatenate([s.x for s in snaps]), dtype=torch.float32)
        ea = torch.tensor(np.concatenate([s.edge_attr for s in snaps]), dtype=torch.float32)
        off = np.repeat(np.arange(B) * N, P)
        s_idx = torch.tensor(np.tile(self.ctx.s_idx, B) + off, dtype=torch.long)
        e_idx = torch.tensor(np.tile(self.ctx.e_idx, B) + off, dtype=torch.long)
        ei = torch.cat([torch.stack([s_idx, e_idx]), torch.stack([e_idx, s_idx])], dim=1)
        ea_dir = torch.cat([ea, ea], dim=0)
        w = torch.tensor(np.tile(self.conductance, 2 * B), dtype=torch.float32)
        return x, ei, ea_dir, w, s_idx, e_idx, ea

    def forward(self, snaps):
        logit, pos = self.model(*self._batch(snaps))
        return logit.view(len(snaps), self.P), torch.sigmoid(pos).view(len(snaps), self.P)

    def fit(self, train: list[Snapshot], val: list[Snapshot], epochs: int = 60, lr: float = 3e-3,
            batch_size: int = 32, verbose: bool = False) -> dict:
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        best, best_state = np.inf, None
        rng = np.random.default_rng(0)
        for ep in range(epochs):
            self.model.train()
            order = rng.permutation(len(train))
            tot = 0.0
            for i in range(0, len(order), batch_size):
                batch = [train[j] for j in order[i:i + batch_size]]
                logit, pos = self.forward(batch)
                tgt = torch.tensor(np.stack([s.target for s in batch]))
                frac = torch.tensor(np.nan_to_num(np.stack([s.frac for s in batch]), nan=0.5), dtype=torch.float32)
                ce = -(tgt * F.log_softmax(logit, dim=1)).sum(1).mean()
                pos_loss = ((pos - frac).abs() * tgt).sum(1).mean()
                loss = ce + 2.0 * pos_loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                opt.step()
                tot += float(loss) * len(batch)
            sched.step()
            m = self.evaluate(val, fast=True)
            self.history.append(dict(epoch=ep, loss=tot / len(train), **m))
            score = m["mean_error_m"]
            if score < best:
                best = score
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            if verbose and ep % 10 == 0:
                print(f"    [{self.arch}] epoch {ep:3d} loss {tot / len(train):.3f} top1 {m['top1']:.3f} err {score:.0f} m")
        if best_state:
            self.model.load_state_dict(best_state)
        return self.history[-1]

    @torch.no_grad() if HAS_TORCH else (lambda f: f)
    def predict(self, snaps: list[Snapshot], batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        probs, poss = [], []
        for i in range(0, len(snaps), batch_size):
            logit, pos = self.forward(snaps[i:i + batch_size])
            probs.append(F.softmax(logit, dim=1).numpy())
            poss.append(pos.numpy())
        return np.concatenate(probs), np.concatenate(poss)

    def evaluate(self, snaps: list[Snapshot], fast: bool = False) -> dict:
        if not snaps:
            return {}
        prob, pos = self.predict(snaps)
        return localization_metrics(self.ctx, snaps, prob, pos, fast=fast)

    def saliency(self, snap: Snapshot) -> pd.Series:
        """Gradient x input of the top pipe logit w.r.t. node residual features (which sensors drove it)."""
        self.model.eval()
        x, ei, ea_dir, w, s, e, ea = self._batch([snap])
        x.requires_grad_(True)
        logit, _ = self.model(x, ei, ea_dir, w, s, e, ea)
        logit[int(torch.argmax(logit))].backward()
        sal = (x.grad[:, :3] * x[:, :3]).abs().sum(1).detach().numpy()
        return pd.Series(sal, index=self.ctx.node_ids).sort_values(ascending=False)

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), d / f"gnn_{self.arch}.pt")
        (d / f"gnn_{self.arch}.json").write_text(json.dumps(dict(arch=self.arch, hidden=self.hidden, layers=self.layers)))

    @classmethod
    def load(cls, directory: str | Path, ctx: NetworkContext, arch: str) -> "GNNLocalizer":
        d = Path(directory)
        meta = json.loads((d / f"gnn_{arch}.json").read_text())
        obj = cls(ctx, meta["arch"], meta["hidden"], meta["layers"])
        obj.model.load_state_dict(torch.load(d / f"gnn_{arch}.pt", weights_only=True))
        return obj


# ---------------------------------------------------------------------- metrics
def localization_metrics(ctx: NetworkContext, snaps: list[Snapshot], prob: np.ndarray, pos: np.ndarray,
                         fast: bool = False) -> dict:
    net = ctx.net
    errs, top1, top3, top5 = [], [], [], []
    for s, p, q in zip(snaps, prob, pos):
        true_idx = np.nonzero(s.target > 0)[0]
        rank = np.argsort(-p)
        top1.append(rank[0] in true_idx)
        top3.append(bool(set(rank[:3]) & set(true_idx)))
        top5.append(bool(set(rank[:5]) & set(true_idx)))
        k = rank[0]
        pid = ctx.pipe_ids[k]
        if fast and k in true_idx:
            errs.append(abs(float(q[k]) - float(np.nan_to_num(s.frac[k], nan=0.5))) * net.pipes.length_m.iat[k])
            continue
        e = min(net.point_distance(pid, float(q[k]), ctx.pipe_ids[t], float(np.nan_to_num(s.frac[t], nan=0.5)))
                for t in true_idx)
        errs.append(e)
    errs = np.array(errs)
    return dict(top1=float(np.mean(top1)), top3=float(np.mean(top3)), top5=float(np.mean(top5)),
                mean_error_m=float(errs.mean()), median_error_m=float(np.median(errs)),
                within_100m=float((errs <= 100).mean()), within_250m=float((errs <= 250).mean()), n=len(snaps))


def predictions_table(ctx: NetworkContext, prob: np.ndarray, pos: np.ndarray, k: int = 5) -> list[pd.DataFrame]:
    out = []
    pipes = ctx.net.pipes
    for p, q in zip(prob, pos):
        rank = np.argsort(-p)[:k]
        rows = []
        for i in rank:
            pid = ctx.pipe_ids[i]
            lat, lon = ctx.net.point_on_pipe(pid, float(q[i]))
            rows.append(dict(pipe_id=pid, probability=float(p[i]), position_frac=float(q[i]),
                             distance_from_start_m=float(q[i] * pipes.at[pid, "length_m"]),
                             start_node=pipes.at[pid, "start_node"], lat=lat, lon=lon))
        out.append(pd.DataFrame(rows))
    return out


# ---------------------------------------------------------------------- classical baseline
class SensitivityLocalizer:
    """Model-based baseline: correlate the residual vector with simulated leak signatures."""

    def __init__(self, ctx: NetworkContext, cfg=None, leak_mm: float = 25.0):
        from aquatwin.physics.degradation import hazen_williams_c
        from aquatwin.physics.hydraulics import solve
        from aquatwin.scenarios.events import LeakSpec, leak_area_m2
        from aquatwin.truth.builder import ModelBuilder
        net = ctx.net
        b = ModelBuilder(net)
        n, p = net.nodes, net.pipes
        dem = n.base_demand_ls.fillna(0).to_numpy() / 1000.0
        hw = hazen_williams_c(p.material.to_numpy(), p.age_years.to_numpy(float))
        levels = n.loc[b.tank_ids, "init_level"].to_numpy(float)
        base, _ = b.build(dem, levels, np.array([1.0]), hw, [])
        r0 = solve(base)
        a = leak_area_m2(leak_mm)
        S = np.zeros((len(n), len(p)))
        for j, pid in enumerate(p.index):
            m, _ = b.build(dem, levels, np.array([1.0]), hw, [LeakSpec("sig", pid, 0.5, a, 0.0)])
            r = solve(m)
            S[:, j] = r.pressure[:len(n)] - r0.pressure
        self.S = S
        self.ctx = ctx

    def predict(self, snaps: list[Snapshot]) -> tuple[np.ndarray, np.ndarray]:
        probs = []
        for s in snaps:
            mask = s.x[:, 3] > 0
            r = np.sign(s.x[:, 1]) * np.expm1(np.abs(s.x[:, 1]))     # undo slog of contrast residual
            corr = signature_correlation(self.S, r, mask)
            z = np.exp(8 * corr)
            probs.append(z / z.sum())
        return np.array(probs), np.full((len(snaps), self.S.shape[1]), 0.5)

    def save(self, directory: str | Path) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        np.save(Path(directory) / "sensitivity.npy", self.S)

    @classmethod
    def load(cls, directory: str | Path, ctx: NetworkContext) -> "SensitivityLocalizer":
        obj = cls.__new__(cls)
        obj.S, obj.ctx = np.load(Path(directory) / "sensitivity.npy"), ctx
        return obj
