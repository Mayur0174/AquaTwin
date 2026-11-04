"""Builds a per-step :class:`HydraulicModel` from the asset model + world state.

Leaks are physical: the leaking pipe is split at the leak position into two
sub-pipes joined by a virtual node that carries a FAVAD emitter.  Hence the
true leak location (pipe, fraction, metres from the start junction) is a
first-class part of the hydraulic truth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aquatwin.network.model import WaterNetwork
from aquatwin.physics.hydraulics import PIPE, PUMP, HydraulicModel
from aquatwin.scenarios.events import LeakSpec


@dataclass
class BuildMap:
    n_base_nodes: int
    first_seg: np.ndarray        # per base pipe -> link index of the start-side segment
    last_seg: np.ndarray         # per base pipe -> link index of the end-side segment
    pump_links: np.ndarray
    leak_pipe_idx: np.ndarray    # per emitter -> base pipe index
    leaks: list[LeakSpec]
    signature: tuple


class ModelBuilder:
    def __init__(self, network: WaterNetwork):
        self.net = network
        nodes, pipes, pumps = network.nodes, network.pipes, network.pumps
        self.node_ids = list(nodes.index)
        self.nidx = {n: i for i, n in enumerate(self.node_ids)}
        self.elev = nodes.elevation.to_numpy(float)
        types = nodes.node_type.to_numpy()
        self.is_reservoir = types == "reservoir"
        self.is_tank = types == "tank"
        self.fixed = self.is_reservoir | self.is_tank
        self.res_head = np.where(self.is_reservoir, nodes.get("head", np.nan).to_numpy(float), np.nan)
        self.tank_ids = list(nodes.index[self.is_tank])
        self.tank_idx = np.array([self.nidx[t] for t in self.tank_ids], int)
        self.tank_area = np.pi * nodes.loc[self.tank_ids, "tank_diameter"].to_numpy(float) ** 2 / 4.0
        self.pipe_ids = list(pipes.index)
        self.pidx = {p: i for i, p in enumerate(self.pipe_ids)}
        self.p_start = np.array([self.nidx[n] for n in pipes.start_node])
        self.p_end = np.array([self.nidx[n] for n in pipes.end_node])
        self.length = pipes.length_m.to_numpy(float)
        self.diameter = pipes.diameter_mm.to_numpy(float) / 1000.0
        self.pump_ids = list(pumps.index)
        self.pu_start = np.array([self.nidx[n] for n in pumps.start_node])
        self.pu_end = np.array([self.nidx[n] for n in pumps.end_node])
        self.pu_a = pumps.pump_a.to_numpy(float)
        self.pu_b = pumps.pump_b.to_numpy(float)

    def build(self, demand_m3s: np.ndarray, tank_levels: np.ndarray, pump_speed: np.ndarray,
              hw_c: np.ndarray, leaks: list[LeakSpec], closed: set[str] = frozenset(),
              throttled: dict[str, float] | None = None, blockage: dict[str, float] | None = None
              ) -> tuple[HydraulicModel, BuildMap]:
        throttled = throttled or {}
        blockage = blockage or {}
        n0, np_ = len(self.node_ids), len(self.pipe_ids)
        # merge leaks on the same pipe (areas add, position = area-weighted)
        merged: dict[str, LeakSpec] = {}
        for lk in leaks:
            if lk.pipe_id in merged:
                m = merged[lk.pipe_id]
                tot = m.area_m2 + lk.area_m2
                merged[lk.pipe_id] = LeakSpec(m.leak_id, m.pipe_id, (m.frac * m.area_m2 + lk.frac * lk.area_m2) / tot,
                                              tot, m.favad_slope + lk.favad_slope, m.cd, m.category, m.origin)
            else:
                merged[lk.pipe_id] = lk
        leak_list = list(merged.values())
        nl = len(leak_list)

        elev = np.concatenate([self.elev, np.zeros(nl)])
        fixed = np.concatenate([self.fixed, np.zeros(nl, bool)])
        fixed_head = np.full(n0 + nl, np.nan)
        res = np.nonzero(self.is_reservoir)[0]
        fixed_head[res] = self.res_head[res]
        fixed_head[self.tank_idx] = self.elev[self.tank_idx] + tank_levels
        demand = np.concatenate([demand_m3s, np.zeros(nl)])
        node_ids = self.node_ids + [f"L:{lk.pipe_id}" for lk in leak_list]

        start = self.p_start.copy()
        end = self.p_end.copy()
        length = self.length.copy()
        dia = self.diameter.copy()
        c = hw_c.copy()
        k = np.zeros(np_)
        is_open = np.ones(np_, bool)
        for pid, frac in blockage.items():
            i = self.pidx[pid]
            dia[i] *= np.sqrt(max(0.05, 1.0 - frac))
        for pid, kk in throttled.items():
            k[self.pidx[pid]] += kk
        for pid in closed:
            if pid in self.pidx:
                is_open[self.pidx[pid]] = False

        last_seg = np.arange(np_)
        extra = {"start": [], "end": [], "length": [], "dia": [], "c": [], "k": [], "open": []}
        em_node, em_a0, em_slope, em_cd, em_pipe = [], [], [], [], []
        for j, lk in enumerate(leak_list):
            i = self.pidx[lk.pipe_id]
            ln = n0 + j
            f = float(np.clip(lk.frac, 0.01, 0.99))
            elev[ln] = self.elev[start[i]] + f * (self.elev[end[i]] - self.elev[start[i]])
            extra["start"].append(ln)
            extra["end"].append(end[i])
            extra["length"].append(length[i] * (1 - f))
            extra["dia"].append(dia[i])
            extra["c"].append(c[i])
            extra["k"].append(k[i] * 0.5)
            extra["open"].append(is_open[i])
            last_seg[i] = np_ + j
            end[i] = ln
            length[i] = length[i] * f
            k[i] *= 0.5
            em_node.append(ln)
            em_a0.append(lk.area_m2)
            em_slope.append(lk.favad_slope)
            em_cd.append(lk.cd)
            em_pipe.append(i)

        npu = len(self.pump_ids)
        all_start = np.concatenate([start, np.array(extra["start"], int), self.pu_start])
        all_end = np.concatenate([end, np.array(extra["end"], int), self.pu_end])
        kind = np.concatenate([np.full(np_ + nl, PIPE), np.full(npu, PUMP)])
        L = np.concatenate([length, np.array(extra["length"]), np.full(npu, 10.0)])
        D = np.concatenate([dia, np.array(extra["dia"]), np.full(npu, 0.6)])
        C = np.concatenate([c, np.array(extra["c"]), np.full(npu, 130.0)])
        K = np.concatenate([k, np.array(extra["k"]), np.zeros(npu)])
        op = np.concatenate([is_open, np.array(extra["open"], bool), np.ones(npu, bool)])
        link_ids = self.pipe_ids + [f"{lk.pipe_id}#b" for lk in leak_list] + self.pump_ids
        model = HydraulicModel(
            node_ids=node_ids, elevation=elev, fixed=fixed, fixed_head=fixed_head, demand=demand,
            link_ids=link_ids, start=all_start, end=all_end, kind=kind, length=L, diameter=D, hw_c=C,
            minor_k=K, is_open=op, pump_a=np.concatenate([np.zeros(np_ + nl), self.pu_a]),
            pump_b=np.concatenate([np.zeros(np_ + nl), self.pu_b]),
            pump_speed=np.concatenate([np.zeros(np_ + nl), pump_speed]),
            emitter_node=np.array(em_node, int), emitter_a0=np.array(em_a0, float),
            emitter_slope=np.array(em_slope, float), emitter_cd=np.array(em_cd, float))
        bmap = BuildMap(n_base_nodes=n0, first_seg=np.arange(np_), last_seg=last_seg,
                        pump_links=np.arange(np_ + nl, np_ + nl + npu), leak_pipe_idx=np.array(em_pipe, int),
                        leaks=leak_list, signature=tuple(sorted(lk.pipe_id for lk in leak_list)))
        return model, bmap


def solve_with_tank_rules(builder: "ModelBuilder", demand_m3s, tank_levels, tank_min, tank_max, pump_speed, hw_c,
                          leaks, closed, throttled=None, blockage=None, sim=None, warm=None, tol_level=1e-6):
    """Solve, then apply the EPANET tank rule (empty tanks cannot drain, full tanks cannot fill)."""
    from aquatwin.physics.hydraulics import solve
    closed = set(closed)
    pmin, preq, tol, it = (sim.pdd_pmin_m, sim.pdd_preq_m, sim.solver_tol, sim.solver_max_iter) if sim else (0.5, 12.0, 1e-6, 60)
    for _ in range(3):
        model, bmap = builder.build(demand_m3s, tank_levels, pump_speed, hw_c, leaks, closed, throttled, blockage)
        use_warm = warm is not None and warm[0] == bmap.signature
        res = solve(model, pmin, preq, tol, it, h_init=warm[1] if use_warm else None,
                    q_init=warm[2] if use_warm else None)
        extra = set()
        for k, t in enumerate(builder.tank_idx):
            for li in np.nonzero((model.start == t) | (model.end == t))[0]:
                pid = model.link_ids[li]
                q_in = res.flow[li] if model.end[li] == t else -res.flow[li]
                empty = tank_levels[k] <= tank_min[k] + tol_level and q_in < 0
                full = tank_levels[k] >= tank_max[k] - tol_level and q_in > 0
                if (empty or full) and pid not in closed:
                    extra.add(pid)
        if not extra:
            break
        closed |= extra
    return model, bmap, res, closed
