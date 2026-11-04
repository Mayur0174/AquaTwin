"""Steady-state hydraulic solver (Global Gradient Algorithm, Todini & Pilati 1988).

This is the same numerical method used by EPANET 2.  It solves, for every
time step, the coupled non-linear system

    energy:      H_i - H_j = h_k(Q_k)                     for every link k=(i,j)
    continuity:  sum(Q_in) - sum(Q_out) = d_i(p_i) + e_i(p_i)  for every junction i

with

* Hazen-Williams friction ``h = 10.67 L Q^1.852 / (C^1.852 D^4.871)`` (SI)
* minor losses ``h = K v^2 / 2g`` (valve throttling, partially closed valves)
* pump head curves ``gain = a s^2 - b Q^2`` with check-valve behaviour
* pressure-driven demand (Wagner et al. 1988): ``d = D * sqrt((p-pmin)/(preq-pmin))``
* FAVAD leak emitters (May 1994): ``e = Cd sqrt(2g) (A0 p^0.5 + m p^1.5)``

Leaks are therefore never "added" to observations: a leak is a physical
orifice whose outflow depends on the local pressure, which in turn depends
on the whole network.  Larger leak areas produce larger losses and larger
pressure drops (Rules 1 & 2) as a mathematical consequence, and higher
demand lowers pressure through friction (Rule 7).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

from aquatwin.config import GRAVITY

PIPE, PUMP = 0, 1
CLOSED_RESISTANCE = 1e8
Q_LINEAR = 1e-4          # m^3/s, below this friction is linearised (numerical regularisation)
SQRT_2G = np.sqrt(2 * GRAVITY)


@dataclass
class HydraulicModel:
    node_ids: list[str]
    elevation: np.ndarray
    fixed: np.ndarray                 # bool: reservoirs and tanks (known head)
    fixed_head: np.ndarray            # head for fixed nodes
    demand: np.ndarray                # required demand, m^3/s
    link_ids: list[str]
    start: np.ndarray
    end: np.ndarray
    kind: np.ndarray                  # PIPE / PUMP
    length: np.ndarray                # m
    diameter: np.ndarray              # m
    hw_c: np.ndarray
    minor_k: np.ndarray
    is_open: np.ndarray               # bool
    pump_a: np.ndarray                # shutoff head (m) at speed 1
    pump_b: np.ndarray                # curve coefficient (m / (m^3/s)^2)
    pump_speed: np.ndarray            # relative speed (0 = off)
    emitter_node: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    emitter_a0: np.ndarray = field(default_factory=lambda: np.zeros(0))       # m^2
    emitter_slope: np.ndarray = field(default_factory=lambda: np.zeros(0))    # m^2 per m head
    emitter_cd: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_links(self) -> int:
        return len(self.link_ids)


@dataclass
class HydraulicResult:
    head: np.ndarray
    pressure: np.ndarray
    flow: np.ndarray                  # m^3/s, positive start->end
    velocity: np.ndarray              # m/s (signed)
    demand_served: np.ndarray         # m^3/s
    emitter_flow: np.ndarray          # m^3/s per emitter
    isolated: np.ndarray              # bool per node (no path to a source)
    converged: bool
    iterations: int
    mass_balance_error: float         # max nodal continuity error, m^3/s


def _pdd_factor(p: np.ndarray, pmin: float, preq: float) -> tuple[np.ndarray, np.ndarray]:
    rng = preq - pmin
    x = np.clip((p - pmin) / rng, 0.0, 1.0)
    s = np.sqrt(x)
    xr = np.clip((p - pmin) / rng, 0.02, 1.0)
    ds = np.where((p - pmin) / rng < 1.0, 0.5 / rng / np.sqrt(xr), 0.0)
    return s, ds


def _emitter(p: np.ndarray, a0: np.ndarray, m: np.ndarray, cd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pp = np.maximum(p, 0.0)
    e = cd * SQRT_2G * (a0 * np.sqrt(pp) + m * pp ** 1.5)
    pr = np.maximum(p, 0.25)
    de = cd * SQRT_2G * (0.5 * a0 / np.sqrt(pr) + 1.5 * m * np.sqrt(pr))
    return e, de


def solve(model: HydraulicModel, pmin: float = 0.5, preq: float = 12.0, tol: float = 1e-6,
          max_iter: int = 60, h_init: np.ndarray | None = None, q_init: np.ndarray | None = None
          ) -> HydraulicResult:
    n, nl = model.n_nodes, model.n_links
    s_idx, e_idx = model.start, model.end
    area = np.pi * model.diameter ** 2 / 4.0

    # ---- topology: nodes cut off from every source are "isolated" (no supply)
    link_open = model.is_open & ~((model.kind == PUMP) & (model.pump_speed <= 0))
    adj = coo_matrix((np.ones(link_open.sum()), (s_idx[link_open], e_idx[link_open])), shape=(n, n))
    _, labels = connected_components(adj, directed=False)
    has_source = np.zeros(labels.max() + 1, bool)
    has_source[np.unique(labels[model.fixed])] = True
    isolated = ~has_source[labels]

    fixed = model.fixed | isolated
    h_fixed = np.where(model.fixed, model.fixed_head, model.elevation)
    demand_req = np.where(isolated, 0.0, model.demand)
    em_active = ~isolated[model.emitter_node] if len(model.emitter_node) else np.zeros(0, bool)

    unknown = ~fixed
    umap = -np.ones(n, int)
    umap[unknown] = np.arange(unknown.sum())
    nu = int(unknown.sum())

    # ---- initial guess
    if h_init is not None and len(h_init) == n:
        H = np.where(fixed, h_fixed, h_init)
    else:
        top = h_fixed[model.fixed].max() if model.fixed.any() else model.elevation.max() + 30
        H = np.where(fixed, h_fixed, top - 2.0)
    if q_init is not None and len(q_init) == nl:
        Q = q_init.copy()
    else:
        Q = 0.3 * area
        pumps = model.kind == PUMP
        Q[pumps] = np.sqrt(np.maximum(model.pump_a[pumps], 1e-6) / (4 * np.maximum(model.pump_b[pumps], 1e-9)))

    # pipe resistances
    r_hw = 10.67 * model.length / (np.power(model.hw_c, 1.852) * np.power(model.diameter, 4.871))
    r_m = model.minor_k / (2 * GRAVITY * area ** 2)
    is_pipe = model.kind == PIPE
    pump_closed = (model.kind == PUMP) & (model.pump_speed <= 0)

    converged, it = False, 0
    relax = 1.0
    prev_err = np.inf
    for it in range(1, max_iter + 1):
        aq = np.abs(Q)
        aql = np.maximum(aq, Q_LINEAR)
        # friction + minor loss, linear below Q_LINEAR for numerical stability
        h_pipe = np.where(aq > Q_LINEAR,
                          r_hw * Q * aql ** 0.852 + r_m * Q * aql,
                          (r_hw * Q_LINEAR ** 0.852 + r_m * Q_LINEAR) * Q)
        g_pipe = np.where(aq > Q_LINEAR,
                          1.852 * r_hw * aql ** 0.852 + 2 * r_m * aql,
                          r_hw * Q_LINEAR ** 0.852 + r_m * Q_LINEAR)
        s2 = model.pump_speed ** 2
        h_pump = model.pump_b * Q * aq - model.pump_a * s2
        g_pump = 2 * model.pump_b * np.maximum(aq, 1e-3)
        h = np.where(is_pipe, h_pipe, h_pump)
        g = np.where(is_pipe, g_pipe, g_pump)
        closed = (~model.is_open) | pump_closed
        h = np.where(closed, CLOSED_RESISTANCE * Q, h)
        g = np.where(closed, CLOSED_RESISTANCE, g)
        g = np.maximum(g, 1e-8)
        p = 1.0 / g
        y = p * h

        ui, uj = umap[s_idx], umap[e_idx]
        mi, mj = ui >= 0, uj >= 0
        both = mi & mj
        rows = np.concatenate([ui[mi], uj[mj], ui[both], uj[both]])
        cols = np.concatenate([ui[mi], uj[mj], uj[both], ui[both]])
        vals = np.concatenate([p[mi], p[mj], -p[both], -p[both]])
        F = np.zeros(nu)
        qy = Q - y
        np.add.at(F, ui[mi], -qy[mi])
        np.add.at(F, uj[mj], qy[mj])
        fi = mi & ~mj
        np.add.at(F, ui[fi], p[fi] * H[e_idx[fi]])
        fj = mj & ~mi
        np.add.at(F, uj[fj], p[fj] * H[s_idx[fj]])

        # pressure-driven demand + leak emitters (linearised about current head)
        pres = H - model.elevation
        sfac, dsfac = _pdd_factor(pres, pmin, preq)
        d_val = demand_req * sfac
        d_der = demand_req * dsfac
        diag = np.zeros(n)
        rhs = np.zeros(n)
        diag += d_der
        rhs -= d_val - d_der * H
        if len(model.emitter_node):
            ep = pres[model.emitter_node]
            ev, ed = _emitter(ep, model.emitter_a0, model.emitter_slope, model.emitter_cd)
            ev, ed = ev * em_active, ed * em_active
            np.add.at(diag, model.emitter_node, ed)
            np.add.at(rhs, model.emitter_node, -(ev - ed * H[model.emitter_node]))
        rows = np.concatenate([rows, np.arange(nu)])
        cols = np.concatenate([cols, np.arange(nu)])
        vals = np.concatenate([vals, diag[unknown] + 1e-12])
        F += rhs[unknown]

        A = coo_matrix((vals, (rows, cols)), shape=(nu, nu)).tocsc()
        H_new_u = spsolve(A, F)
        H_new = H.copy()
        H_new[unknown] = H_new_u
        Q_new = Q - y + p * (H_new[s_idx] - H_new[e_idx])
        Q_new = np.where(closed, 0.0, Q_new)

        dq = np.abs(Q_new - Q).sum() / max(np.abs(Q_new).sum(), 1e-9)
        if dq > prev_err and it > 8:
            relax = max(0.3, relax * 0.7)
        prev_err = dq
        Q = Q + relax * (Q_new - Q)
        H = H + relax * (H_new - H) if relax < 1.0 else H_new

        # pump check valves: close if required lift exceeds shutoff head, reopen otherwise
        status_changed = False
        pumps_idx = np.nonzero(model.kind == PUMP)[0]
        for k in pumps_idx:
            if model.pump_speed[k] <= 0:
                continue
            lift = H[e_idx[k]] - H[s_idx[k]]
            should_close = lift > model.pump_a[k] * model.pump_speed[k] ** 2 or Q[k] < 0
            if should_close != pump_closed[k]:
                pump_closed[k] = should_close
                status_changed = True
        if dq < tol and not status_changed:
            converged = True
            break

    pres = H - model.elevation
    sfac, _ = _pdd_factor(pres, pmin, preq)
    served = demand_req * sfac
    if len(model.emitter_node):
        ev, _ = _emitter(pres[model.emitter_node], model.emitter_a0, model.emitter_slope, model.emitter_cd)
        ev = ev * em_active
    else:
        ev = np.zeros(0)
    # mass balance audit (invariant)
    bal = np.zeros(n)
    np.add.at(bal, e_idx, Q)
    np.add.at(bal, s_idx, -Q)
    bal -= served
    if len(ev):
        np.add.at(bal, model.emitter_node, -ev)
    mbe = float(np.abs(bal[unknown]).max()) if nu else 0.0
    return HydraulicResult(head=H, pressure=pres, flow=Q, velocity=Q / area, demand_served=served,
                           emitter_flow=ev, isolated=isolated, converged=converged, iterations=it,
                           mass_balance_error=mbe)
