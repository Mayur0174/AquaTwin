"""Sensor registry and placement.

Placement is deterministic engineering, not chance:

* pressure loggers: tank levels, pump discharge, then a greedy *max-min
  network distance* (k-center) placement over junctions, which is the
  classical coverage objective used for SCADA/pressure-logger planning;
* flow meters: pump discharge, every tank main, every DMA boundary pipe
  (a DMA is only a DMA if its inflows are metered) plus the largest mains
  until the requested coverage is reached;
* AMI: aggregated customer-meter consumption per junction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aquatwin.config import SensorConfig
from aquatwin.network.model import WaterNetwork


@dataclass
class SensorRegistry:
    table: pd.DataFrame        # sensor_id, kind, target, noise, resolution, dma

    @property
    def pressure(self) -> pd.DataFrame:
        return self.table[self.table.kind == "pressure"]

    @property
    def flow(self) -> pd.DataFrame:
        return self.table[self.table.kind == "flow"]

    @property
    def ami(self) -> pd.DataFrame:
        return self.table[self.table.kind == "ami"]

    @property
    def monitored_nodes(self) -> list[str]:
        return self.pressure.target.tolist()

    @property
    def metered_pipes(self) -> list[str]:
        return self.flow.target.tolist()

    def sensor_for(self, kind: str, target: str) -> str | None:
        m = self.table[(self.table.kind == kind) & (self.table.target == target)]
        return None if m.empty else str(m.sensor_id.iloc[0])

    def nearest_pressure_sensor(self, network: WaterNetwork, pipe_id: str) -> str:
        u, v = network.edge_of(pipe_id)
        D = network.distance_matrix
        best = min(self.monitored_nodes, key=lambda n: min(D.at[u, n], D.at[v, n]))
        return self.sensor_for("pressure", best)


def place_sensors(network: WaterNetwork, cfg: SensorConfig) -> SensorRegistry:
    nodes, pipes = network.nodes, network.pipes
    D = network.distance_matrix
    junctions = [n for n in nodes.index[nodes.node_type == "junction"] if n in D.index]
    n_target = max(3, int(round(cfg.pressure_coverage * len(junctions))))

    # seed: node adjacent to the pumping main and the tank connection points
    seeds = []
    for _, p in pipes.iterrows():
        for a, b in ((p.start_node, p.end_node), (p.end_node, p.start_node)):
            if nodes.at[a, "node_type"] in ("pump_station", "tank") and b in junctions and b not in seeds:
                seeds.append(b)
    chosen = list(seeds)
    sub = D.loc[junctions, junctions].to_numpy()
    idx = {n: i for i, n in enumerate(junctions)}
    while len(chosen) < n_target:
        dmin = sub[:, [idx[c] for c in chosen]].min(axis=1)
        nxt = junctions[int(np.argmax(dmin))]
        if nxt in chosen:
            break
        chosen.append(nxt)

    rows = []
    for n in chosen:
        rows.append(dict(sensor_id=f"PS-{n}", kind="pressure", target=n, noise=cfg.pressure_noise_m,
                         resolution=cfg.pressure_resolution_m, dma=nodes.at[n, "dma"]))
    for t in network.tank_ids:
        rows.append(dict(sensor_id=f"LV-{t}", kind="level", target=t, noise=0.02, resolution=0.01,
                         dma=nodes.at[t, "dma"]))

    mandatory = set(pipes.index[pipes.is_dma_boundary])
    for _, p in pipes.iterrows():
        if nodes.at[p.start_node, "node_type"] in ("pump_station", "tank") or \
                nodes.at[p.end_node, "node_type"] in ("pump_station", "tank"):
            mandatory.add(p.pipe_id)
    n_flow = max(len(mandatory), int(round(cfg.flow_coverage * len(pipes))) + len(mandatory))
    extra = [pid for pid in pipes.sort_values("diameter_mm", ascending=False).index if pid not in mandatory]
    metered = list(mandatory) + extra[: max(0, n_flow - len(mandatory))]
    for pid in sorted(metered):
        rows.append(dict(sensor_id=f"FM-{pid}", kind="flow", target=pid, noise=cfg.flow_noise_frac,
                         resolution=cfg.flow_resolution_ls, dma=pipes.at[pid, "dma"]))
    for pu in network.pumps.index:
        rows.append(dict(sensor_id=f"FM-{pu}", kind="flow", target=pu, noise=cfg.flow_noise_frac,
                         resolution=cfg.flow_resolution_ls, dma="SOURCE"))
    for n in junctions:
        if nodes.at[n, "base_demand_ls"] > 0:
            rows.append(dict(sensor_id=f"AMI-{n}", kind="ami", target=n, noise=cfg.ami_noise_frac,
                             resolution=0.001, dma=nodes.at[n, "dma"]))
    return SensorRegistry(pd.DataFrame(rows).set_index("sensor_id", drop=False))
