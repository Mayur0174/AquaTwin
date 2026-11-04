"""WaterNetwork: the static asset model (graph + attribute tables + exporters)."""
from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

PIPE_COLUMNS = ["pipe_id", "start_node", "end_node", "length_m", "diameter_mm", "material",
                "installation_year", "age_years", "corrosion_index", "repair_count", "historical_leaks",
                "health_score", "failure_risk", "break_rate", "rul_years", "hw_c", "soil_corrosivity",
                "traffic_load", "stress_index", "pressure_cycles", "strength_head_m", "dma",
                "is_dma_boundary", "has_valve", "valve_id", "rehabilitated"]
NODE_COLUMNS = ["node_id", "node_type", "x", "y", "lat", "lon", "elevation", "base_demand_ls",
                "land_use", "population", "dma"]


class WaterNetwork:
    """Thin, cached wrapper around the NetworkX graph."""

    def __init__(self, graph: nx.Graph):
        self.graph = graph

    # ------------------------------------------------------------ tables
    @cached_property
    def nodes(self) -> pd.DataFrame:
        rows = []
        for n, d in self.graph.nodes(data=True):
            row = {c: d.get(c) for c in NODE_COLUMNS}
            row["node_id"] = n
            for extra in ("head", "init_level", "min_level", "max_level", "tank_diameter"):
                if extra in d:
                    row[extra] = d[extra]
            rows.append(row)
        return pd.DataFrame(rows).set_index("node_id", drop=False)

    @cached_property
    def pipes(self) -> pd.DataFrame:
        rows = [{c: d.get(c) for c in PIPE_COLUMNS}
                for _, _, d in self.graph.edges(data=True) if d.get("link_type") == "pipe"]
        df = pd.DataFrame(rows).sort_values("pipe_id")
        return df.set_index("pipe_id", drop=False)

    @cached_property
    def pumps(self) -> pd.DataFrame:
        rows = [dict(d) for _, _, d in self.graph.edges(data=True) if d.get("link_type") == "pump"]
        return pd.DataFrame(rows).set_index("pipe_id", drop=False)

    @property
    def junction_ids(self) -> list[str]:
        return self.nodes.index[self.nodes.node_type.isin(["junction", "pump_station"])].tolist()

    @property
    def tank_ids(self) -> list[str]:
        return self.nodes.index[self.nodes.node_type == "tank"].tolist()

    @property
    def pipe_ids(self) -> list[str]:
        return self.pipes.index.tolist()

    def invalidate(self) -> None:
        for attr in ("nodes", "pipes", "pumps", "distance_matrix", "node_index", "pipe_index", "_geom_arrays"):
            self.__dict__.pop(attr, None)

    @cached_property
    def node_index(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.nodes.index)}

    @cached_property
    def pipe_index(self) -> dict[str, int]:
        return {p: i for i, p in enumerate(self.pipes.index)}

    def edge_of(self, pipe_id: str) -> tuple[str, str]:
        r = self.pipes.loc[pipe_id]
        return r.start_node, r.end_node

    def set_pipe_attrs(self, pipe_id: str, **attrs) -> None:
        u, v = self.edge_of(pipe_id)
        self.graph.edges[u, v].update(attrs)
        self.invalidate()

    # ------------------------------------------------------------ geometry
    @cached_property
    def distance_matrix(self) -> pd.DataFrame:
        """All-pairs network (along-pipe) distance between nodes, metres."""
        g = nx.Graph()
        for _, r in self.pipes.iterrows():
            g.add_edge(r.start_node, r.end_node, w=r.length_m)
        d = dict(nx.all_pairs_dijkstra_path_length(g, weight="w"))
        ids = list(g.nodes)
        mat = np.array([[d[a].get(b, np.inf) for b in ids] for a in ids])
        return pd.DataFrame(mat, index=ids, columns=ids)

    @cached_property
    def _geom_arrays(self) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        D = self.distance_matrix
        pos = {n: i for i, n in enumerate(D.index)}
        p = self.pipes
        s = np.array([pos[n] for n in p.start_node])
        e = np.array([pos[n] for n in p.end_node])
        return {pid: i for i, pid in enumerate(p.index)}, s, e, p.length_m.to_numpy(float), D.to_numpy()

    def point_distance(self, pipe_a: str, frac_a: float, pipe_b: str, frac_b: float) -> float:
        """Network distance between two points located on pipes (fraction from start node)."""
        idx, s, e, L, D = self._geom_arrays
        ia, ib = idx[pipe_a], idx[pipe_b]
        if ia == ib:
            return float(abs(frac_a - frac_b) * L[ia])
        ends_a = ((s[ia], frac_a * L[ia]), (e[ia], (1 - frac_a) * L[ia]))
        ends_b = ((s[ib], frac_b * L[ib]), (e[ib], (1 - frac_b) * L[ib]))
        return float(min(da + D[na, nb] + db for na, da in ends_a for nb, db in ends_b))

    def pipe_midpoints(self) -> pd.DataFrame:
        n = self.nodes
        p = self.pipes
        return pd.DataFrame({
            "lat": 0.5 * (n.loc[p.start_node, "lat"].values + n.loc[p.end_node, "lat"].values),
            "lon": 0.5 * (n.loc[p.start_node, "lon"].values + n.loc[p.end_node, "lon"].values),
        }, index=p.index)

    def point_on_pipe(self, pipe_id: str, frac: float) -> tuple[float, float]:
        p = self.pipes.loc[pipe_id]
        a, b = self.nodes.loc[p.start_node], self.nodes.loc[p.end_node]
        return float(a.lat + frac * (b.lat - a.lat)), float(a.lon + frac * (b.lon - a.lon))

    # ------------------------------------------------------------ io
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.graph, edges="links")
        with open(d / "network.json", "w", encoding="utf-8") as fh:
            json.dump(data, fh, default=_json_default)
        self.nodes.to_csv(d / "nodes.csv", index=False)
        self.pipes.to_csv(d / "pipes.csv", index=False)
        with open(d / "network.geojson", "w", encoding="utf-8") as fh:
            json.dump(self.to_geojson(), fh, default=_json_default)
        self.to_inp(d / "network.inp")

    @classmethod
    def load(cls, directory: str | Path) -> "WaterNetwork":
        with open(Path(directory) / "network.json", "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(nx.node_link_graph(data, edges="links"))

    def to_geojson(self) -> dict:
        feats = []
        for _, n in self.nodes.iterrows():
            props = {k: _json_default(v) if isinstance(v, (np.generic,)) else v for k, v in n.items()}
            feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [n.lon, n.lat]},
                          "properties": props})
        for _, p in self.pipes.iterrows():
            a, b = self.nodes.loc[p.start_node], self.nodes.loc[p.end_node]
            props = {k: _json_default(v) if isinstance(v, (np.generic,)) else v for k, v in p.items()}
            feats.append({"type": "Feature", "geometry": {"type": "LineString",
                                                          "coordinates": [[a.lon, a.lat], [b.lon, b.lat]]},
                          "properties": props})
        return {"type": "FeatureCollection", "features": feats}

    def to_inp(self, path: str | Path) -> None:
        """Export an EPANET 2.2 .inp file so engineers can cross-validate in EPANET/WNTR."""
        n, p = self.nodes, self.pipes
        lines = ["[TITLE]", "AquaTwin AI generated network", "", "[JUNCTIONS]",
                 ";ID  Elev  Demand(LPS)  Pattern"]
        for _, r in n[n.node_type.isin(["junction", "pump_station"])].iterrows():
            pat = {"residential": "RES", "commercial": "COM", "industrial": "IND"}.get(r.land_use, "")
            lines.append(f" {r.node_id}  {r.elevation:.2f}  {r.base_demand_ls:.4f}  {pat}")
        lines += ["", "[RESERVOIRS]", ";ID  Head"]
        for _, r in n[n.node_type == "reservoir"].iterrows():
            lines.append(f" {r.node_id}  {r['head']:.2f}")
        lines += ["", "[TANKS]", ";ID  Elev  InitLvl  MinLvl  MaxLvl  Diam  MinVol"]
        for _, r in n[n.node_type == "tank"].iterrows():
            lines.append(f" {r.node_id}  {r.elevation:.2f}  {r.init_level}  {r.min_level}  {r.max_level}  {r.tank_diameter}  0")
        lines += ["", "[PIPES]", ";ID  Node1  Node2  Length  Diameter  Roughness  MinorLoss  Status"]
        for _, r in p.iterrows():
            lines.append(f" {r.pipe_id}  {r.start_node}  {r.end_node}  {r.length_m:.1f}  {r.diameter_mm:.0f}  {r.hw_c:.1f}  0  Open")
        lines += ["", "[PUMPS]", ";ID  Node1  Node2  Parameters"]
        for _, r in self.pumps.iterrows():
            lines.append(f" {r.pipe_id}  {r.start_node}  {r.end_node}  HEAD CURVE1")
        pump = self.pumps.iloc[0]
        lines += ["", "[CURVES]", f" CURVE1  {pump.design_flow_ls:.2f}  {pump.design_head_m:.2f}", "",
                  "[PATTERNS]"]
        from aquatwin.physics.demand import PATTERNS
        for code, lu in (("RES", "residential"), ("COM", "commercial"), ("IND", "industrial")):
            vals = PATTERNS[(lu, False)]
            for i in range(0, 24, 6):
                lines.append(f" {code}  " + "  ".join(f"{v:.3f}" for v in vals[i:i + 6]))
        lines += ["", "[OPTIONS]", " Units LPS", " Headloss H-W", "", "[TIMES]", " Duration 24:00",
                  " Hydraulic Timestep 0:15", " Pattern Timestep 1:00", "", "[END]"]
        Path(path).write_text("\n".join(lines), encoding="utf-8")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, tuple):
        return list(o)
    return o
