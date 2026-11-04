"""City water distribution network generator.

The generator builds a plausible *engineered* network rather than a random
graph.  Every attribute follows from a design or history rule:

1. Street grid with irregular blocks -> pipe routes (pipes follow streets).
2. Smooth terrain field -> node elevations.
3. City grows outward from a historic core -> installation year by distance.
4. Material by installation era and diameter (cast iron era -> DI/PVC -> HDPE).
5. Land use (industrial estate, commercial core, residential) -> base demand
   from population density x LPCD.
6. Hydraulic pipe sizing: shortest-path supply tree from sources, cumulative
   downstream peak demand, smallest commercial diameter with v <= 1 m/s.
7. Treatment-plant reservoir -> pumping station -> elevated service
   reservoirs (tanks) on high ground, like a typical Indian city scheme.
8. District Metered Areas (DMAs) with metered boundary pipes.
9. Asset condition from the deterioration model, including a *simulated*
   break/repair history (Rule 6: repairs are caused by past breaks which are
   themselves caused by age/corrosion/stress).

The output is a :class:`WaterNetwork` wrapping a NetworkX graph whose nodes
are junctions/reservoirs/pump stations/tanks and whose edges are pipelines
(plus the pump link).
"""
from __future__ import annotations

import numpy as np
import networkx as nx

from aquatwin.config import NetworkConfig
from aquatwin.network.materials import COMMERCIAL_DIAMETERS_MM
from aquatwin.network.model import WaterNetwork
from aquatwin.physics import degradation as deg

RENEWAL_REPAIRS = 6        # utility replaces a main after this many repairs
RENEWAL_CORROSION = 0.85   # ... or when inspection shows severe wall loss


def _terrain(x: np.ndarray, y: np.ndarray, w: float, h: float, cfg: NetworkConfig, rng) -> np.ndarray:
    hill_x, hill_y = 0.78 * w, 0.75 * h
    z = cfg.base_elevation_m
    z = z + cfg.hill_height_m * np.exp(-(((x - hill_x) / (0.38 * w)) ** 2 + ((y - hill_y) / (0.38 * h)) ** 2))
    z = z + 6.0 * (y / h)                               # gentle slope up from the river (south)
    phase = rng.uniform(0, 2 * np.pi, 2)
    z = z + 1.5 * np.sin(2 * np.pi * x / (0.6 * w) + phase[0]) * np.cos(2 * np.pi * y / (0.7 * h) + phase[1])
    return z


def _smooth_field(x, y, w, h, rng, n_blobs=6, lo=0.4, hi=1.6) -> np.ndarray:
    f = np.zeros_like(x, dtype=float)
    for _ in range(n_blobs):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        s = rng.uniform(0.15, 0.35) * max(w, h)
        f += rng.uniform(-1, 1) * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s ** 2))
    f = (f - f.min()) / max(f.max() - f.min(), 1e-9)
    return lo + (hi - lo) * f


def _choose_material(year: int, diameter: float, rng) -> str:
    if diameter >= 400:
        options = (["Cast Iron", "Steel"], [0.5, 0.5]) if year < 1975 else (["Steel", "Ductile Iron"], [0.4, 0.6])
    elif year < 1970:
        options = (["Cast Iron", "Steel"], [0.82, 0.18])
    elif year < 1990:
        options = (["Cast Iron", "Ductile Iron", "Steel", "PVC"], [0.30, 0.40, 0.18, 0.12])
    elif year < 2005:
        options = (["Ductile Iron", "PVC", "HDPE", "Steel"], [0.40, 0.42, 0.12, 0.06])
    else:
        options = (["HDPE", "PVC", "Ductile Iron"], [0.42, 0.28, 0.30])
    names, probs = options
    return str(rng.choice(names, p=probs))


def generate_network(cfg: NetworkConfig | None = None) -> WaterNetwork:
    cfg = cfg or NetworkConfig()
    rng = np.random.default_rng(cfg.seed)
    nx_, ny_, sp = cfg.grid_x, cfg.grid_y, cfg.spacing_m
    W, H = (nx_ - 1) * sp, (ny_ - 1) * sp
    G = nx.Graph(name="AquaTwin City", crs="EPSG:4326")

    # ------------------------------------------------------------ junctions
    ids = {}
    for i in range(nx_):
        for j in range(ny_):
            nid = f"J{i:02d}{j:02d}"
            ids[(i, j)] = nid
            x = i * sp + rng.uniform(-cfg.jitter, cfg.jitter) * sp
            y = j * sp + rng.uniform(-cfg.jitter, cfg.jitter) * sp
            G.add_node(nid, node_type="junction", x=float(x), y=float(y), gi=i, gj=j)

    # streets: 4-neighbour grid + a few diagonal arterials
    streets = []
    for i in range(nx_):
        for j in range(ny_):
            if i + 1 < nx_:
                streets.append((ids[(i, j)], ids[(i + 1, j)], "street"))
            if j + 1 < ny_:
                streets.append((ids[(i, j)], ids[(i, j + 1)], "street"))
    for i in range(nx_ - 1):
        for j in range(ny_ - 1):
            if rng.random() < cfg.diagonal_frac:
                streets.append((ids[(i, j)], ids[(i + 1, j + 1)], "arterial"))
    for u, v, kind in streets:
        G.add_edge(u, v, road=kind)
    # irregular blocks: drop non-bridge segments
    removable = [e for e in list(G.edges()) if G.edges[e]["road"] == "street"]
    rng.shuffle(removable)
    n_drop = int(cfg.edge_drop_frac * len(removable))
    dropped = 0
    for u, v in removable:
        if dropped >= n_drop:
            break
        G.remove_edge(u, v)
        if not nx.is_connected(G) or G.degree(u) < 1 or G.degree(v) < 1:
            G.add_edge(u, v, road="street")
        else:
            dropped += 1

    xs = np.array([G.nodes[n]["x"] for n in G.nodes])
    ys = np.array([G.nodes[n]["y"] for n in G.nodes])
    elev = _terrain(xs, ys, W, H, cfg, rng)
    for n, z in zip(list(G.nodes), elev):
        G.nodes[n]["elevation"] = float(z)

    # ------------------------------------------------------------ land use & demand
    core = np.array([0.38 * W, 0.42 * H])
    area_per_node_km2 = (sp / 1000.0) ** 2
    for n in list(G.nodes):
        d = G.nodes[n]
        pos = np.array([d["x"], d["y"]])
        dist_core = np.linalg.norm(pos - core) / max(W, H)
        if d["x"] > 0.72 * W and d["y"] < 0.32 * H:
            land = "industrial"
        elif dist_core < 0.18:
            land = "commercial"
        else:
            land = "residential"
        density = cfg.population_density_km2 * (1.25 if dist_core < 0.35 else 0.85)
        pop = density * area_per_node_km2
        if land == "residential":
            demand_ls = pop * cfg.lpcd / 86400.0
        elif land == "commercial":
            demand_ls = pop * 0.5 * cfg.lpcd / 86400.0 + 1.2
        else:
            demand_ls = 3.0 + 0.5 * pop * cfg.lpcd / 86400.0
        d.update(land_use=land, population=float(pop), base_demand_ls=float(demand_ls),
                 dist_core=float(dist_core))

    # DMA assignment by grid partition
    cx, cy = cfg.dma_split
    for n in list(G.nodes):
        d = G.nodes[n]
        di = min(cx - 1, d["gi"] * cx // nx_)
        dj = min(cy - 1, d["gj"] * cy // ny_)
        d["dma"] = f"DMA-{dj * cx + di + 1}"

    # ------------------------------------------------------------ sources
    # Water treatment plant reservoir south-west, outside the grid (near the river).
    sw = min(G.nodes, key=lambda n: G.nodes[n]["x"] + G.nodes[n]["y"])
    swd = G.nodes[sw]
    res_elev = float(min(elev) - 6.0)
    G.add_node("R1", node_type="reservoir", x=swd["x"] - 450.0, y=swd["y"] - 250.0, elevation=res_elev,
               head=res_elev + 4.0, land_use="none", population=0.0, base_demand_ls=0.0, dma="SOURCE",
               dist_core=1.0)
    G.add_node("PS1", node_type="pump_station", x=swd["x"] - 380.0, y=swd["y"] - 200.0,
               elevation=res_elev + 0.5, land_use="none", population=0.0, base_demand_ls=0.0,
               dma="SOURCE", dist_core=1.0)
    G.add_edge("R1", "PS1", road="plant")
    G.add_edge("PS1", sw, road="trunk")

    # Elevated service reservoirs on the highest ground, far apart.
    junctions = [n for n in G.nodes if G.nodes[n]["node_type"] == "junction"]
    by_elev = sorted(junctions, key=lambda n: -G.nodes[n]["elevation"])
    tank_sites = []
    for n in by_elev:
        p = np.array([G.nodes[n]["x"], G.nodes[n]["y"]])
        if all(np.linalg.norm(p - np.array([G.nodes[t]["x"], G.nodes[t]["y"]])) > 0.45 * max(W, H) for t in tank_sites):
            tank_sites.append(n)
        if len(tank_sites) >= cfg.n_tanks:
            break
    # a second tank should serve the low-lying west if hills are concentrated east
    tank_ids = []
    for k, site in enumerate(tank_sites, 1):
        sd = G.nodes[site]
        tid = f"T{k}"
        G.add_node(tid, node_type="tank", x=sd["x"] + 60.0, y=sd["y"] + 60.0,
                   elevation=sd["elevation"] + cfg.tank_staging_m, init_level=3.5, min_level=0.5,
                   max_level=6.0, tank_diameter=24.0, land_use="none", population=0.0,
                   base_demand_ls=0.0, dma=sd["dma"], dist_core=sd["dist_core"])
        G.add_edge(tid, site, road="tank_main")
        tank_ids.append(tid)

    # ------------------------------------------------------------ coordinates
    lat0 = cfg.center_lat
    for n in G.nodes:
        d = G.nodes[n]
        d["lat"] = float(lat0 + (d["y"] - H / 2) / 111320.0)
        d["lon"] = float(cfg.center_lon + (d["x"] - W / 2) / (111320.0 * np.cos(np.radians(lat0))))
        d["node_id"] = n

    # ------------------------------------------------------------ pipe sizing
    total_avg = sum(G.nodes[n]["base_demand_ls"] for n in G.nodes)
    for u, v in G.edges:
        a, b = G.nodes[u], G.nodes[v]
        G.edges[u, v]["length_m"] = float(np.hypot(a["x"] - b["x"], a["y"] - b["y"]))
    sources = ["PS1"] + tank_ids
    sub = G.copy()
    sub.remove_node("R1")
    _, paths = nx.multi_source_dijkstra(sub, sources, weight="length_m")
    design_flow = {}
    for n in junctions:
        path = paths[n]
        q = G.nodes[n]["base_demand_ls"] * cfg.peak_factor
        for a, b in zip(path[:-1], path[1:]):
            key = tuple(sorted((a, b)))
            design_flow[key] = design_flow.get(key, 0.0) + q
    # transmission mains: the pump must refill every tank, so the PS1 -> tank paths carry
    # the average demand drawn from each tank zone (otherwise tanks drain and never refill)
    zone_avg = {src: 0.0 for src in sources}
    for n in junctions:
        zone_avg[paths[n][0]] += G.nodes[n]["base_demand_ls"]
    for t in tank_ids:
        path = nx.shortest_path(sub, "PS1", t, weight="length_m")
        for a, b in zip(path[:-1], path[1:]):
            key = tuple(sorted((a, b)))
            design_flow[key] = design_flow.get(key, 0.0) + 1.3 * zone_avg[t]
    # pumping main carries the whole average demand + tank refill
    design_flow[tuple(sorted(("PS1", sw)))] = max(design_flow.get(tuple(sorted(("PS1", sw))), 0), 1.3 * total_avg * cfg.peak_factor / 1.6)

    def size(q_ls: float) -> float:
        for dmm in COMMERCIAL_DIAMETERS_MM:
            area = np.pi * (dmm / 1000.0) ** 2 / 4
            if q_ls / 1000.0 / area <= cfg.design_velocity_ms:
                return float(dmm)
        return float(COMMERCIAL_DIAMETERS_MM[-1])

    for u, v in G.edges:
        if {u, v} == {"R1", "PS1"}:
            continue
        key = tuple(sorted((u, v)))
        if key in design_flow:
            dmm = size(design_flow[key])
        else:  # loop-closing pipe: sized like the smaller neighbouring tree main
            nb = [design_flow.get(tuple(sorted((u, w))), 0) for w in G.neighbors(u)] + \
                 [design_flow.get(tuple(sorted((v, w))), 0) for w in G.neighbors(v)]
            dmm = max(100.0, size(0.35 * min([x for x in nb if x > 0] or [5.0])))
        G.edges[u, v]["diameter_mm"] = dmm

    # ------------------------------------------------------------ history, material, condition
    soil = _smooth_field(np.array([G.nodes[n]["x"] for n in G.nodes]),
                         np.array([G.nodes[n]["y"] for n in G.nodes]), W, H, rng)
    soil_map = dict(zip(G.nodes, soil))
    tank_head_est = max(G.nodes[t]["elevation"] + G.nodes[t]["init_level"] for t in tank_ids) if tank_ids else res_elev + 50

    k = 0
    for u, v in sorted(G.edges, key=lambda e: tuple(sorted(e))):
        e = G.edges[u, v]
        if {u, v} == {"R1", "PS1"}:
            e.update(link_type="pump", pipe_id="PU1", start_node="R1", end_node="PS1", length_m=10.0,
                     diameter_mm=600.0)
            continue
        k += 1
        a, b = G.nodes[u], G.nodes[v]
        # orient pipes from the higher-head side (source side) for readability
        start, end = (u, v) if len(paths[u]) <= len(paths[v]) else (v, u)
        mid_core = 0.5 * (a["dist_core"] + b["dist_core"])
        base_year = 1958 + 70 * min(1.0, mid_core / 0.75)       # city grows outward
        year = int(np.clip(base_year + rng.normal(0, 5), 1950, cfg.current_year - 1))
        if e.get("road") in ("trunk", "tank_main"):
            year = int(rng.integers(1985, 2012))
        rehab = year < 1985 and rng.random() < 0.12              # rehabilitation programme
        if rehab:
            year = int(rng.integers(2005, 2022))
        material = _choose_material(year, e["diameter_mm"], rng)
        age = cfg.current_year - year
        soil_c = 0.5 * (soil_map[u] + soil_map[v])
        major_road = e.get("road") == "arterial" or (a.get("gi", -1) == b.get("gi", -2) and a.get("gi", 0) % 4 == 0) \
            or (a.get("gj", -1) == b.get("gj", -2) and a.get("gj", 0) % 4 == 0)
        traffic = float(np.clip((0.85 if major_road else 0.25) + rng.normal(0, 0.08), 0, 1))
        p_est = float(tank_head_est - 0.5 * (a["elevation"] + b["elevation"]))
        dma_u, dma_v = a.get("dma"), b.get("dma")
        boundary = dma_u != dma_v and "SOURCE" not in (dma_u, dma_v)
        has_valve = boundary or rng.random() < cfg.valve_fraction or e.get("road") in ("trunk", "tank_main")
        e.update(link_type="pipe", pipe_id=f"P{k:03d}", start_node=start, end_node=end,
                 installation_year=year, age_years=float(age), material=material,
                 soil_corrosivity=float(soil_c), traffic_load=traffic, est_pressure_m=p_est,
                 dma=a.get("dma") if a.get("dma") != "SOURCE" else b.get("dma"),
                 is_dma_boundary=bool(boundary), has_valve=bool(has_valve),
                 valve_id=f"V{k:03d}" if has_valve else "", rehabilitated=bool(rehab))

    pipes = [(u, v) for u, v in G.edges if G.edges[u, v]["link_type"] == "pipe"]
    mats = np.array([G.edges[e]["material"] for e in pipes], dtype=object)
    dia = np.array([G.edges[e]["diameter_mm"] for e in pipes])
    age = np.array([G.edges[e]["age_years"] for e in pipes])
    soil_c = np.array([G.edges[e]["soil_corrosivity"] for e in pipes])
    traffic = np.array([G.edges[e]["traffic_load"] for e in pipes])
    p_est = np.array([G.edges[e]["est_pressure_m"] for e in pipes])
    length = np.array([G.edges[e]["length_m"] for e in pipes])
    cycles = np.full(len(pipes), 6.0)  # refined after simulation (see dataset generator)

    # simulated maintenance history: yearly breaks from the hazard model (self-exciting)
    install = (cfg.current_year - age).astype(int)
    rehab_flag = np.array([G.edges[e]["rehabilitated"] for e in pipes])
    repairs = np.zeros(len(pipes))
    leaks = np.zeros(len(pipes))
    break_years: list[list[int]] = [[] for _ in pipes]
    repair_years: list[list[int]] = [[] for _ in pipes]
    for year in range(int(install.min()), cfg.current_year):
        alive = install <= year
        a_y = np.maximum(year - install, 0)
        ci_y = deg.corrosion_index(mats, dia, a_y, soil_c)
        rate = deg.break_rate_per_km_yr(mats, a_y, dia, ci_y, repairs, p_est, cycles, traffic, 0.0)
        n_breaks = rng.poisson(rate * length / 1000.0) * alive
        leaks += n_breaks
        repairs += n_breaks
        # occasional preventive repairs on visibly corroded pipes (inspection-driven)
        prev = alive & (ci_y > 0.55) & (rng.random(len(pipes)) < 0.03)
        repairs += prev.astype(float)
        for idx in np.nonzero(n_breaks)[0]:
            break_years[idx].extend([year] * int(n_breaks[idx]))
        for idx in np.nonzero(prev)[0]:
            repair_years[idx].append(year)
        # renewal policy: pipes with chronic breaks or near-perforated walls are replaced
        renew = alive & ((repairs >= RENEWAL_REPAIRS) | (ci_y > RENEWAL_CORROSION))
        for idx in np.nonzero(renew)[0]:
            install[idx] = year + 1
            mats[idx] = _choose_material(year + 1, dia[idx], rng)
            repairs[idx] = 0
            leaks[idx] = 0
            break_years[idx] = []
            repair_years[idx] = []
            rehab_flag[idx] = True
    age = np.maximum(cfg.current_year - install, 1).astype(float)

    ci = deg.corrosion_index(mats, dia, age, soil_c)
    stress = deg.stress_index(p_est, cycles, traffic)
    rate = deg.break_rate_per_km_yr(mats, age, dia, ci, repairs, p_est, cycles, traffic, 0.0)
    health = deg.health_score(mats, age, ci, repairs, stress)
    risk = deg.annual_failure_probability(rate, length)
    rul = deg.remaining_useful_life(mats, dia, age, soil_c, repairs, stress)
    strength = deg.strength_head_m(mats, dia, ci)
    hwc = deg.hazen_williams_c(mats, age)
    for idx, e in enumerate(pipes):
        G.edges[e].update(material=str(mats[idx]), installation_year=int(install[idx]),
                          age_years=float(age[idx]), rehabilitated=bool(rehab_flag[idx]),
                          break_years=[int(y) for y in break_years[idx]],
                          repair_years=[int(y) for y in repair_years[idx]],
                          corrosion_index=float(ci[idx]), repair_count=int(repairs[idx]),
                          historical_leaks=int(leaks[idx]), health_score=float(health[idx]),
                          failure_risk=float(risk[idx]), break_rate=float(rate[idx]),
                          rul_years=float(rul[idx]), strength_head_m=float(strength[idx]),
                          hw_c=float(hwc[idx]), stress_index=float(stress[idx]),
                          pressure_cycles=float(cycles[idx]))

    # pump design point: lift to the tank top + friction allowance
    lift = tank_head_est - G.nodes["R1"]["head"] + 12.0
    q_design = 1.15 * total_avg / 1000.0 * 1.25
    a_ = 4.0 / 3.0 * lift
    b_ = (a_ - lift) / q_design ** 2
    G.edges["R1", "PS1"].update(pump_a=float(a_), pump_b=float(b_), design_flow_ls=float(q_design * 1000),
                                design_head_m=float(lift))
    G.graph.update(width_m=W, height_m=H, total_avg_demand_ls=total_avg, tanks=tank_ids,
                   current_year=cfg.current_year, seed=cfg.seed)
    return WaterNetwork(G)
