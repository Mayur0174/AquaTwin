# Simulation Design: Network, Scenarios, Truth and Reality

## 1. Water Network Generator (`aquatwin/network/generator.py`)

| Step | Rule (engineering rationale) |
|---|---|
| Street grid | 12 × 10 intersections at about 220 m with ±18% jitter. 10% of segments are dropped (keeping the graph connected) and 4% diagonal arterials are added. Pipes follow streets |
| Terrain | Smooth hill (+22 m) plus a gentle slope up from the river and low-amplitude undulation. Elevations are 545-575 m (Pune-like) |
| Land use | Commercial core (r < 0.18 W), industrial estate (SE corner), residential elsewhere. Base demand = population density × area × 135 LPCD (CPHEEO) |
| Sources | WTP clear-water reservoir → **pumping station** (pump curve `h = a − bQ²`, design point from lift to tank top plus friction) → network → two **elevated service reservoirs** (18 m staging) on the highest ground |
| Pipe sizing | Multi-source shortest-path supply tree. Each pipe carries its cumulative downstream peak demand (PF 2.5), plus the transmission paths PS → tanks carry refill flow. Diameter = smallest commercial size with v ≤ 1 m/s. Loop pipes take 35% of the neighbouring main |
| Installation year | City grows outward: `year ≈ 1958 + 70·(distance from core)`, ±5 y. 12% of pre-1985 pipes have been rehabilitated |
| Material by era | < 1970: cast iron / steel. 1970-90: CI, DI, steel, PVC. 1990-2005: DI, PVC, HDPE. > 2005: HDPE, PVC, DI. Mains ≥ 400 mm are steel or DI |
| DMAs | 2 × 2 District Metered Areas. Every boundary pipe carries a flow meter and valve |
| Condition history | Year-by-year Poisson break simulation from installation to 2026 using the hazard model. Each break adds a repair, which raises future hazard (Rule 6). Pipes with ≥ 6 repairs or corrosion > 0.85 are **renewed** (utility policy) |
| Calibration | One simulated day replaces the static pressure estimate with simulated mean pressure and *pressure cycles/day*, then stress, hazard, health, risk and RUL are recomputed |

Exports: `network.json` (NetworkX), `nodes.csv`, `pipes.csv`, `network.geojson`, and **`network.inp` (EPANET 2.2)**
for independent cross-validation in EPANET or WNTR.

### Pipe attributes

`pipe_id, start_node, end_node, length_m, diameter_mm, material, installation_year, age_years,
corrosion_index, repair_count, historical_leaks, health_score, failure_risk, break_rate, rul_years,
hw_c, soil_corrosivity, traffic_load, stress_index, pressure_cycles, strength_head_m, dma,
is_dma_boundary, has_valve, valve_id, rehabilitated, break_years, repair_years`

## 2. Hydraulic Truth Engine (`aquatwin/physics/hydraulics.py`, `aquatwin/truth/`)

**Solver:** Global Gradient Algorithm (Todini & Pilati, 1988), the EPANET 2 method. It is a Newton
iteration on heads and flows with sparse linear solves (SciPy). It typically converges in 6-8
iterations, about 5 ms per solve for ~125 nodes and ~200 pipes.

| Element | Equation |
|---|---|
| Pipe friction | Hazen-Williams `h = 10.67 L Q^1.852 / (C^1.852 D^4.871)`. C decays with age (tuberculation) |
| Minor losses (valves) | `h = K v² / 2g`. A stuck valve has K = 180 |
| Pump | `gain = a s² − b Q²` with check valve. Level control: ON if any tank < 35%, OFF if all > 92% |
| Tanks | Fixed head per step. Level integrated from net inflow. EPANET rule: an empty tank cannot drain and a full tank cannot fill (altitude valve) |
| Demand | Pressure-driven (Wagner): `d = D·√((p − 0.5)/(12 − 0.5))`, so bursts cause service deficits |
| Leaks | FAVAD emitter `q = Cd √(2g) (A₀ p^0.5 + m p^1.5)`. Metal pipes have m = 0. Plastic pipes have expanding openings (N1 ≈ 1.0-1.2). **The pipe is split at the leak position**, so the leak node's pressure is physically consistent |
| Transients | Joukowsky `ΔH = c Δv / g` for a pump trip or fast closure, decaying `exp(−d / 1500 m)`. Celerity is by material (PVC 420 m/s … CI 1200 m/s) |
| Acute failure | Stress-strength interference: `P = Φ((stress − strength)/(0.22·strength))`, where strength is Barlow `2σt/D` × residual-strength knockdown × (1 − corrosion)^1.2. Only the *excess* probability caused by events (surge, settlement, traffic) is used |
| Chronic hazard | Shamir-Howard `N(t) = N₀ e^{A·age}` × Cox covariates (corrosion, repairs, pressure, cycles, traffic, ground movement, diameter) |
| Isolation | Nodes with no path to a source are flagged `isolated` (zero supply) |
| Audit | Max nodal continuity error is recorded every step (≤ 1e-6 m³/s in the dataset) |

### Truth fields

`true_pressure, true_flow, true_velocity, true_water_loss (per pipe), true_leak_size (equivalent mm),
true_leak_location (pipe, fraction, metres from start node), true_pipe_health, true_corrosion,
failure_risk, served vs required demand, surge, tank levels, pump state`.

### The ten immutable physical truths and how each is guaranteed

| Rule | Mechanism | Automated test |
|---|---|---|
| 1 Larger leaks → larger losses | FAVAD flow increases monotonically in A₀ | `test_rule1_*` (6 sizes × 2 pipes) |
| 2 Larger leaks → stronger pressure anomaly | More outflow → more head loss in the network (GGA) | `test_rule2_*` |
| 3 Older → higher failure probability | Shamir-Howard exponential growth + corrosion growth | `test_rule3_*` (all materials) |
| 4 Corrosion → leakage | β_corr in hazard, larger opening multiplier, lower Barlow strength | `test_rule4_*` |
| 5 Closer sensors → stronger signature | Physics of head loss along paths | `test_rule5_*` (near > far, negative correlation with distance) |
| 6 Repairs → deterioration | β_repair in hazard, 0.92^repairs in health, self-exciting history | `test_rule6_*` |
| 7 Higher demand → lower pressure | Friction ∝ Q^1.852 | `test_rule7_*` |
| 8 Time-of-day demand | Diurnal pattern curves per land use + weekend curves | `test_rule8_*` |
| 9 Health ↓ with age and stress | Health model terms | `test_rule9_*` |
| 10 Combined failures interact | Shared hydraulics: high demand lowers leak flow, leaks superpose, isolation stops loss | `test_rule10_*` |
| Causality | Truth is independent of the sensor random stream | `test_causality_*` |
| Twin consistency | At L0, observed − expected = 0 exactly | `test_level0_twin_matches_truth` |

## 3. Scenario Engine (`aquatwin/scenarios/`)

An `Event(kind, start, end, target, params, cause)` changes *causes*, and `ScenarioEngine.conditions(t)`
aggregates the active events into a physical `Conditions` object. Demand events use a 1 h cosine taper.
Leak events carry growth laws. Emergent events (stress-induced bursts) are created by the truth engine
and logged with their causal explanation.

### Scenario rules

| Category | Scenario | Rules |
|---|---|---|
| Normal | Morning peak | Residential ×1.18 06-10 h; network-wide pressure drop; tanks drain |
| | Evening peak | Residential ×1.15 18-22 h |
| | Night low | All ×0.85 00-05 h; max pressure → max leak flow; MNF window |
| | Weekend | Weekend curves: residential peak +2 h later, commercial ×0.6, industrial ×0.55 |
| | Industrial | Industrial ×1.6 batch window; local DMA dip |
| Leak | Small | 3-6 mm × (1 + 1.5·corrosion); ~0.1-0.4 L/s; hazard-field location |
| | Medium | 8-14 mm; ~1-3 L/s |
| | Large | 18-30 mm; ~5-15 L/s |
| | Catastrophic burst | 40-90 mm (≤ 80% of D); tens of L/s; PDD service loss |
| | Progressive | Small, area +12%/h up to ×25; monotone growth of loss and anomaly |
| Sensor | Drift | Offset grows 0.6 m/day × 5; health partially reflects it |
| | Failure | NaN; health 0 |
| | Bias | ±1.5 m; **undetectable** by self-diagnostics |
| | Frozen | Repeats the last value; flat-line detector sets health 0.3 after 4 steps |
| Comms | Packet loss | +25% loss on top of the bursty channel |
| | Delay | Latency ×60; packets later than 600 s are missed |
| | Duplicates | 8% duplicates; ingestion de-duplicates by `seq` |
| | Missing readings | Gateway outage silences one DMA |
| Infrastructure | Pipe corrosion | Most-corroded main gets +0.15; perforation → weeping leak growing 6%/h |
| | Pipe aging | +10/15/20 years: C decays, corrosion deepens, risk rises |
| | Blockage | Old main loses 55% of its effective area |
| | Water hammer | Pump trip → Joukowsky surge; possible stress-strength bursts; tanks supply |
| | Valve malfunction | Valve stuck 85% closed (K = 180) |
| Environment | Heavy rainfall | 12-35 mm/h: residential −2%/(mm/h) (cap 12%), −4 °C, soil saturation → ground movement → hazard |
| | Ground settlement | 350 m radius, 90-160 m head-equivalent load → stress-strength failures |
| | Construction damage | Excavator strike on an arterial pipe → burst, regardless of age |
| | Traffic stress | Arterial mains +18 m cyclic load 07-21 h; risk and health degrade |
| Complex | Leak + high demand | +30% demand, heatwave; common-mode drop masks the leak |
| | Leak + sensor failure | The nearest logger to the leak is dead (Rule 5 stress test) |
| | Dual leaks | Two overlapping leaks; non-linear superposition |
| | Burst + valve shutdown | Isolation after 1-2.5 h (slow closure, no hammer); may cut supply |
| | Leak + corrosion + demand surge | Corroded main, +35% demand, growing leak |

### Realism levels

| Level | Adds |
|---|---|
| 0 Perfect | Exact sensors, perfect comms, nominal assets |
| 1 Sensor noise | Gaussian noise + quantisation (σ_p = 0.06 m, flow 1.5%, AMI 3%) |
| 2 Communication noise | Gilbert-Elliott loss, log-normal latency, duplicates, outages |
| 3 Infrastructure degradation | Background sensor faults (3%), calibration age, aging/blockage/valve/hammer scenarios, consumer AR(1) variability |
| 4 Compound events | Complex scenarios, 4% background faults |
| 5 Realistic urban | Environment scenarios, weekends, 6% background faults |

## 4. Sensor Reality Engine (`aquatwin/sensors/`)

* **Placement:** pressure loggers on 50% of junctions by greedy k-center (max-min network distance),
  seeded at the pump and tank connections. Level sensors on tanks. Flow meters on the pump, tank mains
  and all DMA boundaries, plus the largest mains (20%). AMI consumption at every demand node.
* **Instrument model:** `v = truth + N(0, σ) + thermal + fault`, quantised to resolution.
* **Faults:** drift (linear), bias (constant), frozen (stuck), failure (NaN or 0).
* **Sensor health (0-1):** device diagnostics + data quality. It drops with calibration age, drift
  magnitude, flat-lining and failure. **Bias stays at 1**, which reflects reality.
* **Communications:** two-state Gilbert-Elliott channel (p_gb = 0.02, p_bg = 0.25, loss 60% in the
  bad state), log-normal latency (median 3 s), a late cutoff at 600 s, and duplicates.
* **Raw telemetry:** every packet carries `seq` and `received_at`. `ingest.align_snapshot`
  de-duplicates and latency-gates packets. This is the same code path a real SCADA feed uses.

## 5. Twin Baseline (operator expectation)

`TwinBaseline` solves the same hydraulics with **only operator knowledge**: observed AMI demand
(gap-filled), observed tank levels, telemetered pump and altitude-valve states, and known operator
closures. It never sees leaks, blockages, malfunctions or hidden consumer variability. The residual
`observed − expected` is therefore physically meaningful. It is exactly 0 at L0 and about 0.065 m
(sensor noise) at L1 without leaks, while a large leak typically shows −2 m locally.
