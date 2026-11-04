# Dataset: Schema and Generation Logic

Generate with `python -m aquatwin generate` (default config: 24 episodes × 24 h × 15 min × 201 pipes
= **463,104 pipe records**, about 5 minutes on a laptop CPU). Output goes to `data/generated/`.

## 1. Files

| File | Grain | Rows (default) | Purpose |
|---|---|---|---|
| `pipes_timeseries.parquet` / `.csv` | pipe × 15 min | 463k | **Main dataset** (spec fields + truth + labels) |
| `nodes_timeseries.parquet` | node × 15 min | 286k | Observed, expected and true pressure; AMI demand; surge; isolation |
| `system_timeseries.parquet` | 15 min | 2.3k | Pump, tanks, totals, solver audit (converged, mass-balance error) |
| `events.parquet` / `.csv` | event | ~90 | Scenario and emergent events with **causal explanation** |
| `telemetry_raw.parquet` | packet | 546k | Raw packets with seq, latency and duplicates (the real-sensor contract) |
| `localization_{nodes,pipes,labels}.parquet` | sample × step | 900 samples | Physics leak sweep for GNN training |
| `asset_history.parquet` | pipe × year | 4.2k | Predictive-maintenance panel |
| `network/` | | | `network.json`, `nodes.csv`, `pipes.csv`, `network.geojson`, `network.inp` |
| `sensors.csv`, `schema.json`, `manifest.json`, `config.yaml` | | | Registry, provenance, lineage, reproducibility |

## 2. Main schema: `pipes_timeseries`

Provenance classes are `observation` (SCADA/AMI, noisy, may be NaN), `twin` (expectation), `asset`
(GIS/CMMS), `environment`, `truth` (never a model input), `label`, and `meta`.

| Field | Source | Unit | Description |
|---|---|---|---|
| timestamp | meta | datetime | 15-min step |
| episode_id, level, scenario | meta | | Episode id, realism level 0-5, scenario composition |
| pipe_id, node_id, end_node, dma | asset | | Pipe, start junction (reference node), end junction, DMA |
| pressure, pressure_end | observation | m | Observed pressure at the start / end junction (NaN if unmonitored or missing) |
| expected_pressure(_end) | twin | m | Twin baseline |
| flow, expected_flow | observation / twin | L/s | Metered pipe flow (NaN if unmetered) / expectation |
| velocity | observation | m/s | flow / area |
| demand | observation | L/s | AMI consumption at node_id |
| pipe_age, material, diameter_mm, length_m | asset | | |
| corrosion | asset | 0-1 | Corrosion index from the condition assessment |
| repair_count | asset | count | |
| pipe_health | asset | 0-100 | Register condition score |
| sensor_health | observation | 0-1 | Worst diagnostic of the pipe's associated sensors |
| temperature, rainfall, ground_movement | environment | °C, mm/h, 0-1 | |
| packet_loss | observation | 0/1 | Any associated packet lost |
| delay_seconds | observation | s | Max latency of associated packets |
| true_pressure, true_flow, true_velocity | truth | | Hidden physics |
| true_pipe_health, true_corrosion | truth | | Dynamic condition |
| water_loss | label | L/s | True leak outflow on this pipe |
| leak_size | label | mm | Equivalent orifice diameter |
| leak_location | label | m | Distance from node_id along the pipe |
| leak_frac | label | 0-1 | Position fraction |
| is_leak | label | 0/1 | Active leak on this pipe |
| failure_risk | label | 1/yr | True annual failure probability under current stress |

The complete machine-readable version is `schema.json`, produced from `aquatwin/data/schema.py`.
The detector asserts at training time that no `truth` or `label` column is used as a feature.

## 3. Generation logic (pseudocode)

```text
network  = generate_network(cfg.network)            # engineered city, simulated asset history
calibrate_network(network)                          # 1 simulated day -> pressure cycles, stress, risk
sensors  = place_sensors(network)                   # k-center loggers, DMA meters, AMI

for e in range(n_episodes):
    level    = levels[e % 6]                        # levels cycle -> stratified test block
    names    = compose_episode(level, rng)          # e.g. night_low + comm_delay + large_leak
    events   = [SCENARIOS[n].builder(ctx) for n in names]   # causes only; hazard-field locations
    run      = SimulationRun(network, sensors, level, ScenarioEngine(events))
    for t in steps:
        cond     = scenario.conditions(t)           # CAUSE
        truth    = truth_engine.step(cond)          # PHYSICS (GGA + PDD + FAVAD + surge + failure)
        obs      = reality.observe(truth, cond)     # OBSERVATION (noise, faults, comms, packets)
        expected = twin_baseline(obs.AMI, obs.tank_levels, SCADA valve/pump states)
        rows    += RowBuilder(truth, obs, expected)
write parquet/csv + events (causes) + telemetry_raw (packets)

for s in range(n_localization_samples):             # GNN coverage sweep
    pipe   = 50% uniform (coverage) | 50% hazard-weighted (realism)
    level  = random 0-5; size category ~ {small .1, medium .3, large .35, burst .25}
    simulate 8 warm-up steps + 3 leak steps at a random time of day and tank state

asset_history = for each pipe, year in last 25 years:
    inputs  = age, material, repairs/leaks so far (from simulated break history),
              corrosion + N(0, 0.06) (inspection), pressure cycles ×(1 + N(0, 0.15)) (loggers)
    targets = hazard-model failure_risk and RUL using hidden exact soil, traffic, wall loss
```

## 4. Statistics of the default dataset

* 124 nodes (120 junctions, 1 reservoir, 1 pumping station, 2 elevated tanks), 201 pipes, 44 km.
* Sensors: 60 pressure loggers, 2 tank-level sensors, 65 flow meters, 120 AMI aggregates.
* Network break rate 0.20 breaks/km/yr (typical of Indian and European utilities, 0.1-0.5).
* Leak rows: about 1,300 (0.3% of rows, realistic class imbalance). The sweep adds about 2,700 leak snapshots.
* Solver convergence 100%, max continuity error 5e-7 m³/s.
