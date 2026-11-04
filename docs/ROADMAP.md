# Development Roadmap

## Hackathon build (48-72 h), done in this repository

| Phase | Hours | Deliverable | Owner role |
|---|---|---|---|
| 0 | 0-4 | Architecture, data contracts, physical-truth rules as tests | Water Resources Eng. + Smart City Consultant |
| 1 | 4-14 | Network generator, GGA solver (PDD, FAVAD, pumps, tanks), truth engine | Hydraulic Modelling Expert |
| 2 | 14-22 | Scenario engine (34 scenarios, 6 levels), reality engine, ingestion | Hydraulic Expert + MLOps |
| 3 | 22-30 | Dataset generator, schema, sweep, asset panel | AI Research Scientist |
| 4 | 30-42 | Feature engineering, XGBoost detector, failure model, SHAP | AI Research Scientist |
| 5 | 34-46 | GNN localisation (GCN/SAGE/GAT) + sensitivity baseline | GNN Specialist |
| 6 | 40-56 | Streamlit/Folium dashboard, control panel, FastAPI | GIS Architect |
| 7 | 50-64 | Evaluation framework, Docker, registry, reports | MLOps Engineer |
| 8 | 64-72 | Pitch deck, demo script, rehearsal | Smart City Consultant |

## Post-hackathon

| Horizon | Milestone | Detail |
|---|---|---|
| Month 1 | **Pilot data onboarding** | Import a real DMA from the utility GIS (Shapefile/GeoPackage → `WaterNetwork`), calibrate C-factors and demand against 2 weeks of SCADA, replace the generator but keep all services |
| Month 2 | Live telemetry | MQTT/SCADA adapters to `/ingest/telemetry`, TimescaleDB store, data-quality alerts |
| Month 3 | Sim-to-real | Fine-tune the detector/GNN on confirmed leaks from work orders. Simulation stays the augmentation source. Active sensor-placement recommendations from the sensitivity matrix |
| Month 4 | Acoustic fusion | Add noise-logger / hydrophone correlation as node features (sub-metre pin-pointing) |
| Month 6 | Operations integration | Work-order push (SAP PM / Maximo), crew dispatch, NRW (IWA water balance) monthly reports |
| Month 9 | Pressure management | Optimise PRV set-points with the twin: FAVAD gives about 1.1% leakage reduction per 1% pressure reduction |
| Year 1 | City scale | Pressure-zone decomposition, distributed solving, multi-DMA GNN (inductive GraphSAGE), capital-plan optimiser using risk × consequence |
| Year 2 | Platform | Multi-utility SaaS, transient (MOC) solver module, digital-twin standards (OGC CityGML Utility Network ADE) |

## Known limitations (and how the design absorbs them)

* **Steady-state hydraulics per 15-min step.** Transients are represented by Joukowsky envelopes, not
  method-of-characteristics. A MOC module can replace `surge` without changing interfaces.
* **Synthetic city.** All parameters are literature-based. Replacing the generator with a real GIS
  import is the first pilot task.
* **Small leaks (< 1 L/s)** are near the physical detection floor of pressure/flow sensing at 50%
  logger coverage. This is realistic, and it is why acoustic fusion is on the roadmap. The evaluation
  reports recall by leak size honestly.
