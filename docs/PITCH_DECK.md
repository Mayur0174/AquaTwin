# Hackathon Pitch Deck Outline (12 slides · 7 minutes + 3-minute live demo)

| # | Slide | Message (one sentence) | Content / visual | Time |
|---|---|---|---|---|
| 1 | **Title**: AquaTwin AI | "A physics-true digital twin that finds leaks before the street floods." | Logo, team of 7 roles, the live map screenshot | 0:15 |
| 2 | **The problem** | Indian cities lose 30-40% of treated water as Non-Revenue Water; most leaks run unseen underground for weeks. | NRW statistics (CPHEEO / IWA), cost per ML, burst photo | 0:40 |
| 3 | **Why current tools fail** | Utilities have sparse, noisy, lossy sensors and no labelled leak data, so ML cannot be trained or trusted. | Three-icon slide: no labels · noisy data · no explanations | 0:30 |
| 4 | **Our idea** | Build the city's water network twice: a physics *truth* and an *observed* world. Train AI where the answer is known, then deploy it where it is not. | CAUSE → PHYSICS → OBSERVATION diagram | 0:40 |
| 5 | **Physics you can audit** | EPANET-grade GGA solver, pressure-driven demand, FAVAD leaks, Joukowsky surges, stress-strength failures, and 10 immutable physical truths enforced as automated tests. | Equations strip + green test badge "50 passed" | 0:45 |
| 6 | **Reality, not a toy** | 33 scenarios, 6 realism levels: drift, bias, frozen sensors, bursty packet loss, rain, settlement, construction, compound failures. | Scenario wheel + level ladder L0→L5 | 0:40 |
| 7 | **The AI stack** | XGBoost detects, a physics-informed GNN localises to pipe and metre, XGBoost predicts failure and remaining life, SHAP explains every decision. | Architecture diagram; GNN schematic | 0:45 |
| 8 | **Results** | 0 missed leak episodes, 92% alarm precision; leaks localised to about 125-150 m median (one city block), with a 32-58% lower mean error than the classical method at L1-L5; failure-risk R² 0.76 on an unseen city. | `reports/evaluation_levels.md` charts: F1 vs level, error GNN vs classical | 0:50 |
| 9 | **Live demo** | Inject a burst during the evening peak, watch the alarm, SHAP reasons, GNN pin on the map, DMA loss estimate, then isolate the pipe. | Dashboard, live | 3:00 |
| 10 | **Real-world ready** | Real sensors plug into the same telemetry contract; the GIS replaces the generator; EPANET `.inp` export for engineers. | Deployment diagram (MQTT/SCADA/AMI → ingestion → services → work orders) | 0:30 |
| 11 | **Impact and business model** | A 20% NRW reduction in a 100 MLD city saves about 7 billion litres a year (~₹15-25 Cr at ₹6-10/kL production cost); SaaS per DMA plus pilot integration. | Impact calculator, AMRUT 2.0 / Smart Cities Mission fit | 0:30 |
| 12 | **Roadmap and ask** | 90-day pilot on one DMA with the municipal corporation; acoustic fusion; pressure management. | Roadmap timeline, contact | 0:20 |

## Demo script (3 minutes)

1. **(0:00)** Dashboard at L4. Map coloured by *pipe health*, with predicted failure zones circled.
   Click a red cast-iron pipe to show its popup: age, corrosion, repairs, AI risk, RUL.
2. **(0:30)** Sidebar: inject a **large leak** using "physics hazard field" (the twin picks the
   likeliest pipe). Click **+1 h**.
3. **(1:00)** KPI shows 🚨 ALARM. AI tab: SHAP bars show *why* (local pressure contrast, DMA
   unaccounted flow). Turn **ground truth** on: the 🔴 true leak sits next to the 🟠 GNN pin, with
   the error in metres.
4. **(1:45)** Inject **sensor failure** on the nearest logger, step again, and show the system still
   localises using more distant sensors (Rule 5).
5. **(2:15)** Isolate the pipe and step. Loss goes to zero, downstream pressure dips, alarm clears.
6. **(2:40)** Evaluation tab: performance stays robust from L0 to L5, and the GNN beats the classical
   method at every level.

## Judge Q&A preparation

* *"Isn't synthetic data cheating?"* The truth is solved from the network equations, never sampled,
  and the tests prove it. Models are evaluated on unseen episodes, and the evaluation reports physical
  detection limits (small leaks) honestly.
* *"Why a GNN?"* Leaks are graph phenomena: the pressure signature propagates along pipes. The GNN
  generalises across missing sensors and beats the classical sensitivity method, which it also uses
  as an input prior.
* *"How do you go live?"* Import the GIS layer, calibrate the twin against two weeks of SCADA, point
  telemetry at `/ingest/telemetry`. No AI component changes.
