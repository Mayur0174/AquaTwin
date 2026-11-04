# GIS Dashboard and Digital Twin Control Panel: Wireframes

Implemented in `dashboard/app.py` (Streamlit + Folium + Plotly). The colour system follows a
validated, colour-blind-safe palette. Health and risk tiers use labelled status colours (good /
warning / serious / critical). Magnitudes (pressure, flow) use a single-hue blue ramp. Leak
probability uses a single-hue orange ramp. SHAP uses a blue↔red diverging pair. Charts never use
dual axes.

## Main layout

```
┌──────────────────────────┬───────────────────────────────────────────────────────────────────────┐
│ 🎛️ TWIN CONTROL PANEL    │ 💧 AquaTwin AI — Water Network Digital Twin                            │
│ ▸ Simulation             │ ┌────────┬────────┬────────────┬──────────────┬──────────┬───────────┐ │
│   Realism level [L3 ▾]   │ │ Time   │ Demand │ AI status  │ Est. loss    │ True loss│ Pump/Tanks│ │
│   Seed [0] Horizon [48h] │ │ 06-01  │ 182L/s │ 🚨 ALARM   │ 9.6 L/s      │ 10.1 L/s │ ON/4.2,6.0│ │
│   [▶ Start simulation]   │ └────────┴────────┴────────────┴──────────────┴──────────┴───────────┘ │
│   [+15min][+1h][+6h]     │ [🗺 Map][⏯ Playback][🧠 AI & XAI][🏗 Asset risk][🚰 Loss][🧪 Eval][📚]  │
│ ▸ Inject events          │ ┌───────────────────────────────────────────────────────────────────┐ │
│   Pipe [hazard field ▾]  │ │ Colour by [Pipe health▾] Material[▾] DMA[▾] Min risk[—] Age[——]    │ │
│   Size  small…burst      │ │                                                                   │ │
│   Position ──●──         │ │        ═══ pipes coloured by layer, width ∝ diameter              │ │
│   [💧 Leak] [💥 Burst]   │ │   🛢 tank      • loggers     ○ predicted failure zone              │ │
│   Sensor [PS-J0305 ▾]    │ │        🔴 true leak (sim only)   🟠 AI candidates #1-3            │ │
│   Fault [failure ▾]      │ │   ⚙ pump station   💧 reservoir                                    │ │
│   [📡 Inject fault]      │ │   click pipe → popup: material, age, corrosion, health,           │ │
│   [🔒 Isolate pipe]      │ │   AI risk, RUL, DMA, valve, pressure, flow, leak probability      │ │
│ ▸ Demand & weather       │ └───────────────────────────────────────────────────────────────────┘ │
│   factor, land use, hours│  ━━ good ≥70  ━━ warning 50–70  ━━ serious 35–50  ━━ critical <35     │
│   temperature, rainfall  │                                                                       │
│ ▸ Replay scenario  [▾]   │                                                                       │
│ [◉] Show ground truth    │                                                                       │
└──────────────────────────┴───────────────────────────────────────────────────────────────────────┘
```

## Tabs

```
⏯ TIMELINE PLAYBACK                              🧠 AI DECISIONS & XAI
┌───────────────────────────────────────────┐    ┌──────────────────────────────┬──────────────────┐
│ Playback step ────────────●──────         │    │ 🚨 08:15 ALARM               │ Why? (SHAP)       │
│ 2026-06-01 08:15 · events: EV00012        │    │ Pipe P118: leak prob 93% …   │ ▇▇▇▇ contrast  +  │
│ ┌──────────────┐ ┌──────────────────────┐ │    │ 📍 08:15 LOCALISATION        │ ▇▇▇ DMA ufw    +  │
│ │ pressure map │ │ observed/expected/   │ │    │ GNN(sage): P118 @ 132 m …    │ ▇ corrosion    +  │
│ │ at step t    │ │ truth at logger      │ │    │ ✅ 14:00 CLEAR               │ ▅ sensor hlth  −  │
│ │              │ ├──────────────────────┤ │    │ Candidates table (top-5)     │ Global |SHAP|     │
│ │              │ │ max leak probability │ │    └──────────────────────────────┴──────────────────┘
│ │              │ │ ─ ─ alarm threshold  │ │
│ │              │ ├──────────────────────┤ │    🏗 ASSET RISK
│ └──────────────┘ │ true loss (L/s)      │ │    ┌──────────────────────────────┬──────────────────┐
│                  └──────────────────────┘ │    │ Priority list (risk, RUL)    │ risk by material │
└───────────────────────────────────────────┘    │ P062 CI 61y 0.71 3 rep 24%   │ Explain pipe [▾] │
                                                  │ …                            │ narrative + SHAP │
🚰 WATER LOSS: DMA estimate vs truth, volumes      └──────────────────────────────┴──────────────────┘
🧪 EVALUATION: F1 & accuracy by level · localisation error GNN vs classical · FP/FN bars · table
📚 SCENARIOS: catalogue with rules · session event log with causal explanations
```

## Interaction requirements mapped

| Requirement | Where |
|---|---|
| Network map | Map tab (Folium, CartoDB Positron tiles) |
| Pipe health / risk score | "Colour pipes by" layer: health, AI risk, pressure, flow, leak probability |
| Leak locations | 🔴 true leaks (toggle), 🟠 GNN candidates |
| Water loss | KPI tiles + Water-loss tab (DMA balance vs truth) |
| Predicted failure zones | ○ circles on the 10 highest-risk pipes |
| Interactive filters | Material, DMA, min risk, age range |
| Timeline playback | Playback slider over recorded steps |
| Scenario playback | Sidebar "Replay scenario" + playback |
| Asset details popup | Click any pipe |
| Control panel | Sidebar: start, inject leak/burst/sensor fault, demand, weather, replay, isolate |
| View AI decisions | AI Decisions tab: log, candidates, SHAP |
