# Cost-Aware Aerial Defence System (CAADS)

A decision-support system designed to analyze incoming aerial threat profiles, evaluate physical and operational feasibility across a countermeasure inventory, calculate synthetic neutralization probabilities ($P_k$), and recommend optimal defense choices using a cost-aware decision engine.

---
<img width="1919" height="859" alt="image" src="https://github.com/user-attachments/assets/f6962f1c-a310-4101-94e0-003e0de14fb8" />

<img width="1893" height="802" alt="image" src="https://github.com/user-attachments/assets/3a7453bd-5552-4cb5-9ae2-0b5809bced7b" />
<img width="1896" height="866" alt="image" src="https://github.com/user-attachments/assets/6fad6f88-52e8-420c-9275-135cd3ef638c" />
<img width="1838" height="612" alt="image" src="https://github.com/user-attachments/assets/18e04a1a-93fe-41c2-96bc-08189aafcc90" />


## Objective
Build a decision-support system that:
Takes an incoming threat weapon as input (from Dataset A).
Scores its priority/danger level using a weighted requirement matrix.
Matches it against a counter-measure inventory (Dataset B) using engagement envelope compatibility.
Computes a neutralization probability for each feasible counter-measure.
Picks the best counter-measure using a cost-aware decision engine — cheaper options win only when they're "close enough" in effectiveness; effectiveness wins outright when the gap is large or when both options are in a low-confidence zone; and in the low-confidence zone the system also proposes a multi-weapon combination to raise the odds of a kill.


## 🛡️ Key Features

- **Stage 1 — Threat Priority Score (TPS)**: Scores and ranks incoming threats based on speed, altitude, payload, detection range, RF emission state, and threat class severity.
- **Stage 2.1 — Hard-Constraint Feasibility Filtering**: Evaluates 6 physical rules before computing effectiveness:
  1. Range Envelope ($\text{eff\_dist} \ge \text{CM min range}$)
  2. Altitude Ceiling ($\text{CM ceiling} \ge \text{Threat altitude}$)
  3. Speed Capability ($\text{CM max speed} \ge \text{Threat max speed}$)
  4. Reaction Time ($\text{TTI} > \text{CM reaction time}$)
  5. RF Link Compatibility ($\text{RequiresTargetRF} \implies \text{LiveRFLink}$)
  6. Inventory Count ($\text{Inventory} > 0$)
- **Stage 2.2 — Neutralization Probability ($P_k$)**: Derives synthetic kill probability combining weapon effectiveness tier, speed ratio, altitude ratio, range fit, reaction margin, and altitude fit.
- **Stage 3 — Cost-Aware Decision Engine**:
  - **Band 1 (Cost Tiebreak)**: When $P_k$ gap $\le 5\text{pp}$ and $P_k \ge 70\%$, select the cheapest option.
  - **Band 2 (Weighted Blend)**: When $5\text{pp} < \text{gap} \le 15\text{pp}$ and $P_k \ge 70\%$, blend effectiveness and cost ($\alpha = 0.70, \beta = 0.30$).
  - **Band 3 (Pk Priority)**: When $\text{gap} > 15\text{pp}$, highest $P_k$ wins outright.
  - **Band 4 (Low Confidence & Combination Mode)**: When $P_k < 70\%$, $P_k$ wins single selection. Below $60\%$, automatically searches multi-weapon salvos for $P_{\text{combined}} \ge 85\%$.

---

## 📁 Repository Structure

```text
aerial_defense_system/
├── app.py                  # Streamlit single-page web dashboard
├── config.yaml             # Tunable system configuration & weights
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── test_audit.py           # Verification test suite
├── data/                   # Threat and Countermeasure datasets (.xlsx)
├── report/                 # Assumptions & methodology reports
└── src/                    # Core Python pipeline packages
    ├── __init__.py
    ├── data_loader.py      # Data cleaning, unit conversion & imputation
    ├── priority_score.py   # Stage 1: Threat Priority Score (TPS)
    ├── feasibility.py      # Stage 2.1: Hard-constraint feasibility filter
    ├── kill_probability.py # Stage 2.2: Synthetic Pk calculation
    ├── decision_engine.py  # Stage 3: Cost-aware decision engine
    └── pipeline.py         # End-to-end pipeline orchestrator
```

---

## 🚀 Quickstart Guide

### 1. Clone Repository & Navigate
```bash
git clone https://github.com/NANDINIyerramilli/CAADS.git
cd CAADS/aerial_defense_system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Web Application
```bash
streamlit run app.py
```

Open **[http://localhost:8501](http://localhost:8501)** in your browser to access the dashboard.

---

## 🧪 Running Verification Tests

To verify scalar vs. vectorized calculations and model consistency:

```bash
python test_audit.py
```

---

## ⚠️ Modeling Assumptions

This project is a synthetic decision-support and academic modeling system. All $P_k$ values are model-estimated based on specification compatibility scores and do not represent empirical live-fire trial data.
