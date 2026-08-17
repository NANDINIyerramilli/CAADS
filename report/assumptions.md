# Aerial Defense Decision-Support System — Assumptions & Methodology

## 1. System Overview

This document accompanies the CAADS (Counter-Air Defense System), a three-stage
decision-support pipeline that:

1. **Scores** incoming threats by danger level (Threat Priority Score)
2. **Filters** counter-measures by hard engagement constraints (Feasibility)
3. **Selects** the best counter-measure using a cost-aware decision engine

---

## 2. Data Cleaning Decisions

### 2.1 Threat Dataset (1,073 raw → 1,065 after cleaning)

| Issue | Rows Affected | Resolution |
|-------|--------------|------------|
| Payload as text (`"270 kg"`, `"3,200 kg (1x Nuclear)"`) | 474 | Regex extraction of first numeric token; commas stripped |
| Payload = NaN / `None` / non-numeric | 2 | Replaced with 0.0 |
| Engagement range with dashes (`"75-100"`) | 6 | Split on `-`/`–`, averaged both values |
| Engagement range as datetime (`"1900-01-20"`) | 1 | Extracted day component as best-guess value |
| Speed with text (`"0 (Tethered)"`) | 1 | Extracted leading number (0) |
| Altitude with commas/text (`"100,000+"`) | 1 | Stripped commas and `+` |
| Altitude with negative/depth values (`"-600 (Depth)"`) | 6 | **Excluded** — these are underwater torpedoes, not aerial threats |
| Missing altitude (NaN after parsing) | 2 | **Excluded** — cannot evaluate without altitude |

### 2.2 Counter-Measure Dataset (398 raw → ~250 after filtering)

| Issue | Resolution |
|-------|------------|
| Non-engagement classes (Radars, Sensors, Ballistic Missiles, SLBMs, Cruise Missiles, Air-to-Air Missiles, etc.) | **Excluded** — 117 rows removed. These are detection/offensive systems, not counter-measures. |
| NaN in reaction_time (39 rows) | Filled with class-median, then global-median fallback |
| NaN in cost_per_engagement (31 rows) | Filled with class-median, then global-median fallback |
| NaN in max_engageable_speed (61 rows) | Filled with class-median, then global-median fallback |
| NaN in max_altitude (11 rows) | Filled with class-median, then global-median fallback |

### 2.3 Unit Conversion

- Threat engagement ranges are in **km**; CM ranges are in **metres**.
- All range comparisons are done in **metres** (threat ranges × 1000).
- Speeds are converted to **m/s** for time-to-impact calculations.

---

## 3. Threat Priority Score (TPS) — Stage 1

### Formula

```
TPS = 0.25 × norm(max_speed)
    + 0.20 × norm(1/altitude)          # low altitude = harder to detect
    + 0.15 × norm(payload)
    + 0.15 × norm(1/detection_range)   # short range = more dangerous
    + 0.10 × (rf_link ? 0 : 1)         # no RF link = autonomous = harder
    + 0.15 × norm(threat_class_severity)
```

### Normalisation

Min-max normalisation per column across the entire threat dataset.
Constant columns receive a default value of 0.5.

### Threat Class Severity

43 unique threat classes mapped to a 1–10 severity scale:
- **10**: Hypersonic Glide, Hypersonic Cruise, Bomber
- **8**: Cruise Missiles (all variants)
- **6**: UCAV, Combat UAV, Loitering Munition, Guided Bombs
- **4**: Surveillance UAV, Helicopter
- **2**: Small Commercial UAV, Aerostat
- **1**: Surveillance UAV (MAV)

Unmapped classes default to severity **5**.

---

## 4. Feasibility Filter — Stage 2.1

Five hard constraints (all must pass):

1. **Range**: `cm.min_range ≤ threat.engagement_range_m ≤ cm.max_range`
2. **Altitude**: `cm.max_altitude ≥ threat.max_altitude`
3. **Reaction time**: `time_to_impact = range / speed > reaction_time`
4. **RF-link (asymmetric)**:
   - Jammer/EW (`rf_link_compatible=True`) → only feasible if threat has RF link
   - Kinetic system (`rf_link_compatible=False`) → feasible regardless
5. **Inventory**: `count > 0`

**Note on jammers**: For jammer-vs-RF-threat pairs, `Pk` represents the
probability of successfully disrupting the guidance link, not a physical kill.
Same 0–100% scale for comparability.

---

## 5. Kill Probability (Pk) — Stage 2.2

### ⚠ CRITICAL ASSUMPTION: Synthetic base_kill_probability

**The counter-measure dataset contains NO real base_kill_probability column.**
We derive a synthetic one. This is the single most defensible thing to
challenge in a viva/demo.

### Synthetic base_pk formula

```
base_pk = 0.30 × speed_ratio        # cm.max_speed / threat.max_speed, clamped [0,1]
        + 0.20 × altitude_ratio      # cm.max_alt / threat.max_alt, clamped [0,1]
        + 0.20 × range_coverage      # envelope_breadth / threat.range, clamped [0,1]
        + 0.30 × weapon_class_tier   # class lookup (e.g. MR-SAM=0.85, MANPADS=0.65)
```

### Adjustment factors

```
Pk = base_pk × range_fit × reaction_margin × altitude_fit
```

- **range_fit**: Triangular function centred on envelope midpoint; 1.0 at optimal, 0.0 at edges
- **reaction_margin**: `(time_to_impact − reaction_time) / time_to_impact`, clamped [0,1]
- **altitude_fit**: `cm.max_alt / threat.max_alt`, clamped [0,1]

### Weapon class tiers

Each of the 120 unique CM weapon classes is assigned a base effectiveness tier
between 0.0 and 1.0 (see `config.yaml` for the full mapping). Examples:
- Long Range SAM: 0.90
- MR-SAM: 0.85
- MANPADS: 0.65
- AA Gun: 0.55
- RF Jammer: 0.60

---

## 6. Cost-Aware Decision Engine — Stage 3

### Decision Bands

| Condition | Band | Logic |
|-----------|------|-------|
| `gap ≤ 5pp` and `Pk_best ≥ 50%` | **Cost Tiebreak** | Pick cheapest among options within 5pp of best Pk |
| `5pp < gap ≤ 15pp` and `Pk_best ≥ 50%` | **Weighted Blend** | Value = 0.7·norm(Pk) − 0.3·norm(cost) |
| `gap > 15pp` | **Pk Priority** | Best Pk wins outright |
| `Pk_best < 50%` | **Combination Mode** | Pk wins single; search multi-weapon salvos |

### Combination Mode

When triggered (best single Pk < 50%):
1. Enumerate all 2-weapon combos: `P_combined = 1 − Π(1 − Pk_i)`
2. Filter combos where `P_combined ≥ 85%` (TARGET_PK)
3. Among those, pick lowest total cost
4. If no 2-combo meets target, try 3-combos
5. If none achievable, return best combo with a warning flag

### Parameters (all tunable in `config.yaml`)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| SMALL_GAP | 5% | Max gap for cost tiebreak |
| LARGE_GAP | 15% | Min gap for Pk priority |
| CRITICAL_PK | 50% | Below this, combo mode triggers |
| TARGET_PK | 85% | Target for combination salvos |
| α | 0.70 | Pk weight in weighted blend |
| β | 0.30 | Cost weight in weighted blend |

---

## 7. Known Limitations

1. **Synthetic Pk is not validated against real-world data.** It is a
   heuristic proxy. Real systems use live-fire trial data, simulation models,
   or classified engagement tables.

2. **Independence assumption in combination mode.** We assume independent
   engagement attempts: `P_combined = 1 − Π(1 − Pk_i)`. In reality,
   interceptors may be correlated (same failure mode, same target track).

3. **Static engagement model.** The system does not model dynamic factors:
   multi-threat saturation, time-sequenced engagements, ammunition depletion
   across a campaign, or sensor-to-shooter handoff delays.

4. **Threat class severity is subjective.** The 1–10 mapping is based on
   general doctrine intuition, not a formal threat assessment methodology.

5. **NaN imputation uses class median.** For CMs with missing specs, this
   is reasonable but may over- or under-estimate specific systems.

---

## 8. Pipeline Results Summary

After running on the full datasets:

| Metric | Value |
|--------|-------|
| Threats processed | 1,065 |
| With recommendation | 902 (84.7%) |
| No feasible CM | 163 (15.3%) |
| Combination-mode picks | 117 (11.0%) |
| Cost tiebreak picks | 466 (43.8%) |
| Weighted blend picks | 192 (18.0%) |
| Pk priority picks | 87 (8.2%) |
| Single-option picks | 19 (1.8%) |

---

## 9. How to Tune

All parameters are in `config.yaml`. Typical tuning scenarios:

- **"System is too cost-focused"** → increase α, decrease β, or raise SMALL_GAP
- **"Too many combo-mode triggers"** → lower CRITICAL_PK (e.g. 40%)
- **"Some threats seem mis-ranked"** → adjust TPS weights or severity mapping
- **"Pk values seem too high/low"** → adjust weapon_class_tier values or
  synthetic_pk_weights

---

*Generated for the CAADS project, August 2026.*
