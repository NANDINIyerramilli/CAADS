"""
kill_probability.py — Stage 2.2: Neutralization Probability (Pk).

Since the CM dataset has NO real base_kill_probability column, we
**derive a SYNTHETIC / MODEL-ESTIMATED one** from a weighted
spec-compatibility score, then apply envelope-fit adjustment factors.

IMPORTANT — SYNTHETIC LABELLING:
    base_kill_probability is SYNTHETIC — NOT measured from live-fire
    trials or empirical data.  It combines weapon-class effectiveness
    tier with speed and altitude ratios to produce a 0–1 score.
    This is an academic/modeling estimate only.

FORMULA:
    Pk = base_pk × range_fit × reaction_margin × altitude_fit

    base_pk = 0.40 × weapon_class_tier
            + 0.35 × speed_ratio
            + 0.25 × altitude_ratio

    range_fit:        1.0 at optimal (midpoint) of CM envelope,
                      decaying toward min/max boundaries.
                      If threat approaches from outside (t_range >= c_max_r),
                      it is engaged at optimal range → 1.0.

    reaction_margin:  (TTI − reaction_time) / TTI, clamped [0, 1]
                      TTI = threat.engagement_range / threat.speed

    altitude_fit:     clamp(cm_max_altitude / threat_altitude, 0, 1)
                      For feasible pairs (cm_alt >= t_alt), this is 1.0.
                      Altitude is primarily a hard feasibility constraint;
                      among feasible systems it has no additional penalty.

ENGAGEMENT DISTANCE ASSUMPTION:
    The dataset provides one range value per threat (detection/engagement range).
    If the threat is detected beyond the CM's max range, the defense system
    fires as the threat enters the CM's optimal engagement zone (midpoint of
    [min_range, max_range]). This yields range_fit = 1.0.
    If the threat is already inside the CM envelope, engagement occurs at
    the threat's range, and range_fit decays away from the midpoint.
    This is an explicitly stated modeling assumption, not measured data.
"""

import numpy as np
import pandas as pd


def _compute_base_pk(
    threat: pd.Series,
    cm: pd.Series,
    config: dict,
) -> tuple[float, float, float, float]:
    """
    Derive SYNTHETIC base kill probability from spec compatibility.

    Formula (weights from config, must sum to 1.0):
        base_pk = 0.40 × weapon_class_tier
                + 0.35 × speed_ratio
                + 0.25 × altitude_ratio

    Range is NOT included (handled by range_fit_factor separately).

    Returns
    -------
    (base_pk, speed_ratio, altitude_ratio, tier_value)
    """
    w = config["synthetic_pk_weights"]
    tier_map = config["weapon_class_tier"]
    default_tier = tier_map.get("default_tier", 0.50)

    # Speed ratio
    t_speed = max(threat["max_speed_kmh"], 1)
    speed_ratio = min(cm["max_engageable_speed_kmh"] / t_speed, 1.0)

    # Altitude ratio
    t_alt = max(threat["max_altitude_m"], 1)
    alt_ratio = min(cm["max_altitude_m"] / t_alt, 1.0)

    # Weapon class tier
    tier = tier_map.get(cm["weapon_class"], default_tier)

    base = (
        w["weapon_class_tier"] * tier
        + w["speed_ratio"]     * speed_ratio
        + w["altitude_ratio"]  * alt_ratio
    )

    return float(np.clip(base, 0.0, 1.0)), speed_ratio, alt_ratio, tier


def _range_fit_factor(threat: pd.Series, cm: pd.Series) -> float:
    """
    1.0 at optimal (mid) range, decaying toward envelope edges.
    If threat approaches from outside (engagement_range_m >= max_range_m),
    it is engaged at optimal range -> 1.0.
    """
    t_range = threat["engagement_range_m"]
    c_min = cm["min_range_m"]
    c_max = cm["max_range_m"]
    if t_range >= c_max:
        return 1.0
    optimal = (c_min + c_max) / 2.0
    half_env = (c_max - c_min) / 2.0
    if half_env <= 0:
        return 1.0
    dist = abs(t_range - optimal)
    return float(np.clip(1.0 - dist / half_env, 0.0, 1.0))


def _reaction_margin_factor(threat: pd.Series, cm: pd.Series) -> float:
    """
    (time_to_impact − reaction_time) / time_to_impact, clamped [0, 1].
    TTI = threat.engagement_range / threat.speed
    """
    t_speed = threat["max_speed_ms"]
    if t_speed <= 0:
        return 1.0                  # stationary target → infinite margin
    tti = threat["engagement_range_m"] / t_speed
    margin = (tti - cm["reaction_time_s"]) / tti if tti > 0 else 0.0
    return float(np.clip(margin, 0.0, 1.0))


def _altitude_fit_factor(threat: pd.Series, cm: pd.Series) -> float:
    """
    Ratio of CM ceiling to threat altitude, clamped [0, 1].
    For feasible pairs (cm_alt >= t_alt), this is always 1.0.
    Altitude is treated primarily as a hard feasibility constraint.
    """
    t_alt = max(threat["max_altitude_m"], 1)
    return float(np.clip(cm["max_altitude_m"] / t_alt, 0.0, 1.0))


def compute_pk(
    threat: pd.Series,
    cm: pd.Series,
    config: dict,
) -> dict:
    """
    Compute the adjusted kill probability Pk for a single
    (threat, counter-measure) pair.

    Returns
    -------
    dict with keys: base_pk, speed_ratio, altitude_ratio, weapon_class_tier,
                    range_fit, reaction_margin, altitude_fit, Pk, Pk_pct
    """
    base, sr, ar, tier = _compute_base_pk(threat, cm, config)
    rf     = _range_fit_factor(threat, cm)
    rm     = _reaction_margin_factor(threat, cm)
    af     = _altitude_fit_factor(threat, cm)

    pk = base * rf * rm * af

    return {
        "base_pk":          round(base, 6),
        "speed_ratio":      round(sr, 6),
        "altitude_ratio":   round(ar, 6),
        "weapon_class_tier": round(tier, 4),
        "range_fit":        round(rf, 6),
        "reaction_margin":  round(rm, 6),
        "altitude_fit":     round(af, 6),
        "Pk":               round(pk, 6),
        "Pk_pct":           round(pk * 100, 2),   # percentage form
    }


def compute_all_pk(
    feasible_pairs: pd.DataFrame,
    threats: pd.DataFrame,
    cms: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    For every feasible (threat, CM) pair, compute Pk and attach columns.

    Uses vectorised numpy operations for speed.

    SYNTHETIC Pk — base_pk is model-estimated, NOT empirical.

    Parameters
    ----------
    feasible_pairs : DataFrame with threat_idx, cm_idx columns
    threats, cms   : cleaned DataFrames (indexed by threat_id / cm_id)
    config         : full config dict

    Returns
    -------
    feasible_pairs with added columns: base_pk, speed_ratio, altitude_ratio,
        weapon_class_tier, range_fit, reaction_margin, altitude_fit, Pk, Pk_pct
    """
    w = config["synthetic_pk_weights"]
    tier_map = config["weapon_class_tier"]
    default_tier = tier_map.get("default_tier", 0.50)

    t_idx = feasible_pairs["threat_idx"].values
    c_idx = feasible_pairs["cm_idx"].values

    # Threat arrays
    t_speed    = threats["max_speed_kmh"].values[t_idx]
    t_alt      = threats["max_altitude_m"].values[t_idx]
    t_range    = threats["engagement_range_m"].values[t_idx]
    t_speed_ms = threats["max_speed_ms"].values[t_idx]

    # CM arrays
    c_speed    = cms["max_engageable_speed_kmh"].values[c_idx]
    c_alt      = cms["max_altitude_m"].values[c_idx]
    c_min_r    = cms["min_range_m"].values[c_idx]
    c_max_r    = cms["max_range_m"].values[c_idx]
    c_react    = cms["reaction_time_s"].values[c_idx]
    c_wclass   = cms["weapon_class"].values[c_idx]

    # --- base_pk components (NO range_coverage — handled by range_fit) ---
    speed_ratio = np.clip(c_speed / np.maximum(t_speed, 1), 0, 1)
    alt_ratio   = np.clip(c_alt / np.maximum(t_alt, 1), 0, 1)
    tier_vals   = np.array([tier_map.get(wc, default_tier) for wc in c_wclass])

    base_pk = (
        w["weapon_class_tier"] * tier_vals
        + w["speed_ratio"]     * speed_ratio
        + w["altitude_ratio"]  * alt_ratio
    )
    base_pk = np.clip(base_pk, 0, 1)

    # --- range_fit (triangular centered on midpoint of CM envelope) ---
    # If threat approaches from outside (t_range >= c_max_r), engaged at optimal -> 1.0
    # If threat is inside envelope (t_range < c_max_r), triangular decay from optimal
    optimal  = (c_min_r + c_max_r) / 2.0
    half_env = np.maximum((c_max_r - c_min_r) / 2.0, 1e-9)   # avoid div-by-zero
    dist_from_optimal = np.where(t_range >= c_max_r, 0.0, np.abs(t_range - optimal))
    range_fit = np.clip(1.0 - dist_from_optimal / half_env, 0.0, 1.0)

    # --- reaction_margin: (TTI - reaction_time) / TTI ---
    # TTI = threat.engagement_range / threat.speed (total available reaction window)
    t_speed_ms_capped = np.maximum(np.minimum(t_speed_ms, 10000.0), 1e-6)
    tti = t_range / t_speed_ms_capped
    with np.errstate(divide="ignore", invalid="ignore"):
        reaction_margin = np.clip(np.where(tti > 0, (tti - c_react) / tti, 1.0), 0.0, 1.0)

    # --- altitude_fit ---
    # For feasible pairs, cm_alt >= t_alt, so this is always 1.0.
    # Kept for completeness and formula transparency.
    altitude_fit = np.clip(c_alt / np.maximum(t_alt, 1), 0, 1)

    # --- final Pk ---
    pk = base_pk * range_fit * reaction_margin * altitude_fit

    result = feasible_pairs.copy()
    result["base_pk"]          = np.round(base_pk, 6)
    result["speed_ratio"]      = np.round(speed_ratio, 6)
    result["altitude_ratio"]   = np.round(alt_ratio, 6)
    result["weapon_class_tier"] = tier_vals
    result["range_fit"]        = np.round(range_fit, 6)
    result["reaction_margin"]  = np.round(reaction_margin, 6)
    result["altitude_fit"]     = np.round(altitude_fit, 6)
    result["Pk"]               = np.round(pk, 6)
    result["Pk_pct"]           = np.round(pk * 100, 2)

    # Attach cost and inventory for the decision engine
    result["cost_per_engagement"] = cms["cost_per_engagement"].values[c_idx]
    result["inventory_count"]     = cms["inventory_count"].values[c_idx]
    result["cm_weapon_class"]     = c_wclass

    return result


if __name__ == "__main__":
    from data_loader import load_all
    from priority_score import compute_tps
    from feasibility import filter_feasible

    cfg, threats, cms = load_all()
    ranked = compute_tps(threats, cfg)
    feas = filter_feasible(ranked, cms)
    feas_pk = compute_all_pk(feas, ranked, cms, cfg)

    print(f"Feasible pairs with Pk: {len(feas_pk)}")
    print(f"\nPk distribution:")
    print(feas_pk["Pk_pct"].describe())
    print(f"\nTop-10 by Pk:")
    print(feas_pk.nlargest(10, "Pk_pct")[
        ["threat_name", "cm_name", "base_pk", "Pk_pct", "cost_per_engagement"]
    ].to_string())
