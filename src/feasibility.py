"""
feasibility.py — Stage 2.1: Hard-constraint feasibility filter.

Checks six hard rules for every (threat, counter-measure) pair:
  1. Range envelope     : threat must enter CM's engagement zone
                          eff_dist = min(threat.range, cm.max_range); check eff_dist >= cm.min_range
  2. Altitude ceiling   : cm.max_altitude >= threat.max_altitude
  3. Speed capability   : cm.max_engageable_speed >= threat.max_speed
                          A system that cannot physically track the target is infeasible.
  4. Reaction time      : time_to_impact = threat.engagement_range / threat.speed > cm.reaction_time
                          Uses threat detection/engagement range for TTI, representing the
                          total time available from detection to impact.
  5. RF Link rule       : Rejected ONLY if cm.RequiresTargetRF == True and threat.LiveRFLink == False
                          (RF-dependent CM cannot operate against autonomous/non-RF target)
  6. Inventory          : cm.inventory_count > 0

ENGAGEMENT DISTANCE MODELING ASSUMPTION:
  The dataset provides a single range value per threat (detection/engagement range).
  This is interpreted as: the maximum distance at which the threat is detected or
  engaged. Time-to-impact is computed as threat.engagement_range / threat.speed,
  giving the total available reaction window from detection.
  The range envelope check uses min(threat.range, cm.max_range) to determine
  whether the threat's trajectory passes through the CM's engagement zone.
"""

import pandas as pd
import numpy as np


def check_feasibility(
    threat: pd.Series,
    cm: pd.Series,
) -> tuple[bool, str]:
    """
    Check whether a single counter-measure is feasible against a threat.

    Returns
    -------
    (is_feasible, reason)
        reason is empty string if feasible, else the failing constraint.
    """
    t_range = threat["engagement_range_m"]
    t_alt = threat["max_altitude_m"]
    t_speed = threat["max_speed_ms"]
    t_speed_kmh = threat["max_speed_kmh"]
    t_rf = threat["rf_link"]
    c_rf = cm["rf_link_compatible"]

    # 1. Range envelope — threat must enter CM's engagement zone
    eff_dist = min(t_range, cm["max_range_m"])
    if eff_dist < cm["min_range_m"]:
        return False, "range_envelope"

    # 2. Altitude ceiling
    if cm["max_altitude_m"] < t_alt:
        return False, "altitude_ceiling"

    # 3. Speed capability — CM must be able to track/engage the threat
    if t_speed_kmh > 0 and cm["max_engageable_speed_kmh"] < t_speed_kmh:
        return False, "speed_capability"

    # 4. Reaction time vs time-to-impact (from detection range)
    t_speed_capped = min(t_speed, 10000.0)  # max ~36,000 km/h
    if t_speed_capped > 0:
        tti = t_range / t_speed_capped
        if tti <= cm["reaction_time_s"]:
            return False, "reaction_time"

    # 5. RF Link rule (asymmetric):
    # Threat RF=False, CM RequiresTargetRF=True -> Infeasible (RF-dependent CM unavailable)
    # All other combinations -> Feasible
    if c_rf and not t_rf:
        return False, "rf_dependent_cm_unavailable"

    # 6. Inventory
    if cm["inventory_count"] <= 0:
        return False, "no_inventory"

    return True, ""


def filter_feasible(
    threats: pd.DataFrame,
    cms: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the full feasibility matrix and return only feasible pairs.
    Uses vectorised numpy broadcasting for speed.
    """
    t_range   = threats["engagement_range_m"].values[:, None]   # (N, 1)
    t_alt     = threats["max_altitude_m"].values[:, None]
    t_speed   = threats["max_speed_ms"].values[:, None]
    t_speed_kmh = threats["max_speed_kmh"].values[:, None]
    t_rf      = threats["rf_link"].values[:, None]

    c_min_r   = cms["min_range_m"].values[None, :]              # (1, M)
    c_max_r   = cms["max_range_m"].values[None, :]
    c_alt     = cms["max_altitude_m"].values[None, :]
    c_speed_kmh = cms["max_engageable_speed_kmh"].values[None, :]
    c_react   = cms["reaction_time_s"].values[None, :]
    c_rf      = cms["rf_link_compatible"].values[None, :]
    c_inv     = cms["inventory_count"].values[None, :]

    # 1. Range envelope: effective engagement distance reaches CM min range
    eff_dist = np.minimum(t_range, c_max_r)
    ok_range = eff_dist >= c_min_r

    # 2. Altitude ceiling
    ok_alt = c_alt >= t_alt

    # 3. Speed capability: CM must handle threat speed
    ok_speed = (t_speed_kmh <= 0) | (c_speed_kmh >= t_speed_kmh)

    # 4. Reaction time (using threat detection range for TTI)
    t_speed_capped = np.minimum(t_speed, 10000.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        tti = np.where(t_speed_capped > 0, t_range / t_speed_capped, np.inf)
    ok_react = tti > c_react

    # 5. RF Link rule: Rejected ONLY if (c_rf == True and t_rf == False)
    # ok_rf is True unless (c_rf == True and t_rf == False) -> ~c_rf | t_rf
    ok_rf = ~c_rf | t_rf

    # 6. Inventory
    ok_inv = c_inv > 0

    # Combined mask
    feasible_mask = ok_range & ok_alt & ok_speed & ok_react & ok_rf & ok_inv

    t_indices, c_indices = np.where(feasible_mask)

    records = pd.DataFrame({
        "threat_idx": t_indices,
        "cm_idx":     c_indices,
        "threat_name": threats["threat_name"].values[t_indices],
        "cm_name":     cms["cm_name"].values[c_indices],
    })

    return records


if __name__ == "__main__":
    from data_loader import load_all
    from priority_score import compute_tps

    cfg, threats, cms = load_all()
    ranked = compute_tps(threats, cfg)
    feas = filter_feasible(ranked, cms)
    print(f"Total feasible pairs: {len(feas)}")
    print(f"Threats with >=1 feasible CM: {feas['threat_idx'].nunique()} / {len(threats)}")
    print(f"CMs used at least once: {feas['cm_idx'].nunique()} / {len(cms)}")

