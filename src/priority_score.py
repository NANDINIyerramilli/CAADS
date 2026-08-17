"""
priority_score.py — Stage 1: Threat Priority Score (TPS).

Computes a weighted composite score for each threat to determine
engagement priority.  Higher TPS = more dangerous = engage first.

TPS = w1·norm(speed) + w2·norm(1/altitude) + w3·norm(payload)
    + w4·norm(1/detection_range) + w5·(no_rf_link ? 1 : 0)
    + w6·norm(threat_class_severity)
"""

import numpy as np
import pandas as pd


def _min_max_normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize a Series to [0, 1].  Constant series → 0.5."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


def _map_threat_class_severity(
    threat_classes: pd.Series,
    severity_map: dict,
    default: int = 5,
) -> pd.Series:
    """Map each threat class string to its severity integer."""
    default_val = severity_map.get("default_severity", default)
    return threat_classes.map(
        lambda tc: severity_map.get(tc, default_val)
    ).astype(float)


def compute_tps(threats: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Add TPS and TPS_rank columns to the threats DataFrame.

    Parameters
    ----------
    threats : DataFrame  — cleaned threat data (from data_loader)
    config  : dict       — full config (needs 'tps_weights' and
                           'threat_class_severity' sections)

    Returns
    -------
    DataFrame with added columns: TPS, TPS_rank  (sorted by TPS desc)
    """
    w = config["tps_weights"]
    sev_map = config["threat_class_severity"]

    df = threats.copy()

    # --- Normalize components ---
    norm_speed = _min_max_normalize(df["max_speed_kmh"])

    # Low altitude → harder to detect → more dangerous → use 1/altitude
    inv_alt = 1.0 / df["max_altitude_m"].replace(0, np.nan).fillna(1)
    norm_detectability = _min_max_normalize(inv_alt)

    norm_payload = _min_max_normalize(df["payload_kg"])

    # Short detection/engagement range → more dangerous → 1/range
    inv_range = 1.0 / df["engagement_range_km"].replace(0, np.nan).fillna(1)
    norm_det_range = _min_max_normalize(inv_range)

    # RF link: LiveRFLink == False (no usable RF link / autonomous) -> score 1 (harder to counter)
    rf_score = (~df["rf_link"]).astype(float)

    # Threat class severity
    severity_raw = _map_threat_class_severity(df["threat_class"], sev_map)
    norm_severity = _min_max_normalize(severity_raw)

    # --- Weighted sum ---
    df["TPS"] = (
        w["speed"]           * norm_speed
        + w["detectability"] * norm_detectability
        + w["payload"]       * norm_payload
        + w["detection_range"] * norm_det_range
        + w["rf_link"]       * rf_score
        + w["threat_class"]  * norm_severity
    )

    # Clamp to [0, 1] (should already be, but safety)
    df["TPS"] = df["TPS"].clip(0.0, 1.0)

    # Rank: 1 = most dangerous
    df["TPS_rank"] = df["TPS"].rank(ascending=False, method="min").astype(int)

    # Sort by TPS descending
    df = df.sort_values("TPS", ascending=False).reset_index(drop=True)

    return df


if __name__ == "__main__":
    from data_loader import load_all

    cfg, threats, _cms = load_all()
    ranked = compute_tps(threats, cfg)
    print("Top-20 threats by TPS:")
    print(ranked[["threat_name", "threat_class", "TPS", "TPS_rank"]].head(20).to_string())
    print(f"\nTPS range: {ranked['TPS'].min():.4f} – {ranked['TPS'].max():.4f}")
