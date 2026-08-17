"""
data_loader.py — Load and clean both Excel datasets.

Handles:
  - Payload text extraction  ("270 kg", "3,200 kg (1x Nuclear)" → 270.0, 3200.0)
  - Range-format parsing     ("75-100" → 87.5;  datetime artefacts → manual fix)
  - Altitude cleaning        ("100,000+" → 100000;  negative/depth → excluded)
  - Speed cleaning           ("0 (Tethered)" → 0)
  - NaN imputation           (class-median, then global-median fallback)
  - CM filtering             (drop non-engagement classes: radars, sensors, BMs…)
  - Unit conversion          (threat engagement range km → m)
"""

import re
import yaml
import pathlib
import numpy as np
import pandas as pd


def load_config(config_path: str | None = None) -> dict:
    """Load the YAML configuration file."""
    if config_path is None:
        config_path = pathlib.Path(__file__).resolve().parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
#  Helpers — numeric extraction
# ─────────────────────────────────────────────

def _extract_number(val) -> float:
    """
    Pull the first numeric token from an arbitrary value.
    '270 kg' → 270.0 | '3,200 kg (1x Nuclear)' → 3200.0 | NaN → 0.0
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0
    s = str(val).strip()
    if s == "" or s.lower() in ("na", "nan", "none", "non", "nil", "-", "n/a"):
        return 0.0
    # Remove commas from numbers like "3,200"
    s = s.replace(",", "")
    m = re.search(r"-?\d+\.?\d*", s)
    return float(m.group()) if m else 0.0


def _parse_range_field(val) -> float:
    """
    Parse engagement/detection range.
    Plain number → float  |  '75-100' → 87.5  |  datetime string → attempt fix
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    s = str(val).strip()

    # Detect datetime artefacts like '1900-01-20 00:00:00'
    if re.match(r"\d{4}-\d{2}-\d{2}", s):
        # The original value was probably a range like '20-XX'
        # that Excel interpreted as a date.  Extract day as best-guess.
        day_match = re.search(r"-(\d{2})\s", s)
        if day_match:
            return float(day_match.group(1))
        return np.nan

    # Range with dash/en-dash: '75-100', '6000–10000'
    parts = re.split(r"[–\-]", s)
    if len(parts) == 2:
        try:
            a, b = float(parts[0].replace(",", "")), float(parts[1].replace(",", ""))
            return (a + b) / 2.0
        except ValueError:
            pass

    # Plain number
    return _extract_number(s) or np.nan


def _parse_altitude(val) -> float:
    """
    Parse altitude.  '100,000+' → 100000 | '-600 (Depth)' → NaN (underwater)
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    s = str(val).strip().replace(",", "")
    # Negative / depth → not an aerial threat
    if "depth" in s.lower() or (s.startswith("-") and "depth" in s.lower()):
        return np.nan
    m = re.search(r"-?\d+\.?\d*", s)
    if m:
        v = float(m.group())
        return v if v >= 0 else np.nan   # negative altitudes → exclude
    return np.nan


# ─────────────────────────────────────────────
#  Load & clean THREATS
# ─────────────────────────────────────────────

def load_threats(config: dict) -> pd.DataFrame:
    """Return a clean threats DataFrame."""
    root = pathlib.Path(__file__).resolve().parent.parent
    path = root / config["data"]["threats_file"]
    df = pd.read_excel(path)

    # Standardise column names internally
    df = df.rename(columns={
        "THREAT Name": "threat_name",
        "Threat Class": "threat_class",
        "Max Speed (km/h)": "max_speed_kmh",
        "Altitude Ceiling (m)": "max_altitude_m",
        "Engagement / Detection Range (km)": "engagement_range_km",
        "Payload": "payload_kg",
        "Live RF Link (True/False)": "rf_link",
    })

    # --- Clean each column ---
    df["max_speed_kmh"] = df["max_speed_kmh"].apply(_extract_number)
    df["max_altitude_m"] = df["max_altitude_m"].apply(_parse_altitude)
    df["engagement_range_km"] = df["engagement_range_km"].apply(_parse_range_field)
    df["payload_kg"] = df["payload_kg"].apply(_extract_number)

    # Convert RF link to bool (already bool in source, but safeguard)
    df["rf_link"] = df["rf_link"].astype(bool)

    # Drop underwater / non-aerial entries (negative or NaN altitude)
    df = df.dropna(subset=["max_altitude_m"])
    df = df[df["max_altitude_m"] > 0].copy()

    # Drop rows with NaN engagement range (very rare edge case)
    df = df.dropna(subset=["engagement_range_km"])

    # Convert engagement range km → metres for matching with CM
    df["engagement_range_m"] = df["engagement_range_km"] * 1000.0

    # Speed in m/s (for time-to-impact calculations)
    df["max_speed_ms"] = df["max_speed_kmh"] / 3.6

    # Reset index
    df = df.reset_index(drop=True)
    df.index.name = "threat_id"

    return df


# ─────────────────────────────────────────────
#  Load & clean COUNTER-MEASURES
# ─────────────────────────────────────────────

def _is_engagement_capable(weapon_class: str, exclude_patterns: list[str]) -> bool:
    """Return True if the weapon class represents an engagement-capable system."""
    wc = str(weapon_class)
    for pat in exclude_patterns:
        if pat.lower() in wc.lower():
            return False
    return True


def load_countermeasures(config: dict) -> pd.DataFrame:
    """Return a clean counter-measures DataFrame (engagement-capable only)."""
    root = pathlib.Path(__file__).resolve().parent.parent
    path = root / config["data"]["countermeasures_file"]
    df = pd.read_excel(path)

    # Standardise column names
    df = df.rename(columns={
        "Weapon Name": "cm_name",
        "Weapon Class": "weapon_class",
        "Min Range (m)": "min_range_m",
        "Max Range (m)": "max_range_m",
        "Max Engageable Speed (km/h)": "max_engageable_speed_kmh",
        "Max Altitude (m)": "max_altitude_m",
        "Reaction Time (s)": "reaction_time_s",
        "Cost per Engagement (INR)": "cost_per_engagement",
        "Requires RF Link (True/False)": "rf_link_compatible",
        "Inventory Count (India)": "inventory_count",
    })

    # ---- Filter to engagement-capable classes ----
    exclude_pats = config.get("excluded_cm_classes_patterns", [])
    mask = df["weapon_class"].apply(lambda wc: _is_engagement_capable(wc, exclude_pats))
    df = df[mask].copy()

    # ---- NaN imputation: class-median then global-median ----
    numeric_cols = [
        "min_range_m", "max_range_m", "max_engageable_speed_kmh",
        "max_altitude_m", "reaction_time_s", "cost_per_engagement",
    ]
    for col in numeric_cols:
        class_medians = df.groupby("weapon_class")[col].transform("median")
        df[col] = df[col].fillna(class_medians)
        global_median = df[col].median()
        df[col] = df[col].fillna(global_median)

    # Convert RF link compatible to bool
    df["rf_link_compatible"] = df["rf_link_compatible"].astype(bool)

    # Speed in m/s
    df["max_engageable_speed_ms"] = df["max_engageable_speed_kmh"] / 3.6

    # Reset index
    df = df.reset_index(drop=True)
    df.index.name = "cm_id"

    return df


# ─────────────────────────────────────────────
#  Convenience: load everything
# ─────────────────────────────────────────────

def load_all(config_path: str | None = None):
    """Return (config, threats_df, countermeasures_df)."""
    cfg = load_config(config_path)
    threats = load_threats(cfg)
    cms = load_countermeasures(cfg)
    return cfg, threats, cms


if __name__ == "__main__":
    cfg, threats, cms = load_all()
    print(f"Threats loaded:  {len(threats)} rows,  columns: {list(threats.columns)}")
    print(f"CMs loaded:      {len(cms)} rows,  columns: {list(cms.columns)}")
    print(f"\nThreats sample:\n{threats.head(3).to_string()}")
    print(f"\nCMs sample:\n{cms.head(3).to_string()}")
