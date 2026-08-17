"""
pipeline.py — Full orchestration: threat → ranked recommendation.

Ties together all stages:
    1. Load & clean data        (data_loader)
    2. Compute TPS              (priority_score)
    3. Feasibility filter       (feasibility)
    4. Kill probability (Pk)    (kill_probability)
    5. Decision engine          (decision_engine)

Provides both batch (all threats) and single-threat query modes.
"""

import pathlib
import sys
import json
import pandas as pd

# Ensure src/ is on the path when running from project root
_SRC = pathlib.Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_loader import load_all
from priority_score import compute_tps
from feasibility import filter_feasible
from kill_probability import compute_all_pk
from decision_engine import decide_for_threat


def run_pipeline(config_path: str | None = None) -> tuple[list[dict], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Execute the full pipeline.

    Returns
    -------
    (results, threats_ranked, cms, feasible_pk)

    results : list of dicts — one per threat, ordered by TPS rank,
              each containing the §5.4 output structure.
    threats_ranked : DataFrame — threats with TPS columns
    cms            : DataFrame — cleaned counter-measures
    feasible_pk    : DataFrame — all feasible pairs with Pk
    """
    # --- Stage 0: Load ---
    cfg, threats, cms = load_all(config_path)

    # --- Stage 1: TPS ---
    threats_ranked = compute_tps(threats, cfg)

    # --- Stage 2.1: Feasibility ---
    feasible_pairs = filter_feasible(threats_ranked, cms)

    if feasible_pairs.empty:
        # Edge case: nothing is feasible (shouldn't happen with real data)
        results = []
        for _, t in threats_ranked.iterrows():
            results.append({
                "threat_name": t["threat_name"],
                "threat_class": t["threat_class"],
                "TPS": round(t["TPS"], 4),
                "TPS_rank": int(t["TPS_rank"]),
                "recommended_option": None,
                "combination_option": None,
                "alternatives": [],
            })
        return results, threats_ranked, cms, feasible_pairs

    # --- Stage 2.2: Kill probability ---
    feasible_pk = compute_all_pk(feasible_pairs, threats_ranked, cms, cfg)

    # --- Stage 3: Decision engine (per threat) ---
    results = []
    for _, t in threats_ranked.iterrows():
        t_idx = _  # row index in threats_ranked
        t_options = feasible_pk[feasible_pk["threat_idx"] == t_idx].copy()

        decision = decide_for_threat(t_options, cfg)

        entry = {
            "threat_name":      t["threat_name"],
            "threat_class":     t["threat_class"],
            "TPS":              round(t["TPS"], 4),
            "TPS_rank":         int(t["TPS_rank"]),
            "max_speed_kmh":    t["max_speed_kmh"],
            "max_altitude_m":   t["max_altitude_m"],
            "engagement_range_km": t["engagement_range_km"],
            "payload_kg":       t["payload_kg"],
            "rf_link":          bool(t["rf_link"]),
            "num_feasible_cms": len(t_options),
            **decision,
        }
        results.append(entry)

    return results, threats_ranked, cms, feasible_pk


def query_single_threat(
    threat_name: str,
    config_path: str | None = None,
) -> dict | None:
    """
    Run the pipeline and return the result for a single named threat.
    Returns None if the threat is not found.
    """
    results, *_ = run_pipeline(config_path)
    for r in results:
        if r["threat_name"].lower() == threat_name.lower():
            return r
    return None


if __name__ == "__main__":
    print("Running full pipeline...")
    results, threats, cms, feas = run_pipeline()

    # Summary stats
    n_threats = len(results)
    n_with_rec = sum(1 for r in results if r["recommended_option"] is not None)
    n_combo = sum(
        1 for r in results
        if r["recommended_option"] and r["recommended_option"]["type"] == "combination"
    )
    bands = {}
    for r in results:
        if r["recommended_option"]:
            b = r["recommended_option"]["decision_band_used"]
            bands[b] = bands.get(b, 0) + 1

    print(f"\n{'='*60}")
    print(f"  Pipeline complete")
    print(f"  Threats processed       : {n_threats}")
    print(f"  With recommendation     : {n_with_rec}")
    print(f"  No feasible CMs         : {n_threats - n_with_rec}")
    print(f"  Combination-mode picks  : {n_combo}")
    print(f"  Decision band breakdown : {bands}")
    print(f"{'='*60}")

    # Show top-5
    print("\nTop-5 threats by TPS:")
    for r in results[:5]:
        rec = r["recommended_option"]
        if rec:
            print(
                f"  #{r['TPS_rank']:>4d}  TPS={r['TPS']:.4f}  "
                f"{r['threat_name']:<40s}  ->  {rec['counter_measures'][0]:<35s}  "
                f"Pk={rec['Pk']:.1f}%  cost=Rs{rec['total_cost']:>12,.0f}  "
                f"[{rec['decision_band_used']}]"
            )
        else:
            print(
                f"  #{r['TPS_rank']:>4d}  TPS={r['TPS']:.4f}  "
                f"{r['threat_name']:<40s}  ->  NO FEASIBLE CM"
            )
