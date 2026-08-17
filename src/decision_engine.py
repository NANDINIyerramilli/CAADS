"""
decision_engine.py — Stage 3: Cost-Aware Decision Engine.

Implements four decision bands + Combination Mode:

  Band 1 — COST TIEBREAK
      gap <= SMALL_GAP  and  Pk_best >= CRITICAL_PK
      → pick cheapest among options within SMALL_GAP of best Pk

  Band 2 — WEIGHTED BLEND
      SMALL_GAP < gap <= LARGE_GAP  and  Pk_best >= CRITICAL_PK
      → Value = α·norm(Pk) − β·norm(cost);  pick highest

  Band 3 — PK PRIORITY
      gap > LARGE_GAP
      → best Pk wins outright

  Band 4 — LOW CONFIDENCE (Pk_best < CRITICAL_PK)
      → Pk wins single-shot (cost is irrelevant)
      → If Pk_best < COMBO_TRIGGER_PK, also search 2- then 3-combos
        for P_combined >= TARGET_PK at lowest cost

Parameters (from config.yaml):
    SMALL_GAP      = 5   (percentage points)
    LARGE_GAP      = 15  (percentage points)
    CRITICAL_PK    = 70  (%) — below this, ignore cost for single selection
    COMBO_TRIGGER_PK = 60  (%) — below this, also search combinations
    TARGET_PK      = 85  (%) — target for combination mode
    alpha          = 0.7
    beta           = 0.3

Every recommendation includes a human-readable justification_text.
"""

from itertools import combinations
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
#  Normalisation helper
# ─────────────────────────────────────────────

def _norm_series(values: list[float]) -> list[float]:
    """Min-max normalise a list of floats to [0,1].  Constant → 0.5."""
    mn, mx = min(values), max(values)
    if mx == mn:
        return [0.5] * len(values)
    return [(v - mn) / (mx - mn) for v in values]


# ─────────────────────────────────────────────
#  Combination Mode (§15)
# ─────────────────────────────────────────────

def _search_combinations(
    options: list[dict],
    target_pk: float,
    max_combo: int = 3,
) -> dict | None:
    """
    Search 2-item then 3-item combos for P_combined >= target_pk.

    Each option dict must have: cm_name, cm_idx, Pk (0–1), cost, inventory_count.

    Independence assumption: P_combined = 1 − Π(1 − Pk_i)
    This is a SYNTHETIC modeling assumption — real-world weapon independence
    is not guaranteed and would require operational correlation data.

    Returns best combo dict or None.
    """
    # Cap candidate pool to top 15 options to avoid combinatorial explosion
    options = options[:15]
    if len(options) < 2:
        return None

    best_target_combo = None
    best_overall_combo = None

    for size in range(2, min(max_combo + 1, len(options) + 1)):
        for combo in combinations(range(len(options)), size):
            cm_ids = [options[i]["cm_idx"] for i in combo]
            if len(set(cm_ids)) != len(cm_ids):
                continue   # duplicate CM in combo

            # Respect inventory: each CM used at most inventory_count times
            inv_ok = all(options[i]["inventory_count"] >= 1 for i in combo)
            if not inv_ok:
                continue

            pks   = [options[i]["Pk"] for i in combo]
            costs = [options[i]["cost"] for i in combo]

            p_combined = 1.0 - float(np.prod([1.0 - pk for pk in pks]))
            total_cost = sum(costs)
            pk_pct = round(p_combined * 100, 2)

            item = {
                "indices": combo,
                "counter_measures": [options[i]["cm_name"] for i in combo],
                "cm_indices": cm_ids,
                "individual_pks": [round(pk * 100, 2) for pk in pks],
                "Pk": pk_pct,
                "total_cost": total_cost,
                "combo_size": size,
            }

            if pk_pct >= target_pk:
                if best_target_combo is None or total_cost < best_target_combo["total_cost"]:
                    best_target_combo = item

            if best_overall_combo is None or pk_pct > best_overall_combo["Pk"]:
                best_overall_combo = item

    if best_target_combo:
        best_target_combo["meets_target"] = True
        return best_target_combo
    elif best_overall_combo:
        best_overall_combo["meets_target"] = False
        return best_overall_combo

    return None


# ─────────────────────────────────────────────
#  Core decision logic
# ─────────────────────────────────────────────

def decide_for_threat(
    threat_options: pd.DataFrame,
    config: dict,
) -> dict:
    """
    Apply the cost-aware decision bands for one threat.

    Parameters
    ----------
    threat_options : DataFrame — rows of feasible CMs for this threat.
        Required columns: cm_name, cm_idx, Pk_pct, cost_per_engagement,
                          inventory_count, cm_weapon_class, Pk, base_pk,
                          speed_ratio, altitude_ratio, weapon_class_tier,
                          range_fit, reaction_margin, altitude_fit
    config : dict — must contain 'decision_engine' section.

    Returns
    -------
    dict with keys: recommended_option, combination_option, alternatives
    """
    de = config["decision_engine"]
    SMALL_GAP        = de["SMALL_GAP"]
    LARGE_GAP        = de["LARGE_GAP"]
    CRITICAL_PK      = de["CRITICAL_PK"]
    COMBO_TRIGGER_PK = de.get("COMBO_TRIGGER_PK", 60)
    TARGET_PK        = de["TARGET_PK"]
    alpha            = de["alpha"]
    beta             = de["beta"]
    max_combo        = de.get("max_combo_size", 3)

    if threat_options.empty:
        return {
            "recommended_option": None,
            "alternatives": [],
            "decision_band_used": "no_feasible_options",
            "justification_text": "No counter-measure passed the feasibility filter for this threat.",
        }

    # Sort by Pk descending
    opts = threat_options.sort_values("Pk_pct", ascending=False).reset_index(drop=True)

    # Build alternatives list (for debuggability — spec §12)
    alternatives = []
    for _, r in opts.iterrows():
        alternatives.append({
            "cm_name":           r["cm_name"],
            "cm_idx":            int(r["cm_idx"]),
            "weapon_class":      r["cm_weapon_class"],
            "Pk":                r["Pk_pct"],
            "cost":              r["cost_per_engagement"],
            "base_pk":           round(r["base_pk"] * 100, 2),
            "speed_ratio":       round(r.get("speed_ratio", 0) * 100, 2),
            "altitude_ratio":    round(r.get("altitude_ratio", 0) * 100, 2),
            "weapon_class_tier": round(r.get("weapon_class_tier", 0) * 100, 2),
            "range_fit":         round(r["range_fit"] * 100, 2),
            "reaction_margin":   round(r["reaction_margin"] * 100, 2),
            "altitude_fit":      round(r["altitude_fit"] * 100, 2),
            "inventory":         int(r["inventory_count"]),
        })

    best_row = opts.iloc[0]
    pk_best = best_row["Pk_pct"]

    # ------- Only one feasible option -------
    if len(opts) == 1:
        band = "single_option"
        justification = (
            f"Only one counter-measure is feasible: {best_row['cm_name']} "
            f"with Pk={pk_best:.1f}%."
        )
        rec = {
            "type": "single",
            "counter_measures": [best_row["cm_name"]],
            "cm_indices": [int(best_row["cm_idx"])],
            "Pk": pk_best,
            "total_cost": best_row["cost_per_engagement"],
            "decision_band_used": band,
            "justification_text": justification,
        }
        # Low-confidence warning
        if pk_best < CRITICAL_PK:
            rec["decision_band_used"] = "low_confidence"
            rec["justification_text"] = (
                f"Only option is {best_row['cm_name']} at Pk={pk_best:.1f}% "
                f"(below CRITICAL_PK={CRITICAL_PK}%). "
                f"No other CMs are feasible for combination."
            )

        return {
            "recommended_option": rec,
            "combination_option": None,
            "alternatives": alternatives,
        }

    pk_second = opts.iloc[1]["Pk_pct"]
    gap = pk_best - pk_second

    # ------- Case 4: LOW CONFIDENCE (Pk_best < CRITICAL_PK) -------
    # Pk wins for single selection — cost is irrelevant
    # If Pk_best < COMBO_TRIGGER_PK, also search combinations
    if pk_best < CRITICAL_PK:
        band = "low_confidence"
        selected = best_row  # best Pk wins (ignore cost)

        justification = (
            f"Best single Pk={pk_best:.1f}% is below CRITICAL_PK={CRITICAL_PK}% "
            f"— cost is irrelevant in this zone. "
            f"Selected {best_row['cm_name']} as best single option. "
        )

        # Search combinations only if below COMBO_TRIGGER_PK
        combo = None
        if pk_best < COMBO_TRIGGER_PK:
            combo_opts = [
                {
                    "cm_name": r["cm_name"],
                    "cm_idx": int(r["cm_idx"]),
                    "Pk": r["Pk"],        # 0–1 scale
                    "cost": r["cost_per_engagement"],
                    "inventory_count": int(r["inventory_count"]),
                }
                for _, r in opts.iterrows()
            ]
            combo = _search_combinations(combo_opts, TARGET_PK, max_combo)

            if combo:
                if combo["meets_target"]:
                    justification += (
                        f"Combination mode (triggered at Pk<{COMBO_TRIGGER_PK}%) "
                        f"found a {combo['combo_size']}-weapon salvo "
                        f"({', '.join(combo['counter_measures'])}) achieving "
                        f"P_combined={combo['Pk']:.1f}% ≥ TARGET_PK={TARGET_PK}% "
                        f"at total cost ₹{combo['total_cost']:,.0f}."
                    )
                else:
                    justification += (
                        f"⚠ WARNING: Best achievable combo "
                        f"({', '.join(combo['counter_measures'])}) reaches only "
                        f"P_combined={combo['Pk']:.1f}% — below TARGET_PK={TARGET_PK}%. "
                        f"Target Pk NOT achievable with current inventory."
                    )

        rec = {
            "type": "combination" if combo else "single",
            "counter_measures": combo["counter_measures"] if combo else [best_row["cm_name"]],
            "cm_indices": combo["cm_indices"] if combo else [int(best_row["cm_idx"])],
            "Pk": combo["Pk"] if combo else pk_best,
            "total_cost": combo["total_cost"] if combo else best_row["cost_per_engagement"],
            "decision_band_used": band,
            "justification_text": justification,
        }

        # Also provide the single-best as a separate field
        single_rec = {
            "type": "single",
            "counter_measures": [best_row["cm_name"]],
            "cm_indices": [int(best_row["cm_idx"])],
            "Pk": pk_best,
            "total_cost": best_row["cost_per_engagement"],
        }

        return {
            "recommended_option": rec,
            "single_best": single_rec,
            "combination_option": combo,
            "alternatives": alternatives,
        }

    # ------- Band 1: COST TIEBREAK (gap <= SMALL_GAP, Pk >= CRITICAL) -------
    if gap <= SMALL_GAP:
        band = "cost_tiebreak"
        # All options within SMALL_GAP of best Pk
        threshold = pk_best - SMALL_GAP
        near_best = opts[opts["Pk_pct"] >= threshold]
        cheapest = near_best.loc[near_best["cost_per_engagement"].idxmin()]

        justification = (
            f"Gap={gap:.1f}pp ≤ SMALL_GAP={SMALL_GAP}pp and Pk_best={pk_best:.1f}% "
            f"≥ CRITICAL_PK={CRITICAL_PK}%. "
            f"Effectiveness is statistically indistinguishable. "
            f"Selected {cheapest['cm_name']} (Pk={cheapest['Pk_pct']:.1f}%, "
            f"cost=₹{cheapest['cost_per_engagement']:,.0f}) as cheapest "
            f"among {len(near_best)} options within {SMALL_GAP}pp of best."
        )

        rec = {
            "type": "single",
            "counter_measures": [cheapest["cm_name"]],
            "cm_indices": [int(cheapest["cm_idx"])],
            "Pk": cheapest["Pk_pct"],
            "total_cost": cheapest["cost_per_engagement"],
            "decision_band_used": band,
            "justification_text": justification,
        }

        return {
            "recommended_option": rec,
            "combination_option": None,
            "alternatives": alternatives,
        }

    # ------- Band 2: WEIGHTED BLEND (SMALL < gap <= LARGE, Pk >= CRITICAL) -------
    if gap <= LARGE_GAP:
        band = "weighted"
        pks   = opts["Pk_pct"].tolist()
        costs = opts["cost_per_engagement"].tolist()

        norm_pk   = _norm_series(pks)
        norm_cost = _norm_series(costs)

        values = [alpha * npk - beta * nc for npk, nc in zip(norm_pk, norm_cost)]
        best_val_idx = int(np.argmax(values))
        selected = opts.iloc[best_val_idx]

        justification = (
            f"Gap={gap:.1f}pp is between SMALL_GAP={SMALL_GAP}pp and "
            f"LARGE_GAP={LARGE_GAP}pp — moderate gap. "
            f"Applied weighted blend (α={alpha}, β={beta}). "
            f"Selected {selected['cm_name']} with Value={values[best_val_idx]:.3f} "
            f"(Pk={selected['Pk_pct']:.1f}%, cost=₹{selected['cost_per_engagement']:,.0f})."
        )

        rec = {
            "type": "single",
            "counter_measures": [selected["cm_name"]],
            "cm_indices": [int(selected["cm_idx"])],
            "Pk": selected["Pk_pct"],
            "total_cost": selected["cost_per_engagement"],
            "decision_band_used": band,
            "justification_text": justification,
        }

        return {
            "recommended_option": rec,
            "combination_option": None,
            "alternatives": alternatives,
        }

    # ------- Band 3: PK PRIORITY (gap > LARGE_GAP) -------
    band = "pk_priority"
    justification = (
        f"Gap={gap:.1f}pp > LARGE_GAP={LARGE_GAP}pp — effectiveness gap is too "
        f"large to trade away for cost savings. "
        f"Selected {best_row['cm_name']} (Pk={pk_best:.1f}%, "
        f"cost=₹{best_row['cost_per_engagement']:,.0f}) outright."
    )

    rec = {
        "type": "single",
        "counter_measures": [best_row["cm_name"]],
        "cm_indices": [int(best_row["cm_idx"])],
        "Pk": pk_best,
        "total_cost": best_row["cost_per_engagement"],
        "decision_band_used": band,
        "justification_text": justification,
    }

    return {
        "recommended_option": rec,
        "combination_option": None,
        "alternatives": alternatives,
    }
