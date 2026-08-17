"""Verification test script for CAADS audit changes."""
import numpy as np
import pandas as pd
from src.data_loader import load_all
from src.priority_score import compute_tps
from src.feasibility import filter_feasible
from src.kill_probability import compute_all_pk, compute_pk
from src.pipeline import run_pipeline


def run_all_tests():
    cfg, threats, cms = load_all()
    ranked = compute_tps(threats, cfg)
    feas = filter_feasible(ranked, cms)
    feas_pk = compute_all_pk(feas, ranked, cms, cfg)

    print("=== Pk STATISTICS ===")
    print(feas_pk[["base_pk", "range_fit", "reaction_margin", "altitude_fit", "Pk_pct"]].describe().to_string())
    print()

    # Test 1: scalar vs vectorized (first 50 pairs)
    mismatches = 0
    for i in range(min(50, len(feas_pk))):
        row = feas_pk.iloc[i]
        t = ranked.iloc[int(row["threat_idx"])]
        c = cms.iloc[int(row["cm_idx"])]
        s = compute_pk(t, c, cfg)
        if abs(s["Pk"] - row["Pk"]) > 0.001:
            mismatches += 1
            print(f"  MISMATCH {i}: scalar={s['Pk']:.6f} vec={row['Pk']:.6f}")
            print(f"    base: s={s['base_pk']:.6f} v={row['base_pk']:.6f}")
            print(f"    rf:   s={s['range_fit']:.6f} v={row['range_fit']:.6f}")
            print(f"    rm:   s={s['reaction_margin']:.6f} v={row['reaction_margin']:.6f}")
            print(f"    af:   s={s['altitude_fit']:.6f} v={row['altitude_fit']:.6f}")
    status = "PASS" if mismatches == 0 else f"{mismatches} FAILURES"
    print(f"Test 1 - Scalar vs Vectorized (50 pairs): {status}")

    # Test 2: Pk = base * range * reaction * altitude
    product = feas_pk["base_pk"] * feas_pk["range_fit"] * feas_pk["reaction_margin"] * feas_pk["altitude_fit"]
    diff = np.abs(feas_pk["Pk"] - product)
    status = "PASS" if diff.max() < 0.001 else "FAIL"
    print(f"Test 2 - Pk product consistency: max_diff={diff.max():.10f} {status}")

    # Test 3: All factors in [0, 1]
    for col in ["base_pk", "range_fit", "reaction_margin", "altitude_fit", "Pk"]:
        vals = feas_pk[col]
        ok = (vals >= 0).all() and (vals <= 1).all()
        status = "PASS" if ok else "FAIL"
        print(f"Test 3 - {col} in [0,1]: {status} (min={vals.min():.6f}, max={vals.max():.6f})")

    # Test 4: Config parameter verification
    de = cfg["decision_engine"]
    w = cfg["synthetic_pk_weights"]
    assert de["CRITICAL_PK"] == 70, f"CRITICAL_PK should be 70, got {de['CRITICAL_PK']}"
    assert de.get("COMBO_TRIGGER_PK") == 60, f"COMBO_TRIGGER_PK should be 60"
    assert de["SMALL_GAP"] == 5
    assert de["LARGE_GAP"] == 15
    assert de["TARGET_PK"] == 85
    assert "range_coverage" not in w, "range_coverage should NOT be in base_pk weights"
    weight_sum = w["weapon_class_tier"] + w["speed_ratio"] + w["altitude_ratio"]
    assert abs(weight_sum - 1.0) < 0.001, f"Weights must sum to 1.0, got {weight_sum}"
    print(f"Test 4 - Config parameters: PASS")

    # Test 5: Band distribution with new thresholds
    results, _, _, _ = run_pipeline()
    bands = {}
    for r in results:
        if r.get("recommended_option"):
            b = r["recommended_option"]["decision_band_used"]
            bands[b] = bands.get(b, 0) + 1
    total = len(results)
    matched = sum(1 for r in results if r.get("recommended_option"))
    print(f"\nTest 5 - Band Distribution:")
    print(f"  Total: {total}, Matched: {matched}, Unmatched: {total - matched}")
    for b, c in sorted(bands.items()):
        print(f"  {b}: {c}")

    # Test 6: Debug fields present
    sample = feas_pk.iloc[0]
    for field in ["speed_ratio", "altitude_ratio", "weapon_class_tier"]:
        assert field in feas_pk.columns, f"Missing debug field: {field}"
    print(f"Test 6 - Debug fields present: PASS")

    print("\n=== ALL TESTS COMPLETE ===")


if __name__ == "__main__":
    run_all_tests()
