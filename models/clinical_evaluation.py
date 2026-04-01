#!/usr/bin/env python3
"""
Clinical scenario-based evaluation of SepsisGuard.
Tests XGBoost predictions against expected clinical outcomes.
"""

from pathlib import Path
import numpy as np
import joblib

project_root = Path(__file__).resolve().parent.parent
xgb_data = joblib.load(project_root / "models" / "xgboost_model.pkl")
model = xgb_data["model"]
features = xgb_data["feature_cols"]
# Use clinically meaningful thresholds instead of percentile-based
t_low = 0.15   # below 15% = LOW risk
t_high = 0.35  # above 35% = HIGH risk

# Clinical scenarios with expected outcomes
scenarios = [
    {
        "name": "Healthy adult",
        "expected": "LOW",
        "vitals": {"HR": 72, "O2Sat": 98, "Temp": 36.8, "SBP": 120, "DBP": 75, "MAP": 90,
                   "Resp": 14, "Lactate": 0.8, "WBC": 7, "Creatinine": 0.9,
                   "Bilirubin_total": 0.5, "Platelets": 250, "Age": 35, "Gender": 0,
                   "HospAdmTime": 0, "ICULOS": 12},
    },
    {
        "name": "Elderly post-surgery (stable)",
        "expected": "LOW-MODERATE",
        "vitals": {"HR": 88, "O2Sat": 95, "Temp": 37.2, "SBP": 110, "DBP": 65, "MAP": 80,
                   "Resp": 18, "Lactate": 1.5, "WBC": 10, "Creatinine": 1.2,
                   "Bilirubin_total": 0.8, "Platelets": 180, "Age": 72, "Gender": 1,
                   "HospAdmTime": -2, "ICULOS": 24},
    },
    {
        "name": "Early sepsis — tachycardia + fever",
        "expected": "MODERATE",
        "vitals": {"HR": 105, "O2Sat": 94, "Temp": 38.8, "SBP": 100, "DBP": 58, "MAP": 72,
                   "Resp": 22, "Lactate": 2.5, "WBC": 15, "Creatinine": 1.4,
                   "Bilirubin_total": 1.2, "Platelets": 140, "Age": 60, "Gender": 0,
                   "HospAdmTime": -5, "ICULOS": 48},
    },
    {
        "name": "Sepsis with organ dysfunction",
        "expected": "HIGH",
        "vitals": {"HR": 118, "O2Sat": 90, "Temp": 39.2, "SBP": 88, "DBP": 48, "MAP": 60,
                   "Resp": 28, "Lactate": 4.2, "WBC": 20, "Creatinine": 2.5,
                   "Bilirubin_total": 2.8, "Platelets": 90, "Age": 68, "Gender": 1,
                   "HospAdmTime": -8, "ICULOS": 72},
    },
    {
        "name": "Septic shock — critical",
        "expected": "HIGH",
        "vitals": {"HR": 140, "O2Sat": 84, "Temp": 40.1, "SBP": 70, "DBP": 35, "MAP": 45,
                   "Resp": 34, "Lactate": 7.5, "WBC": 28, "Creatinine": 4.0,
                   "Bilirubin_total": 5.0, "Platelets": 40, "Age": 75, "Gender": 0,
                   "HospAdmTime": -12, "ICULOS": 96},
    },
    {
        "name": "Hypothermic sepsis (atypical)",
        "expected": "MODERATE-HIGH",
        "vitals": {"HR": 110, "O2Sat": 92, "Temp": 35.2, "SBP": 95, "DBP": 50, "MAP": 65,
                   "Resp": 24, "Lactate": 3.8, "WBC": 3, "Creatinine": 2.0,
                   "Bilirubin_total": 1.5, "Platelets": 100, "Age": 80, "Gender": 1,
                   "HospAdmTime": -6, "ICULOS": 60},
    },
]

print("=" * 80)
print("CLINICAL SCENARIO EVALUATION")
print(f"Risk thresholds: LOW < {t_low:.3f} < MODERATE < {t_high:.3f} < HIGH")
print("=" * 80)

results = []
for s in scenarios:
    v = s["vitals"]
    # Add temporal features (single-point: delta=0, roll3=current)
    for col in ["HR", "O2Sat", "Temp", "MAP", "Resp", "SBP", "DBP"]:
        v[f"{col}_delta"] = 0.0
        v[f"{col}_roll3"] = v.get(col, 0)

    X = np.array([[v.get(c, 0) for c in features]])
    prob = float(model.predict_proba(X)[0][1])
    level = "HIGH" if prob >= t_high else "MODERATE" if prob >= t_low else "LOW"

    match = "PASS" if s["expected"] in level or level in s["expected"] else "CHECK"

    print(f"\n  Scenario: {s['name']}")
    print(f"  Key vitals: HR={v['HR']}, MAP={v['MAP']}, Temp={v['Temp']}, Lactate={v['Lactate']}, WBC={v['WBC']}")
    print(f"  Risk score: {prob:.1%}  |  Level: {level}  |  Expected: {s['expected']}  |  {match}")

    results.append({"scenario": s["name"], "score": prob, "level": level,
                     "expected": s["expected"], "match": match})

print(f"\n{'='*80}")
passed = sum(1 for r in results if r["match"] == "PASS")
print(f"SUMMARY: {passed}/{len(results)} scenarios matched expected risk level")
print("=" * 80)
