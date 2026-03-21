import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.datasets import load_iris
from stats_engine.assumption_checker import build_data_profile
from stats_engine.executor import run_test
from stats_engine.hitl import HITLCheckpoint


# ── 1. Load data ──────────────────────────────────────────────
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]

print("Dataset:")
print(f"  Total samples: {len(df)}")
print(f"  Species: {df['species'].unique()}")
print(f"  Variables: {df.columns.tolist()}")
print()
print("Research Question:")
print("  Does sepal length differ across the 3 iris species?")
print()

# ── 2. Prepare groups ─────────────────────────────────────────
target_var = 'sepal length (cm)'
groups = {
    species: df[df['species'] == species][target_var]
    for species in df['species'].unique()
}

# ── 3. Assumption checking ────────────────────────────────────
profile = build_data_profile(groups, design='independent')

print("Assumption Checks:")
for species, norm_result in profile['normality'].items():
    print(f"\n  {species}:")
    print(f"    Normality test : {norm_result['test']}")
    print(f"    p-value        : {norm_result['p_value']:.4f}")
    print(f"    Passed         : {norm_result['passed']}")
    print(f"    Skewness       : {norm_result['skewness']:.4f}")

print(f"\nVariance Homogeneity:")
print(f"  Test         : {profile['homogeneity']['test']}")
print(f"  p-value      : {profile['homogeneity']['p_value']:.4f}")
print(f"  Equal var    : {profile['homogeneity']['passed']}")

print(f"\nPipeline recommendation : {profile['recommendation']['recommended_test']}")
print(f"Rationale               : {profile['recommendation']['rationale']}")

# ── 4. HITL checkpoint — user confirms or overrides ───────────

checkpoint = HITLCheckpoint(enabled=True)
decision = checkpoint.review(profile, groups)

if decision['test'] is None:
    print("Analysis blocked. Exiting.")
    sys.exit(1)

print(f"\nDecision source : {decision['source']}")
if decision['source'] == 'user_override':
    print(f"Pipeline had recommended : {decision['pipeline_recommendation']}")
    print(f"User chose               : {decision['test']}")

# ── 5. Execute the decided test ───────────────────────────────
result = run_test(decision['test'], groups)

print(f"\nTest      : {result['test']}")
print(f"Statistic : {result['statistic']:.4f}")
print(f"p-value   : {result['p_value']:.4f}")
print(f"Effect size ({result['effect_type']}) : {result['effect_size']:.4f}")

# ── 6. Interpret results ──────────────────────────────────────
if result['p_value'] < 0.05:
    print("\nStatistically significant difference detected")
    print(f"  p = {result['p_value']:.4f} < 0.05")

    if result['effect_size'] > 0.14:
        effect_magnitude = "LARGE"
    elif result['effect_size'] > 0.06:
        effect_magnitude = "MEDIUM"
    else:
        effect_magnitude = "SMALL"

    print(f"  Effect size: {effect_magnitude}")
    print("\nConclusion:")
    print("  Sepal length differs significantly across the 3 iris species.")
else:
    print("\nNo statistically significant difference")
    print(f"  p = {result['p_value']:.4f} > 0.05")

# ── 7. Audit log (shows any human overrides) ──────────────────
overrides = checkpoint.get_audit_log()
if overrides:
    print(f"\nAudit log — {len(overrides)} override(s) recorded:")
    for entry in overrides:
        print(f"  Pipeline said: {entry['pipeline_recommendation']}")
        print(f"  User chose   : {entry['user_choice']}")
        print(f"  Reason       : {entry['reason']}")
else:
    print("\nAudit log: no overrides — pipeline recommendation accepted.")

# Example of automated mode (no HITL prompt)
# Useful for CI, batch processing, or scripted pipelines.
# Uncomment to run without user interaction:
#
# checkpoint_auto = HITLCheckpoint(enabled=False)
# decision_auto = checkpoint_auto.review(profile, groups)
# result_auto = run_test(decision_auto['test'], groups)