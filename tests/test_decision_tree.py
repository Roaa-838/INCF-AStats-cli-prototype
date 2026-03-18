import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from stats_engine.assumption_checker import build_data_profile


np.random.seed(42)

# Test 1: Two normal groups, equal variance → independent_t
print("="*60)
print("Test 1: Two normal groups, equal variance")
print("="*60)
groups = {
    'control': pd.Series(np.random.normal(0, 1, 50)),
    'treatment': pd.Series(np.random.normal(0.5, 1, 50))
}
profile = build_data_profile(groups, design='independent')
print(f"Recommended test: {profile['recommendation']['recommended_test']}")
print(f"Rationale: {profile['recommendation']['rationale']}")
print()

# Test 2: Two normal groups, unequal variance → welch_t
print("="*60)
print("Test 2: Two normal groups, unequal variance")
print("="*60)
groups = {
    'control': pd.Series(np.random.normal(0, 1, 50)),
    'treatment': pd.Series(np.random.normal(0, 5, 50))  # SD=5 (much larger)
}
profile = build_data_profile(groups, design='independent')
print(f"Recommended test: {profile['recommendation']['recommended_test']}")
print(f"Rationale: {profile['recommendation']['rationale']}")
print()

# Test 3: Two non-normal groups → mann_whitney_u
print("="*60)
print("Test 3: Two non-normal groups (exponential)")
print("="*60)
groups = {
    'control': pd.Series(np.random.exponential(1, 50)),
    'treatment': pd.Series(np.random.exponential(2, 50))
}
profile = build_data_profile(groups, design='independent')
print(f"Recommended test: {profile['recommendation']['recommended_test']}")
print(f"Rationale: {profile['recommendation']['rationale']}")
print()

# Test 4: Three normal groups, equal variance → one_way_anova
print("="*60)
print("Test 4: Three normal groups, equal variance")
print("="*60)
groups = {
    'A': pd.Series(np.random.normal(0, 1, 30)),
    'B': pd.Series(np.random.normal(0.5, 1, 30)),
    'C': pd.Series(np.random.normal(1, 1, 30))
}
profile = build_data_profile(groups, design='independent')
print(f"Recommended test: {profile['recommendation']['recommended_test']}")
print(f"Rationale: {profile['recommendation']['rationale']}")