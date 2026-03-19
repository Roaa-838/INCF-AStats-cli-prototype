import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_utils.simulator import get_all_scenarios
from stats_engine.assumption_checker import build_data_profile
from stats_engine.executor import run_test


scenarios = get_all_scenarios()

# Test one scenario end-to-end
scenario = scenarios['two_groups_normal_equal_variance']
df = scenario['df']
target_col = scenario['target_col']
group_col = scenario['group_col']
print("Testing scenario:", scenario['description'])
print()


groups = {
    name: df[df[group_col] == name][target_col]
    for name in df[group_col].unique()
}
profile = build_data_profile(groups, design='independent')
print("Step 1: Build data profile")
print(f"  Recommended test: {profile['recommendation']['recommended_test']}")
print(f"  Rationale: {profile['recommendation']['rationale']}")
print()


test_name = profile['recommendation']['recommended_test']
result = run_test(test_name, groups)
print("Step 2: Execute test")
print(f"  Test: {result['test']}")
print(f"  Statistic: {result['statistic']:.4f}")
print(f"  p-value: {result['p_value']:.4f}")
print(f"  Effect size ({result['effect_type']}): {result['effect_size']:.4f}")
print()

# (manual for now)
print("Step 3: Interpret results")
if result['p_value'] < 0.05:
    print(f"  Significant difference (p={result['p_value']:.4f})")
    if abs(result['effect_size']) < 0.5:
        magnitude = "small"
    elif abs(result['effect_size']) < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"
    print(f"  Effect size: {magnitude}")
else:
    print(f"  No significant difference (p={result['p_value']:.4f})")

