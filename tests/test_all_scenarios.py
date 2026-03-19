import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from stats_engine.assumption_checker import build_data_profile
from data_utils.simulator import get_all_scenarios

scenarios = get_all_scenarios()
total = len(scenarios)
correct_count = 0

for scenario_name, scenario in scenarios.items():
    print(f"Scenario: {scenario['description']}")
    
    df = scenario['df']
    target_col = scenario['target_col']
    group_col = scenario['group_col']
    expected_test = scenario['correct_test']
    
    # Determine design type
    if 'correlation' in scenario_name:
        design = 'correlation'
        groups = {target_col: df[target_col], group_col: df[group_col]}
    else:
        design = 'independent'
        groups = {
            name: df[df[group_col] == name][target_col]
            for name in df[group_col].unique()
        }
    
    # Run decision tree
    profile = build_data_profile(groups, design=design)
    recommended = profile['recommendation']['recommended_test']
    
    # Check if correct
    correct = (recommended == expected_test)
    if correct:
        correct_count += 1
    
    print(f"Expected: {expected_test}")
    print(f"Recommended: {recommended}")
    print(f"{'CORRECT' if correct else 'WRONG'}")
    print(f"Rationale: {profile['recommendation']['rationale']}")
    print()

accuracy = (correct_count / total) * 100
print(f"RESULTS: {correct_count}/{total} ({accuracy:.1f}%)")

if accuracy == 100:
    print("Decision tree is working correctly")
elif accuracy >= 70:
    print("PASSING - Above 70% threshold")
else:
    print("FAILING - Below 70% threshold - debug needed")