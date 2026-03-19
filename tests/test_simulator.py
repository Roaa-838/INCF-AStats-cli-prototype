import numpy as np
from data_utils.simulator import get_two_group_scenarios
from stats_engine.assumption_checker import build_data_profile


scenarios = get_two_group_scenarios()

for scenario_name, scenario in scenarios.items():

    print(f"Scenario: {scenario['description']}")
    
    df = scenario['df']
    target_col = scenario['target_col']
    group_col = scenario['group_col']
    expected_test = scenario['correct_test']
    
    # Build groups dictionary
    groups = {
        name: df[df[group_col] == name][target_col]
        for name in df[group_col].unique()
    }
    
    # Run decision tree
    profile = build_data_profile(groups, design='independent')
    recommended = profile['recommendation']['recommended_test']
    
    # Check if correct
    correct = (recommended == expected_test)
    
    print(f"Expected test: {expected_test}")
    print(f"Recommended test: {recommended}")
    print(f"CORRECT" if correct else "WRONG")
    print(f"Rationale: {profile['recommendation']['rationale']}")
    print()

