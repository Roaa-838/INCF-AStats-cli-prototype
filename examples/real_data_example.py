import numpy as np
import pandas as pd

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from sklearn.datasets import load_iris
from stats_engine.assumption_checker import build_data_profile
from stats_engine.executor import run_test


# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]


print(f"  Total samples: {len(df)}")
print(f"  Species: {df['species'].unique()}")
print(f"  Variables: {df.columns.tolist()}")
print()

print("Research Question:")
print("  Does sepal length differ across the 3 iris species?")
print()

# Prepare data
target_var = 'sepal length (cm)'
groups = {
    species: df[df['species'] == species][target_var]
    for species in df['species'].unique()
}

# Build profile
profile = build_data_profile(groups, design='independent')

for species, norm_result in profile['normality'].items():
    print(f"\n{species}:")
    print(f"  Normality test: {norm_result['test']}")
    print(f"  p-value: {norm_result['p_value']:.4f}")
    print(f"  Passed: {norm_result['passed']}")
    print(f"  Skewness: {norm_result['skewness']:.4f}")

print(f"\nVariance Homogeneity:")
print(f"  Test: {profile['homogeneity']['test']}")
print(f"  p-value: {profile['homogeneity']['p_value']:.4f}")
print(f"  Equal variance: {profile['homogeneity']['passed']}")


recommendation = profile['recommendation']
print(f"Recommended test: {recommendation['recommended_test']}")
print(f"Rationale: {recommendation['rationale']}")
print()


result = run_test(recommendation['recommended_test'], groups)

print(f"\nTest: {result['test']}")
print(f"Statistic: {result['statistic']:.4f}")
print(f"p-value: {result['p_value']:.4f}")
print(f"Effect size ({result['effect_type']}): {result['effect_size']:.4f}")



if result['p_value'] < 0.05:
    print("Statistically significant difference detected")
    print(f"  p = {result['p_value']:.4f} < 0.05")
    
    if result['effect_size'] > 0.14:
        effect_magnitude = "LARGE"
    elif result['effect_size'] > 0.06:
        effect_magnitude = "MEDIUM"
    else:
        effect_magnitude = "SMALL"
    
    print(f"  Effect size: {effect_magnitude}")
    print()
    print("Conclusion:")
    print("  Sepal length differs significantly across the 3 iris species.")
else:
    print("No statistically significant difference")
    print(f"  p = {result['p_value']:.4f} > 0.05")

