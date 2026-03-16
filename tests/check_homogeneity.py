import numpy as np
import pandas as pd
from stats_engine.assumption_checker import check_homogeneity

np.random.seed(42)

# Test 1: Equal variance (should pass)
g1 = pd.Series(np.random.normal(0, 1, 50))
g2 = pd.Series(np.random.normal(2, 1, 50))
result = check_homogeneity([g1, g2])
print("Equal variance test:")
print(f"  Passed: {result['passed']}")
print(f"  p-value: {result['p_value']}")
print(f"  Note: {result['note']}")

# Test 2: Unequal variance (should fail)
g1 = pd.Series(np.random.normal(0, 1, 50))
g2 = pd.Series(np.random.normal(0, 10, 50))
result = check_homogeneity([g1, g2])
print("\nUnequal variance test:")
print(f"  Passed: {result['passed']}")
print(f"  p-value: {result['p_value']}")
print(f"  Note: {result['note']}")

# Test 3: Three groups
g1 = pd.Series(np.random.normal(0, 1, 30))
g2 = pd.Series(np.random.normal(1, 1, 30))
g3 = pd.Series(np.random.normal(2, 1, 30))
result = check_homogeneity([g1, g2, g3])
print("\nThree groups test:")
print(f"  n_groups: {result['n_groups']}")
print(f"  group_sizes: {result['group_sizes']}")
print(f"  Passed: {result['passed']}")