import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from stats_engine.assumption_checker import check_normality



np.random.seed(42)

print("="*60)
print("Test 1: Normal data (should pass)")
print("="*60)
normal_data = pd.Series(np.random.normal(0, 1, 100))
result = check_normality(normal_data)
print(f"Test: {result['test']}")
print(f"Passed: {result['passed']}")
print(f"p-value: {result['p_value']}")
print(f"Skewness: {result['skewness']}")
print(f"Kurtosis: {result['kurtosis']}")
print(f"Note: {result['note']}")
print()

print("="*60)
print("Test 2: Exponential data (should fail - right-skewed)")
print("="*60)
skewed_data = pd.Series(np.random.exponential(1, 100))
result = check_normality(skewed_data)
print(f"Test: {result['test']}")
print(f"Passed: {result['passed']}")
print(f"p-value: {result['p_value']}")
print(f"Skewness: {result['skewness']}")
print(f"Kurtosis: {result['kurtosis']}")
print(f"Note: {result['note']}")
print()

print("="*60)
print("Test 3: Large sample (should use D'Agostino-Pearson)")
print("="*60)
large_data = pd.Series(np.random.normal(0, 1, 6000))
result = check_normality(large_data)
print(f"Test: {result['test']}")
print(f"n: {result['n']}")
print(f"Expected: dagostino_pearson")
print()

print("="*60)
print("Test 4: Insufficient data")
print("="*60)
small_data = pd.Series([1, 2, 3])
result = check_normality(small_data)
print(f"Test: {result['test']}")
print(f"Passed: {result['passed']}")
print(f"Note: {result['note']}")