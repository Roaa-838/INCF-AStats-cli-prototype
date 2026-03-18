import numpy as np
import pandas as pd
from stats_engine.executor import run_test

np.random.seed(42)

# Test 1: independent_t
print("="*60)
print("Test 1: Independent t-test")
print("="*60)
groups = {
    'control': pd.Series(np.random.normal(0, 1, 50)),
    'treatment': pd.Series(np.random.normal(0.5, 1, 50))
}
result = run_test('independent_t', groups)
print(f"Success: {result['success']}")
print(f"Statistic: {result['statistic']:.4f}")
print(f"p-value: {result['p_value']:.4f}")
print(f"Cohen's d: {result['effect_size']:.4f}")
print()

# Test 2: welch_t
print("="*60)
print("Test 2: Welch's t-test")
print("="*60)
groups = {
    'control': pd.Series(np.random.normal(0, 1, 50)),
    'treatment': pd.Series(np.random.normal(0, 5, 50))  # Different SD
}
result = run_test('welch_t', groups)
print(f"Success: {result['success']}")
print(f"Statistic: {result['statistic']:.4f}")
print(f"p-value: {result['p_value']:.4f}")
print(f"Note: {result['note']}")
print()

# Test 3: mann_whitney_u
print("="*60)
print("Test 3: Mann-Whitney U test")
print("="*60)
groups = {
    'control': pd.Series(np.random.exponential(1, 50)),
    'treatment': pd.Series(np.random.exponential(2, 50))
}
result = run_test('mann_whitney_u', groups)
print(f"Success: {result['success']}")
print(f"U statistic: {result['statistic']:.4f}")
print(f"p-value: {result['p_value']:.4f}")
print(f"Rank-biserial r: {result['effect_size']:.4f}")