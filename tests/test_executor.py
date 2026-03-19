import numpy as np
import pandas as pd
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stats_engine.executor import run_test

np.random.seed(42)

print("Test 1: Independent t-test")
groups = {
    'control': pd.Series(np.random.normal(0, 1, 50)),
    'treatment': pd.Series(np.random.normal(0.5, 1, 50))
}
result = run_test('independent_t', groups)
print(f"Success: {result['success']}")
print(f"t-statistic: {result['statistic']:.4f}")
print(f"p-value: {result['p_value']:.4f}")
print(f"Cohen's d: {result['effect_size']:.4f}")
print(f"Degrees of freedom: {result['degrees_of_freedom']}")
print()


print("Test 2: Welch's t-test (unequal variance)")
groups = {
    'control': pd.Series(np.random.normal(0, 1, 50)),
    'treatment': pd.Series(np.random.normal(0, 5, 50))
}
result = run_test('welch_t', groups)
print(f"Success: {result['success']}")
print(f"t-statistic: {result['statistic']:.4f}")
print(f"p-value: {result['p_value']:.4f}")
print(f"Cohen's d: {result['effect_size']:.4f}")
print(f"Note: {result['note']}")
print()


print("Test 3: Mann-Whitney U test")
groups = {
    'control': pd.Series(np.random.exponential(1, 50)),
    'treatment': pd.Series(np.random.exponential(2, 50))
}
result = run_test('mann_whitney_u', groups)
print(f"Success: {result['success']}")
print(f"U statistic: {result['statistic']:.4f}")
print(f"p-value: {result['p_value']:.4f}")
print(f"Rank-biserial r: {result['effect_size']:.4f}")
print()



print("Test 4: One-way ANOVA (3 groups)")
groups = {
    'A': pd.Series(np.random.normal(0, 1, 30)),
    'B': pd.Series(np.random.normal(0.5, 1, 30)),
    'C': pd.Series(np.random.normal(1, 1, 30))
}
result = run_test('one_way_anova', groups)
print(f"Success: {result['success']}")
print(f"F-statistic: {result['statistic']:.4f}")
print(f"p-value: {result['p_value']:.4f}")
print(f"Eta-squared: {result['effect_size']:.4f}")
print()


print("Test 5: Kruskal-Wallis (3 non-normal groups)")
groups = {
    'A': pd.Series(np.random.exponential(1, 30)),
    'B': pd.Series(np.random.exponential(1.5, 30)),
    'C': pd.Series(np.random.exponential(2, 30))
}
result = run_test('kruskal_wallis', groups)
print(f"Success: {result['success']}")
print(f"H-statistic: {result['statistic']:.4f}")
print(f"p-value: {result['p_value']:.4f}")
print(f"Epsilon-squared: {result['effect_size']:.4f}")
print()


print("Test 6: Paired t-test")
before = pd.Series(np.random.normal(100, 15, 30))
after = before + np.random.normal(5, 10, 30)
groups = {'before': before, 'after': after}
result = run_test('paired_t', groups)
print(f"Success: {result['success']}")
print(f"t-statistic: {result['statistic']:.4f}")
print(f"p-value: {result['p_value']:.4f}")
print(f"Mean difference: {result['mean_difference']:.4f}")
print()


print("Test 7: Wilcoxon signed-rank")
before = pd.Series(np.random.exponential(1, 30))
after = before * np.random.uniform(1.1, 1.3, 30)
groups = {'before': before, 'after': after}
result = run_test('wilcoxon_signed_rank', groups)
print(f"Success: {result['success']}")
print(f"W-statistic: {result['statistic']:.4f}")
print(f"p-value: {result['p_value']:.4f}")
print()


print("Test 8: Pearson correlation")
x = pd.Series(np.random.normal(0, 1, 100))
y = 0.7 * x + np.random.normal(0, 0.5, 100)
groups = {'x': x, 'y': y}
result = run_test('pearson_r', groups)
print(f"Success: {result['success']}")
print(f"r: {result['correlation']:.4f}")
print(f"r²: {result['effect_size']:.4f}")
print(f"p-value: {result['p_value']:.4f}")
print()


print("Test 9: Spearman correlation")
x = pd.Series(np.random.uniform(0, 10, 100))
y = x ** 2 + np.random.normal(0, 5, 100)
groups = {'x': x, 'y': y}
result = run_test('spearman_r', groups)
print(f"Success: {result['success']}")
print(f"ρ: {result['correlation']:.4f}")
print(f"ρ²: {result['effect_size']:.4f}")
print(f"p-value: {result['p_value']:.4f}")
print()
