import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from stats_engine.executor import run_test

np.random.seed(42)


def test_tukey_hsd_finds_differences():

    groups = {
        'A': pd.Series(np.random.normal(0, 1, 40)),
        'B': pd.Series(np.random.normal(3, 1, 40)),  # clearly different from A
        'C': pd.Series(np.random.normal(0.2, 1, 40)) # similar to A
    }
    
    result = run_test('tukey_hsd', groups)
    
    assert result['success'], f"Tukey HSD failed: {result.get('error')}"
    assert result['n_comparisons'] == 3  # C(3,2) = 3 pairs
    
    sig_pairs = result['significant_pairs']
    # A vs B should be significant, A vs C probably not
    assert any('A' in p and 'B' in p for p in sig_pairs), (
        f"A vs B should be significant. Got: {sig_pairs}"
    )
    print(f"Tukey HSD: {len(sig_pairs)}/3 pairs significant: {sig_pairs}")


def test_dunns_test_finds_differences():
    """Kruskal-Wallis significant → Dunn's should find which groups differ"""
    groups = {
        'low': pd.Series(np.random.exponential(1, 40)),
        'mid': pd.Series(np.random.exponential(3, 40)),
        'high': pd.Series(np.random.exponential(8, 40))
    }
    
    result = run_test('dunns_test', groups)
    
    assert result['success'], f"Dunn's test failed: {result.get('error')}"
    sig_pairs = result['significant_pairs']
    
    # low vs high should definitely be significant
    assert any(
        ('low' in p and 'high' in p) or ('high' in p and 'low' in p)
        for p in sig_pairs
    ), f"low vs high should be significant. Got: {sig_pairs}"
    print(f"✓ Dunn's test: {len(sig_pairs)} significant pairs found")


def test_friedman_then_posthoc():
    """Full repeated measures pipeline: Friedman → pairwise Wilcoxon"""
    n_subjects = 25
    np.random.seed(42)
    
    # Clear effect: condition C is much higher
    baseline = np.random.normal(50, 5, n_subjects)
    groups = {
        'condition_A': pd.Series(baseline + np.random.normal(0, 2, n_subjects)),
        'condition_B': pd.Series(baseline + np.random.normal(2, 2, n_subjects)),
        'condition_C': pd.Series(baseline + np.random.normal(10, 2, n_subjects))
    }
    
    # Friedman test
    friedman_result = run_test('friedman', groups)
    assert friedman_result['success']
    assert friedman_result['p_value'] < 0.05, (
        f"Friedman should be significant, got p={friedman_result['p_value']:.4f}"
    )
    print(f"Friedman: χ²={friedman_result['statistic']:.2f}, "
          f"p={friedman_result['p_value']:.4f}, W={friedman_result['effect_size']:.3f}")
    
    # Post-hoc
    posthoc_result = run_test('pairwise_wilcoxon', groups)
    assert posthoc_result['success']
    sig_pairs = posthoc_result['significant_pairs']
    
    # Condition C vs others should be significant
    assert any('condition_C' in p for p in sig_pairs), (
        f"condition_C should differ from others. Got: {sig_pairs}"
    )
    print(f"Post-hoc Wilcoxon: {len(sig_pairs)} significant pairs: {sig_pairs}")


def test_guardrails_catch_zero_variance():
    from stats_engine.assumption_checker import check_guardrails
    
    groups = {
        'normal': pd.Series(np.random.normal(0, 1, 30)),
        'zero_var': pd.Series([5.0] * 30)  # all identical
    }
    
    result = check_guardrails(groups, design='independent')
    assert result['blocked'], "Zero variance should block analysis"
    assert any('zero variance' in issue.lower() for issue in result['issues'])
    print(f"Guardrail caught zero variance: {result['issues'][0][:60]}...")


def test_guardrails_catch_small_n():
    from stats_engine.assumption_checker import check_guardrails
    
    groups = {
        'tiny': pd.Series([1, 2, 3]),  # n=3
        'normal': pd.Series(np.random.normal(0, 1, 30))
    }
    
    result = check_guardrails(groups, design='independent')
    assert result['blocked'], "n<5 should block analysis"
    print(f"Guardrail caught small n: {result['issues'][0][:60]}...")


if __name__ == '__main__':
    test_tukey_hsd_finds_differences()
    test_dunns_test_finds_differences()
    test_friedman_then_posthoc()
    test_guardrails_catch_zero_variance()
    test_guardrails_catch_small_n()
    print("\nAll post-hoc and guardrail tests passed")
