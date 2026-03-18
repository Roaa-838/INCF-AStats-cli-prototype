import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Union, List

try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False


def run_test(
    test_name: str,
    data: Union[Dict[str, pd.Series], pd.DataFrame],
    **kwargs
) -> Dict[str, Any]:
 
    try:
        # Dispatch to specific test implementation
        test_map = {
            'independent_t': _independent_t,
            'welch_t': _welch_t,
            'mann_whitney_u': _mann_whitney_u,
            'paired_t': _paired_t,
            'wilcoxon_signed_rank': _wilcoxon,
            'one_way_anova': _one_way_anova,
            'kruskal_wallis': _kruskal_wallis,
            'pearson_r': _pearson_correlation,
            'spearman_r': _spearman_correlation
        }
        
        if test_name not in test_map:
            return {
                'test': test_name,
                'success': False,
                'error': f"Unknown test: {test_name}. Available tests: {list(test_map.keys())}"
            }
        
        test_func = test_map[test_name]
        return test_func(data, **kwargs)
    
    except Exception as e:
        return {
            'test': test_name,
            'success': False,
            'error': f"Test execution failed: {type(e).__name__}: {str(e)}"
        }


def _independent_t(data: Dict[str, pd.Series], **kwargs) -> Dict[str, Any]:

    groups = list(data.values())
    if len(groups) != 2:
        raise ValueError(f"Independent t-test requires exactly 2 groups, got {len(groups)}")
    
    # Clean data
    g1, g2 = groups[0].dropna(), groups[1].dropna()
    n1, n2 = len(g1), len(g2)
    
    # Perform t-test
    result = stats.ttest_ind(g1, g2, equal_var=True)
    
    # Cohen's d effect size (pooled standard deviation formula)
    # d = (M1 - M2) / SD_pooled
    # SD_pooled = sqrt(((n1-1)*SD1^2 + (n2-1)*SD2^2) / (n1+n2-2))
    
    mean_diff = g1.mean() - g2.mean()
    var_pooled = ((n1 - 1) * g1.std()**2 + (n2 - 1) * g2.std()**2) / (n1 + n2 - 2)
    sd_pooled = np.sqrt(var_pooled)
    
    cohen_d = mean_diff / sd_pooled if sd_pooled > 0 else 0.0
    
    return {
        'test': 'independent_t',
        'statistic': float(result.statistic),
        'p_value': float(result.pvalue),
        'effect_size': float(cohen_d),
        'effect_type': 'cohen_d',
        'n_samples': {'group1': n1, 'group2': n2},
        'means': {'group1': float(g1.mean()), 'group2': float(g2.mean())},
        'stds': {'group1': float(g1.std()), 'group2': float(g2.std())},
        'degrees_of_freedom': n1 + n2 - 2,
        'success': True,
        'error': None
    }


def _welch_t(data: Dict[str, pd.Series], **kwargs) -> Dict[str, Any]:

    groups = list(data.values())
    if len(groups) != 2:
        raise ValueError(f"Welch's t-test requires exactly 2 groups, got {len(groups)}")
    
    g1, g2 = groups[0].dropna(), groups[1].dropna()
    n1, n2 = len(g1), len(g2)
    
    # Welch's t-test
    result = stats.ttest_ind(g1, g2, equal_var=False)
    
    # Cohen's d (using unequal variance formula)
    # Use average of the two SDs as denominator
    mean_diff = g1.mean() - g2.mean()
    sd_avg = np.sqrt((g1.std()**2 + g2.std()**2) / 2)
    cohen_d = mean_diff / sd_avg if sd_avg > 0 else 0.0
    
    return {
        'test': 'welch_t',
        'statistic': float(result.statistic),
        'p_value': float(result.pvalue),
        'effect_size': float(cohen_d),
        'effect_type': 'cohen_d',
        'n_samples': {'group1': n1, 'group2': n2},
        'means': {'group1': float(g1.mean()), 'group2': float(g2.mean())},
        'stds': {'group1': float(g1.std()), 'group2': float(g2.std())},
        'note': "Welch's correction applied (does not assume equal variance)",
        'success': True,
        'error': None
    }


def _mann_whitney_u(data: Dict[str, pd.Series], **kwargs) -> Dict[str, Any]:

    groups = list(data.values())
    if len(groups) != 2:
        raise ValueError(f"Mann-Whitney U requires exactly 2 groups, got {len(groups)}")
    
    g1, g2 = groups[0].dropna(), groups[1].dropna()
    n1, n2 = len(g1), len(g2)
    
    # Mann-Whitney U test (two-sided)
    result = stats.mannwhitneyu(g1, g2, alternative='two-sided')
    
    # Rank-biserial correlation as effect size
    # r = 1 - (2U) / (n1*n2)
    # This is the non-parametric equivalent of Cohen's d
    U = result.statistic
    rank_biserial_r = 1 - (2 * U) / (n1 * n2)
    
    return {
        'test': 'mann_whitney_u',
        'statistic': float(result.statistic),  # U statistic
        'p_value': float(result.pvalue),
        'effect_size': float(rank_biserial_r),
        'effect_type': 'rank_biserial_r',
        'n_samples': {'group1': n1, 'group2': n2},
        'medians': {'group1': float(g1.median()), 'group2': float(g2.median())},
        'note': "Non-parametric test - tests difference in distributions, not means",
        'success': True,
        'error': None
    }