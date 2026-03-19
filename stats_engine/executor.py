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

def _paired_t(data: Dict[str, pd.Series], **kwargs) -> Dict[str, Any]:

    groups = list(data.values())
    if len(groups) != 2:
        raise ValueError(f"Paired t-test requires exactly 2 groups, got {len(groups)}")
    
    g1, g2 = groups[0].dropna(), groups[1].dropna()
    
    if len(g1) != len(g2):
        raise ValueError(
            f"Paired test requires equal sample sizes, got {len(g1)} and {len(g2)}. "
            "Ensure data is properly aligned (same indices)."
        )
    
    # Paired t-test
    result = stats.ttest_rel(g1, g2)
    
    # Cohen's d for paired samples
    # d = mean_difference / SD_difference
    differences = g1 - g2
    cohen_d = differences.mean() / differences.std() if differences.std() > 0 else 0.0
    
    return {
        'test': 'paired_t',
        'statistic': float(result.statistic),
        'p_value': float(result.pvalue),
        'effect_size': float(cohen_d),
        'effect_type': 'cohen_d',
        'n_pairs': len(g1),
        'mean_difference': float(differences.mean()),
        'std_difference': float(differences.std()),
        'note': "Tests if mean difference (before - after) is significantly different from zero",
        'success': True,
        'error': None
    }

def _wilcoxon(data: Dict[str, pd.Series], **kwargs) -> Dict[str, Any]:

    groups = list(data.values())
    if len(groups) != 2:
        raise ValueError(f"Wilcoxon test requires exactly 2 groups, got {len(groups)}")
    
    g1, g2 = groups[0].dropna(), groups[1].dropna()
    
    if len(g1) != len(g2):
        raise ValueError(f"Paired test requires equal sample sizes, got {len(g1)} and {len(g2)}")
    
    # Wilcoxon signed-rank test
    result = stats.wilcoxon(g1, g2)
    
    # Rank-biserial correlation for paired data
    # Simplified: (# positive differences - # negative differences) / total
    differences = g1 - g2
    n_positive = sum(differences > 0)
    n_negative = sum(differences < 0)
    rank_biserial_r = (n_positive - n_negative) / len(differences)
    
    return {
        'test': 'wilcoxon_signed_rank',
        'statistic': float(result.statistic),
        'p_value': float(result.pvalue),
        'effect_size': float(rank_biserial_r),
        'effect_type': 'rank_biserial_r',
        'n_pairs': len(g1),
        'median_difference': float((g1 - g2).median()),
        'note': "Non-parametric test - tests if median difference is zero",
        'success': True,
        'error': None
    }

def _one_way_anova(data: Dict[str, pd.Series], **kwargs) -> Dict[str, Any]:

    groups = [g.dropna() for g in data.values()]
    if len(groups) < 2:
        raise ValueError(f"ANOVA requires at least 2 groups, got {len(groups)}")
    
    # One-way ANOVA
    result = stats.f_oneway(*groups)
    
    # Eta-squared effect size
    # η² = SS_between / SS_total
    # Can be computed from F-statistic and sample sizes
    n_total = sum(len(g) for g in groups)
    k = len(groups)  # number of groups
    df_between = k - 1
    df_within = n_total - k
    
    F = result.statistic
    eta_squared = (df_between * F) / (df_between * F + df_within)
    
    group_sizes = [len(g) for g in groups]
    group_means = [float(g.mean()) for g in groups]
    
    return {
        'test': 'one_way_anova',
        'statistic': float(result.statistic),  # F statistic
        'p_value': float(result.pvalue),
        'effect_size': float(eta_squared),
        'effect_type': 'eta_squared',
        'n_samples': {f'group_{i+1}': n for i, n in enumerate(group_sizes)},
        'means': {f'group_{i+1}': m for i, m in enumerate(group_means)},
        'n_groups': len(groups),
        'degrees_of_freedom': {'between': df_between, 'within': df_within},
        'note': "If significant, follow up with post-hoc tests (Tukey HSD)",
        'success': True,
        'error': None
    }


def _kruskal_wallis(data: Dict[str, pd.Series], **kwargs) -> Dict[str, Any]:

    groups = [g.dropna() for g in data.values()]
    if len(groups) < 2:
        raise ValueError(f"Kruskal-Wallis requires at least 2 groups, got {len(groups)}")
    
    # Kruskal-Wallis test
    result = stats.kruskal(*groups)
    
    # Epsilon-squared effect size
    # ε² = (H - k + 1) / (n - k)
    # where H is the test statistic, k is number of groups, n is total sample size
    n_total = sum(len(g) for g in groups)
    k = len(groups)
    H = result.statistic
    
    epsilon_squared = (H - k + 1) / (n_total - k) if (n_total - k) > 0 else 0.0
    
    group_sizes = [len(g) for g in groups]
    group_medians = [float(g.median()) for g in groups]
    
    return {
        'test': 'kruskal_wallis',
        'statistic': float(result.statistic),  # H statistic
        'p_value': float(result.pvalue),
        'effect_size': float(epsilon_squared),
        'effect_type': 'epsilon_squared',
        'n_samples': {f'group_{i+1}': n for i, n in enumerate(group_sizes)},
        'medians': {f'group_{i+1}': m for i, m in enumerate(group_medians)},
        'n_groups': len(groups),
        'note': "Non-parametric test - if significant, follow up with Dunn's test",
        'success': True,
        'error': None
    }

def _pearson_correlation(*args, **kwargs):
    raise NotImplementedError("pearson correlation not implemented yet")

def _spearman_correlation(*args, **kwargs):
    raise NotImplementedError("spearman correlation not implemented yet")