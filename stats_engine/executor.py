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
            'spearman_r': _spearman_correlation,
            'friedman': _friedman,
            'welch_anova': _welch_anova,
            'tukey_hsd': _tukey_hsd,
            'dunns_test': _dunns_test,
            'pairwise_wilcoxon': _pairwise_wilcoxon
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

def _pearson_correlation(data: Dict[str, pd.Series], **kwargs) -> Dict[str, Any]:

    groups = list(data.values())
    if len(groups) != 2:
        raise ValueError(f"Correlation requires exactly 2 variables, got {len(groups)}")
    
    x, y = groups[0].dropna(), groups[1].dropna()
    
    # Align by index (in case of missing values)
    df = pd.DataFrame({'x': x, 'y': y}).dropna()
    x, y = df['x'], df['y']
    
    if len(x) < 3:
        raise ValueError(f"Correlation requires at least 3 pairs, got {len(x)}")
    
    # Pearson correlation
    result = stats.pearsonr(x, y)
    
    return {
        'test': 'pearson_r',
        'statistic': float(result.statistic),  # r value
        'p_value': float(result.pvalue),
        'effect_size': float(result.statistic ** 2),  # r²
        'effect_type': 'r_squared',
        'correlation': float(result.statistic),
        'n_pairs': len(x),
        'note': f"r = {result.statistic:.3f} explains {result.statistic**2*100:.1f}% of variance",
        'success': True,
        'error': None
    }

def _spearman_correlation(data: Dict[str, pd.Series], **kwargs) -> Dict[str, Any]:
 
    groups = list(data.values())
    if len(groups) != 2:
        raise ValueError(f"Correlation requires exactly 2 variables, got {len(groups)}")
    
    x, y = groups[0].dropna(), groups[1].dropna()
    
    # Align by index
    df = pd.DataFrame({'x': x, 'y': y}).dropna()
    x, y = df['x'], df['y']
    
    if len(x) < 3:
        raise ValueError(f"Correlation requires at least 3 pairs, got {len(x)}")
    
    # Spearman correlation
    result = stats.spearmanr(x, y)
    
    return {
        'test': 'spearman_r',
        'statistic': float(result.statistic),  # rho value
        'p_value': float(result.pvalue),
        'effect_size': float(result.statistic ** 2),  # rho²
        'effect_type': 'rho_squared',
        'correlation': float(result.statistic),
        'n_pairs': len(x),
        'note': "Non-parametric - measures monotonic (not necessarily linear) relationship",
        'success': True,
        'error': None
    }


def _friedman(data: Dict[str, pd.Series], **kwargs) -> Dict[str, Any]:
    
    groups = [g.dropna() for g in data.values()]
    if len(groups) < 3:
        raise ValueError(
            f"Friedman test requires at least 3 conditions, got {len(groups)}. "
            "Use Wilcoxon signed-rank for 2 paired conditions."
        )
    
    # All groups must have same length (repeated measures requirement)
    lengths = [len(g) for g in groups]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"Friedman test requires equal observations per condition. "
            f"Got sizes: {lengths}. This indicates missing data or "
            f"unequal repeated measures — consider pairwise Wilcoxon instead."
        )
    
    result = stats.friedmanchisquare(*groups)
    
    n = len(groups[0])   # number of subjects
    k = len(groups)       # number of conditions
    
    # W = chi2 / (n * (k - 1))
    kendalls_w = result.statistic / (n * (k - 1))
    
    group_medians = {
        name: float(g.median())
        for name, g in data.items()
    }
    
    return {
        'test': 'friedman',
        'statistic': float(result.statistic),  # chi-squared statistic
        'p_value': float(result.pvalue),
        'effect_size': float(kendalls_w),
        'effect_type': 'kendalls_w',
        'n_subjects': n,
        'n_conditions': k,
        'medians': group_medians,
        'note': (
            "Friedman test: non-parametric repeated measures. "
            "If significant, follow up with pairwise Wilcoxon + Bonferroni correction."
        ),
        'success': True,
        'error': None
    }


def _welch_anova(data: Dict[str, pd.Series], **kwargs) -> Dict[str, Any]:

    try:
        from statsmodels.stats.oneway import anova_oneway
    except ImportError:
        raise ImportError(
            "statsmodels required for Welch's ANOVA. "
            "Install with: pip install statsmodels"
        )
    
    groups = [g.dropna() for g in data.values()]
    if len(groups) < 3:
        raise ValueError(
            f"Welch's ANOVA requires at least 3 groups, got {len(groups)}."
        )
    
    result = anova_oneway(groups, use_var='unequal')
    
    # eta² = (df_between * F) / (df_between * F + df_within)
    k = len(groups)
    n_total = sum(len(g) for g in groups)
    df_between = k - 1
    df_within_approx = n_total - k
    
    F = float(result.statistic)
    eta_squared = (df_between * F) / (df_between * F + df_within_approx)
    
    group_means = {
        name: float(g.mean())
        for name, g in data.items()
    }
    
    return {
        'test': 'welch_anova',
        'statistic': F,
        'p_value': float(result.pvalue),
        'effect_size': eta_squared,
        'effect_type': 'eta_squared',
        'n_groups': k,
        'means': group_means,
        'note': (
            "Welch's ANOVA does not assume equal variance. "
            "If significant, follow up with Games-Howell post-hoc test."
        ),
        'success': True,
        'error': None
    }


def _tukey_hsd(data: Dict[str, pd.Series], **kwargs) -> Dict[str, Any]:

    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    
    # Reshape into long format for statsmodels
    values = []
    labels = []
    for group_name, series in data.items():
        clean = series.dropna()
        values.extend(clean.tolist())
        labels.extend([group_name] * len(clean))
    
    values_arr = np.array(values)
    labels_arr = np.array(labels)
    
    result = pairwise_tukeyhsd(values_arr, labels_arr, alpha=0.05)
    
    # Extract pairwise results into readable format
    comparisons = []
    summary = result.summary()
    
    # result._results_table has the data we need
    for row in result._results_table.data[1:]:  # skip header
        group1, group2, meandiff, p_adj, lower, upper, reject = row
        comparisons.append({
            'group1': str(group1),
            'group2': str(group2),
            'mean_difference': float(meandiff),
            'p_adjusted': float(p_adj),
            'ci_lower': float(lower),
            'ci_upper': float(upper),
            'significant': bool(reject)
        })
    
    significant_pairs = [
        f"{c['group1']} vs {c['group2']}"
        for c in comparisons
        if c['significant']
    ]
    
    return {
        'test': 'tukey_hsd',
        'comparisons': comparisons,
        'n_comparisons': len(comparisons),
        'significant_pairs': significant_pairs,
        'alpha_corrected': 0.05,
        'method': 'Tukey HSD (controls familywise error rate)',
        'note': (
            f"Found {len(significant_pairs)} significant pairwise difference(s): "
            f"{', '.join(significant_pairs) if significant_pairs else 'none'}."
        ),
        'success': True,
        'error': None
    }


def _dunns_test(data: Dict[str, pd.Series], **kwargs) -> Dict[str, Any]:

    try:
        import scikit_posthocs as sp
    except ImportError:
        # Fallback: manual Bonferroni-corrected Mann-Whitney U
        return _dunns_test_manual(data, **kwargs)
    
    # Build long-format DataFrame
    records = []
    for group_name, series in data.items():
        for val in series.dropna():
            records.append({'group': group_name, 'value': float(val)})
    
    df = pd.DataFrame(records)
    
    # Dunn's test with Holm correction
    p_matrix = sp.posthoc_dunn(
        df, val_col='value', group_col='group', p_adjust='holm'
    )
    
    # Extract pairwise results
    groups = list(data.keys())
    comparisons = []
    
    for i, g1 in enumerate(groups):
        for j, g2 in enumerate(groups):
            if j <= i:
                continue
            p_adj = float(p_matrix.loc[g1, g2])
            comparisons.append({
                'group1': g1,
                'group2': g2,
                'p_adjusted': round(p_adj, 4),
                'significant': p_adj < 0.05
            })
    
    significant_pairs = [
        f"{c['group1']} vs {c['group2']}"
        for c in comparisons
        if c['significant']
    ]
    
    return {
        'test': 'dunns_test',
        'comparisons': comparisons,
        'n_comparisons': len(comparisons),
        'significant_pairs': significant_pairs,
        'correction': 'Holm-Bonferroni',
        'note': (
            f"Dunn's test (Holm correction): "
            f"{len(significant_pairs)} significant pair(s): "
            f"{', '.join(significant_pairs) if significant_pairs else 'none'}."
        ),
        'success': True,
        'error': None
    }


def _dunns_test_manual(data: Dict[str, pd.Series], **kwargs) -> Dict[str, Any]:

    groups = list(data.keys())
    n_comparisons = len(groups) * (len(groups) - 1) // 2
    
    comparisons = []
    for i, g1 in enumerate(groups):
        for j, g2 in enumerate(groups):
            if j <= i:
                continue
            s1 = data[g1].dropna()
            s2 = data[g2].dropna()
            _, p_raw = stats.mannwhitneyu(s1, s2, alternative='two-sided')
            # Bonferroni correction
            p_adj = min(float(p_raw) * n_comparisons, 1.0)
            comparisons.append({
                'group1': g1,
                'group2': g2,
                'p_adjusted': round(p_adj, 4),
                'significant': p_adj < 0.05,
                'correction': 'Bonferroni'
            })
    
    significant_pairs = [
        f"{c['group1']} vs {c['group2']}"
        for c in comparisons
        if c['significant']
    ]
    
    return {
        'test': 'dunns_test',
        'comparisons': comparisons,
        'n_comparisons': n_comparisons,
        'significant_pairs': significant_pairs,
        'correction': 'Bonferroni (manual fallback)',
        'note': f"Install scikit-posthocs for Holm correction. {len(significant_pairs)} significant pair(s).",
        'success': True,
        'error': None
    }


def _pairwise_wilcoxon(data: Dict[str, pd.Series], **kwargs) -> Dict[str, Any]:

    groups = list(data.keys())
    conditions = list(data.values())
    
    n = len(conditions[0].dropna())
    n_comparisons = len(groups) * (len(groups) - 1) // 2
    
    raw_results = []
    for i, g1 in enumerate(groups):
        for j, g2 in enumerate(groups):
            if j <= i:
                continue
            s1 = data[g1].dropna().values
            s2 = data[g2].dropna().values
            min_len = min(len(s1), len(s2))
            stat, p_raw = stats.wilcoxon(s1[:min_len], s2[:min_len])
            raw_results.append({
                'group1': g1,
                'group2': g2,
                'statistic': float(stat),
                'p_raw': float(p_raw)
            })
    
    # Holm-Bonferroni correction
    raw_results.sort(key=lambda x: x['p_raw'])
    comparisons = []
    for rank, r in enumerate(raw_results):
        p_adj = min(r['p_raw'] * (n_comparisons - rank), 1.0)
        comparisons.append({
            'group1': r['group1'],
            'group2': r['group2'],
            'statistic': r['statistic'],
            'p_adjusted': round(p_adj, 4),
            'significant': p_adj < 0.05
        })
    
    significant_pairs = [
        f"{c['group1']} vs {c['group2']}"
        for c in comparisons
        if c['significant']
    ]
    
    return {
        'test': 'pairwise_wilcoxon',
        'comparisons': comparisons,
        'n_comparisons': n_comparisons,
        'significant_pairs': significant_pairs,
        'correction': 'Holm-Bonferroni',
        'note': f"{len(significant_pairs)} significant pair(s) after correction.",
        'success': True,
        'error': None
    }

def _games_howell(data: Dict[str, pd.Series], **kwargs) -> Dict[str, Any]:
    try:
        import pingouin as pg
    except ImportError:
        return {
            'success': False,
            'error': 'pingouin required for Games-Howell. pip install pingouin'
        }
    
    # Build long-format DataFrame
    records = []
    for group_name, series in data.items():
        for val in series.dropna():
            records.append({'group': group_name, 'value': float(val)})
    df = pd.DataFrame(records)
    
    result = pg.pairwise_gameshowell(df, dv='value', between='group')
    
    comparisons = []
    for _, row in result.iterrows():
        comparisons.append({
            'group1':      row['A'],
            'group2':      row['B'],
            'mean_diff':   round(float(row['mean(A)'] - row['mean(B)']), 4),
            'p_adjusted':  round(float(row['pval']), 4),
            'significant': float(row['pval']) < 0.05
        })
    
    significant_pairs = [
        f"{c['group1']} vs {c['group2']}"
        for c in comparisons if c['significant']
    ]
    
    return {
        'test':             'games_howell',
        'comparisons':      comparisons,
        'n_comparisons':    len(comparisons),
        'significant_pairs': significant_pairs,
        'correction':       'Games-Howell (Welch-Satterthwaite df)',
        'note': (
            f"Games-Howell post-hoc (does not assume equal variance). "
            f"{len(significant_pairs)} significant pair(s): "
            f"{', '.join(significant_pairs) if significant_pairs else 'none'}."
        ),
        'success': True,
        'error':   None
    }