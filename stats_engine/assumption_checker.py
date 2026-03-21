import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Any


def check_homogeneity(
    groups: List[pd.Series],
    alpha: float = 0.05
) -> Dict[str, Any]:


    # Validate input
    if len(groups) < 2:
        return {
            "test": "insufficient_groups",
            "statistic": None,
            "p_value": None,
            "passed": False,
            "n_groups": len(groups),
            "group_sizes": [len(g) for g in groups],
            "note": f"Need at least 2 groups for comparison, got {len(groups)}"
        }
    

    # Clean data (drop NaN)
    cleaned_groups = [g.dropna() for g in groups]
    group_sizes = [len(g) for g in cleaned_groups]
    

    # Validate each group has sufficient data
    for i, g in enumerate(cleaned_groups):
        if len(g) < 2:
            return {
                "test": "insufficient_data",
                "statistic": None,
                "p_value": None,
                "passed": False,
                "n_groups": len(groups),
                "group_sizes": group_sizes,
                "note": f"Group {i} has only {len(g)} observation(s), need at least 2 per group"
            }
    
    
    # Perform Levene's test with median center (most robust)
    try:
        result = stats.levene(*cleaned_groups, center="median")
        stat = float(np.float64(result.statistic))
        p = float(np.float64(result.pvalue))
        
        return {
            "test": "levene",
            "statistic": round(stat, 4),
            "p_value": round(p, 4),
            "passed": p > alpha,
            "n_groups": len(groups),
            "group_sizes": group_sizes,
            "note": f"{'Equal' if p > alpha else 'Unequal'} variance across {len(groups)} groups (p={p:.4f})"
        }
    
    except Exception as e:
        return {
            "test": "error",
            "statistic": None,
            "p_value": None,
            "passed": False,
            "n_groups": len(groups),
            "group_sizes": group_sizes,
            "note": f"Levene's test failed: {str(e)}"
        }
    

def check_normality(data: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:

    # Drop NaN values
    clean_data = data.dropna()
    n = len(clean_data)
    
    # Insufficient data check
    if n < 5:
        return {
            "test": "insufficient_data",
            "statistic": None,
            "p_value": None,
            "passed": False,
            "n": n,
            "skewness": None,
            "kurtosis": None,
            "note": f"Need at least 5 observations for normality test, got {n}"
        }
    
    # Choose test based on sample size
    if n <= 5000:
        # Shapiro-Wilk for small-medium samples
        result = stats.shapiro(clean_data)
        test_name = "shapiro_wilk"
    else:
        # D'Agostino-Pearson for large samples
        result = stats.normaltest(clean_data)
        test_name = "dagostino_pearson"
    
    # Extract test results
    stat = float(np.float64(result.statistic))
    p = float(np.float64(result.pvalue))
    
    # Compute shape metrics
    skew_val = float(stats.skew(clean_data))
    kurt_val = float(stats.kurtosis(clean_data))  # Excess kurtosis (normal=0)
    
    # Interpret shape
    shape_notes = []
    
    # Skewness interpretation
    if abs(skew_val) > 1:
        direction = "right" if skew_val > 0 else "left"
        shape_notes.append(f"strongly {direction}-skewed")
    elif abs(skew_val) > 0.5:
        direction = "right" if skew_val > 0 else "left"
        shape_notes.append(f"{direction}-skewed")
    
    # Kurtosis interpretation
    if abs(kurt_val) > 1:
        tail_type = "heavy" if kurt_val > 0 else "light"
        shape_notes.append(f"{tail_type} tails")
    
    shape_desc = ", ".join(shape_notes) if shape_notes else "approximately symmetric"
    
    # Build note
    normality_status = "Normal" if p > alpha else "Non-normal"
    note = f"{normality_status} (p={p:.4f}); {shape_desc}"
    
    return {
        "test": test_name,
        "statistic": round(stat, 4),
        "p_value": round(p, 4),
        "passed": p > alpha,
        "n": n,
        "skewness": round(skew_val, 4),
        "kurtosis": round(kurt_val, 4),
        "note": note
    }


def recommend_test(
    data_profile: Dict[str, Any],
    design: str = "independent",
    n_groups: int = 2
) -> Dict[str, Any]:
    
    warnings = []
    assumptions = {}
    alternative_tests = []
    
    normality_results = data_profile.get('normality', {})
    homogeneity_result = data_profile.get('homogeneity', {})
    
    all_normal = all(
        check_result.get('passed', False)
        for check_result in normality_results.values()
    )
    assumptions['all_groups_normal'] = all_normal
    
    variance_equal = homogeneity_result.get('passed', True)
    assumptions['variance_equal'] = variance_equal
    
    non_normal_count = sum(
        1 for check in normality_results.values()
        if not check.get('passed', False)
    )

    # ── Repeated measures (3+ conditions) ─────────────────────
    # Must come before the n_groups == 2 branch so 'repeated'
    # design with 3 conditions doesn't fall through to paired logic
    if design == 'repeated':
        if n_groups < 3:
            # Fall through to paired logic below
            design = 'paired'
        else:
            if all_normal:
                test = "repeated_anova"
                rationale = (
                    f"Repeated measures design with {n_groups} conditions, "
                    "normal differences — repeated measures ANOVA."
                )
                alternative_tests = ["friedman"]
                warnings.append(
                    "Repeated ANOVA assumes sphericity. "
                    "Consider Greenhouse-Geisser correction if sphericity is violated."
                )
            else:
                test = "friedman"
                rationale = (
                    f"Repeated measures design with {n_groups} conditions, "
                    f"{non_normal_count} condition(s) failed normality — Friedman test."
                )
                alternative_tests = ["repeated_anova"]
                warnings.append(
                    "Friedman significant? Follow up with pairwise Wilcoxon + Holm correction."
                )
            return {
                'recommended_test': test,
                'rationale': rationale,
                'assumptions_met': assumptions,
                'warnings': warnings,
                'alternative_tests': alternative_tests,
                'n_groups': n_groups,
                'design': design
            }

    # DECISION TREE
    
    if design == "paired":
        # Paired samples - test differences for normality
        if all_normal:
            test = "paired_t"
            rationale = "Paired design with normally distributed differences"
            alternative_tests = ["wilcoxon_signed_rank"]
        else:
            test = "wilcoxon_signed_rank"
            rationale = "Paired design with non-normal differences (non-parametric test)"
            alternative_tests = ["sign_test"]
            warnings.append(
                "Wilcoxon signed-rank has lower statistical power than paired t-test "
                "when normality holds. Consider data transformation if appropriate."
            )
    
    elif design == "correlation":
        # Correlation between two continuous variables
        if all_normal:
            test = "pearson_r"
            rationale = "Linear correlation between two normally distributed variables"
            alternative_tests = ["spearman_r"]
        else:
            test = "spearman_r"
            rationale = "Monotonic correlation (at least one variable is non-normal)"
            alternative_tests = ["kendall_tau"]
            warnings.append(
                f"Spearman correlation detected. {non_normal_count} variable(s) "
                "failed normality. Spearman measures monotonic (not necessarily linear) relationships."
            )
    
    elif n_groups == 2:
        # Two independent groups
        if all_normal:
            if variance_equal:
                test = "independent_t"
                rationale = "Two independent groups, normal distributions, equal variance"
                alternative_tests = ["welch_t", "mann_whitney_u"]
            else:
                test = "welch_t"
                rationale = (
                    "Two independent groups, normal distributions, unequal variance "
                    "(Welch's correction applied)"
                )
                alternative_tests = ["mann_whitney_u"]
                warnings.append(
                    "Welch's t-test does not assume equal variance. "
                    "Degrees of freedom are adjusted using Satterthwaite approximation."
                )
        else:
            test = "mann_whitney_u"
            rationale = (
                f"Two independent groups, {non_normal_count} group(s) failed normality "
                "(non-parametric test)"
            )
            alternative_tests = ["permutation_test"]
            
            # Explain WHY non-parametric even if one group is normal
            if non_normal_count == 1:
                warnings.append(
                    "Mann-Whitney U test chosen because at least one group is non-normal. "
                    "Parametric tests (t-test, Welch) assume normality in BOTH groups."
                )
            
            warnings.append(
                "Mann-Whitney U has ~95% power efficiency compared to t-test when normality holds, "
                "but is more robust to outliers and skewed distributions."
            )
    

    elif n_groups >= 3:
        if all_normal and variance_equal:
            test = "one_way_anova"
            rationale = f"{n_groups} independent groups, all normal, equal variance"
            alternative_tests = ["kruskal_wallis"]
            warnings.append(
                "ANOVA significant? Run Tukey HSD post-hoc to identify which groups differ."
            )
        
        elif all_normal and not variance_equal:
            test = "welch_anova"
            rationale = (
                f"{n_groups} independent groups, normal distributions, unequal variance. "
                "Welch's ANOVA is more powerful than Kruskal-Wallis when normality holds."
            )
            alternative_tests = ["kruskal_wallis"]
            warnings.append(
                "Welch's ANOVA chosen over Kruskal-Wallis: data is normal but variance is heterogeneous. "
                "If significant, use Games-Howell post-hoc (does not assume equal variance)."
            )
        
        else:
            test = "kruskal_wallis"
            reasons = []
            if not all_normal:
                reasons.append(f"{non_normal_count} group(s) failed normality")
            if not variance_equal:
                reasons.append("unequal variance")
            rationale = f"{n_groups} independent groups, {' and '.join(reasons)} (non-parametric)"
            alternative_tests = ["welch_anova"]
            warnings.append(
                "Kruskal-Wallis significant? Run Dunn's test (Holm correction) to identify which groups differ."
            )

    else:
        raise ValueError(
            f"Invalid n_groups: {n_groups}. Must be >= 2 for group comparisons."
        )
    
    return {
        'recommended_test': test,
        'rationale': rationale,
        'assumptions_met': assumptions,
        'warnings': warnings,
        'alternative_tests': alternative_tests,
        'n_groups': n_groups,
        'design': design
    }


def build_data_profile(
    groups: Dict[str, pd.Series],
    design: str = "independent",
    alpha: float = 0.05,
    run_posthoc: bool = False,         
    primary_result: Dict = None        
) -> Dict[str, Any]:
    
    profile = {
        'n_groups': len(groups),
        'design': design,
        'alpha': alpha,
        'normality': {},
        'sample_sizes': {},
        'descriptive_stats': {},
        'guardrails': {}     
    }


    guardrail_result = check_guardrails(groups, design)
    profile['guardrails'] = guardrail_result
    
    if guardrail_result['blocked']:
        profile['recommendation'] = {
            'recommended_test': None,
            'rationale': 'Analysis blocked by data quality issues.',
            'blocked': True,
            'issues': guardrail_result['issues'],
            'warnings': [],
            'alternative_tests': []
        }
        return profile


    # Check normality for each group
    for group_name, data in groups.items():
        profile['normality'][group_name] = check_normality(data, alpha)
        profile['sample_sizes'][group_name] = len(data.dropna())
        
        # Basic descriptive stats
        clean_data = data.dropna()
        if len(clean_data) > 0:
            profile['descriptive_stats'][group_name] = {
                'mean': round(float(clean_data.mean()), 4),
                'median': round(float(clean_data.median()), 4),
                'std': round(float(clean_data.std()), 4),
                'min': round(float(clean_data.min()), 4),
                'max': round(float(clean_data.max()), 4)
            }
    
    # Check homogeneity of variance (only for independent design with 2+ groups)
    if design == "independent" and len(groups) >= 2:
        group_list = list(groups.values())
        profile['homogeneity'] = check_homogeneity(group_list, alpha)
    
    # Get test recommendation
    recommendation = recommend_test(
        profile,
        design=design,
        n_groups=len(groups)
    )
    profile['recommendation'] = recommendation
    
    if run_posthoc and primary_result and primary_result.get('p_value', 1.0) < alpha:
        recommended = profile['recommendation']['recommended_test']
        posthoc_map = {
            'one_way_anova': 'tukey_hsd',
            'welch_anova': 'tukey_hsd',    
            'kruskal_wallis': 'dunns_test',
            'friedman': 'pairwise_wilcoxon'
        }
        posthoc_test = posthoc_map.get(recommended)
        if posthoc_test:
            from stats_engine.executor import run_test
            profile['posthoc'] = run_test(posthoc_test, groups)

    return profile


def check_guardrails(
    groups: Dict[str, pd.Series],
    design: str = 'independent'
) -> Dict[str, Any]:

    issues = []
    
    for group_name, series in groups.items():
        clean = series.dropna()
        n = len(clean)
        
        # Critically small sample
        if n < 5:
            issues.append(
                f"Variable '{group_name}' has only {n} observations. "
                f"Statistical tests are unreliable with n < 5."
            )
        
        # Zero variance
        if n >= 2 and float(clean.std()) < 1e-10:
            issues.append(
                f"Variable '{group_name}' has zero variance — all values are identical. "
                f"Statistical testing is meaningless. Check for data entry errors."
            )
        
        # Extreme missingness
        original_n = len(series)
        missing_pct = (original_n - n) / original_n * 100 if original_n > 0 else 0
        if missing_pct > 20:
            issues.append(
                f"Variable '{group_name}' has {missing_pct:.0f}% missing values. "
                f"Results may be biased."
            )
    
    # Paired size check — skip for correlation (two variables, not paired observations)
    if design == 'paired':
        sizes = [len(g.dropna()) for g in groups.values()]
        if len(set(sizes)) > 1:
            issues.append(
                f"Paired design requires equal group sizes, but found: "
                f"{dict(zip(groups.keys(), sizes))}. "
                f"Cannot run paired test with mismatched sizes."
            )
    
    return {
        'blocked': len(issues) > 0,
        'issues': issues,
        'n_issues': len(issues)
    }