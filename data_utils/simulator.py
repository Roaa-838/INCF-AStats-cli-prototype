import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


def two_groups_normal_equal_var(
    n_per_group: int = 50,
    mean1: float = 0.0,
    mean2: float = 0.5,
    std: float = 1.0,
    seed: int = 42
) -> Tuple[pd.DataFrame, str, str, str, str]:

    np.random.seed(seed)
    
    group1 = np.random.normal(mean1, std, n_per_group)
    group2 = np.random.normal(mean2, std, n_per_group)
    
    df = pd.DataFrame({
        'value': np.concatenate([group1, group2]),
        'group': ['control'] * n_per_group + ['treatment'] * n_per_group
    })
    
    return (
        df,
        'value',
        'group',
        'independent_t',
        'two_groups_normal_equal_variance'
    )


def two_groups_normal_unequal_var(
    n_per_group: int = 50,
    mean1: float = 0.0,
    mean2: float = 0.0,  # Same mean (H0 true), but different variance
    std1: float = 1.0,
    std2: float = 5.0,  # Much larger variance
    seed: int = 42
) -> Tuple[pd.DataFrame, str, str, str, str]:

    np.random.seed(seed)
    
    group1 = np.random.normal(mean1, std1, n_per_group)
    group2 = np.random.normal(mean2, std2, n_per_group)
    
    df = pd.DataFrame({
        'value': np.concatenate([group1, group2]),
        'group': ['control'] * n_per_group + ['treatment'] * n_per_group
    })
    
    return (
        df,
        'value',
        'group',
        'welch_t',
        'two_groups_normal_unequal_variance'
    )


def two_groups_nonnormal(
    n_per_group: int = 50,
    scale1: float = 1.0,
    scale2: float = 2.0,
    seed: int = 42
) -> Tuple[pd.DataFrame, str, str, str, str]:

    np.random.seed(seed)
    
    group1 = np.random.exponential(scale1, n_per_group)
    group2 = np.random.exponential(scale2, n_per_group)
    
    df = pd.DataFrame({
        'value': np.concatenate([group1, group2]),
        'group': ['control'] * n_per_group + ['treatment'] * n_per_group
    })
    
    return (
        df,
        'value',
        'group',
        'mann_whitney_u',
        'two_groups_nonnormal_exponential'
    )


def get_two_group_scenarios() -> Dict[str, Any]:

    scenarios = {}
    
    # Scenario 1: Normal + Equal Var
    df, target, group, test, name = two_groups_normal_equal_var()
    scenarios[name] = {
        'df': df,
        'target_col': target,
        'group_col': group,
        'correct_test': test,
        'scenario_name': name,
        'description': 'Two normal groups with equal variance (Student\'s t-test)'
    }
    
    # Scenario 2: Normal + Unequal Var
    df, target, group, test, name = two_groups_normal_unequal_var()
    scenarios[name] = {
        'df': df,
        'target_col': target,
        'group_col': group,
        'correct_test': test,
        'scenario_name': name,
        'description': 'Two normal groups with unequal variance (Welch\'s t-test)'
    }
    
    # Scenario 3: Non-Normal
    df, target, group, test, name = two_groups_nonnormal()
    scenarios[name] = {
        'df': df,
        'target_col': target,
        'group_col': group,
        'correct_test': test,
        'scenario_name': name,
        'description': 'Two exponential (non-normal) groups (Mann-Whitney U)'
    }
    
    return scenarios


def three_groups_normal_homosc(
    n_per_group: int = 30,
    means: Tuple[float, float, float] = (0.0, 0.5, 1.0),
    std: float = 1.0,
    seed: int = 42
) -> Tuple[pd.DataFrame, str, str, str, str]:
    
    np.random.seed(seed)
    
    group_a = np.random.normal(means[0], std, n_per_group)
    group_b = np.random.normal(means[1], std, n_per_group)
    group_c = np.random.normal(means[2], std, n_per_group)
    
    df = pd.DataFrame({
        'value': np.concatenate([group_a, group_b, group_c]),
        'group': ['A'] * n_per_group + ['B'] * n_per_group + ['C'] * n_per_group
    })
    
    return (
        df,
        'value',
        'group',
        'one_way_anova',
        'three_groups_normal_homoscedastic'
    )


def three_groups_nonnormal(
    n_per_group: int = 30,
    scales: Tuple[float, float, float] = (1.0, 1.5, 2.0),
    seed: int = 42
) -> Tuple[pd.DataFrame, str, str, str, str]:

    np.random.seed(seed)
    
    group_a = np.random.exponential(scales[0], n_per_group)
    group_b = np.random.exponential(scales[1], n_per_group)
    group_c = np.random.exponential(scales[2], n_per_group)
    
    df = pd.DataFrame({
        'value': np.concatenate([group_a, group_b, group_c]),
        'group': ['A'] * n_per_group + ['B'] * n_per_group + ['C'] * n_per_group
    })
    
    return (
        df,
        'value',
        'group',
        'kruskal_wallis',
        'three_groups_nonnormal_exponential'
    )


def continuous_correlation_normal(
    n: int = 100,
    correlation: float = 0.7,
    seed: int = 42
) -> Tuple[pd.DataFrame, str, str, str, str]:

    np.random.seed(seed)
    
    x = np.random.normal(0, 1, n)
    # Create correlated y: y = r*x + sqrt(1-r²)*noise
    noise = np.random.normal(0, 1, n)
    y = correlation * x + np.sqrt(1 - correlation**2) * noise
    
    df = pd.DataFrame({'x': x, 'y': y})
    
    return (
        df,
        'x',  # using both x and y, but target_col represents first variable
        'y',  # group_col represents second variable (repurposed for correlation)
        'pearson_r',
        'continuous_correlation_normal_linear'
    )


def continuous_correlation_nonnormal(
    n: int = 100,
    seed: int = 42
) -> Tuple[pd.DataFrame, str, str, str, str]:

    np.random.seed(seed)
    
    x = np.random.exponential(1, n)  # Non-normal
    # Quadratic relationship (monotonic but not linear)
    y = x**2 + np.random.normal(0, 5, n)
    
    df = pd.DataFrame({'x': x, 'y': y})
    
    return (
        df,
        'x',
        'y',
        'spearman_r',
        'continuous_correlation_nonnormal_monotonic'
    )


def repeated_measures_normal(n_subjects=25, seed=42):
    """Paired design, normal differences → paired t-test"""
    np.random.seed(seed)
    before = np.random.normal(50, 10, n_subjects)
    after = before + np.random.normal(5, 3, n_subjects)
    subject_ids = np.arange(n_subjects)
    
    df = pd.DataFrame({
        'subject': np.concatenate([subject_ids, subject_ids]),
        'condition': ['before'] * n_subjects + ['after'] * n_subjects,
        'score': np.concatenate([before, after])
    })
    return df, 'score', 'condition', 'paired_t', 'repeated_normal_2cond'


def repeated_measures_nonnormal(n_subjects=25, seed=42):
    """Paired design, non-normal → Wilcoxon"""
    np.random.seed(seed)
    before = np.random.exponential(1.0, n_subjects)
    after = before * np.random.uniform(1.2, 1.5, n_subjects)
    subject_ids = np.arange(n_subjects)
    
    df = pd.DataFrame({
        'subject': np.concatenate([subject_ids, subject_ids]),
        'condition': ['before'] * n_subjects + ['after'] * n_subjects,
        'score': np.concatenate([before, after])
    })
    return df, 'score', 'condition', 'wilcoxon_signed_rank', 'repeated_nonnormal_2cond'


def three_groups_normal_unequal_var(n_per_group=40, seed=42):
    """3 groups, normal, unequal variance → Welch's ANOVA (NOT Kruskal-Wallis)"""
    np.random.seed(seed)
    a = np.random.normal(0, 1, n_per_group)
    b = np.random.normal(0.5, 1, n_per_group)
    c = np.random.normal(1, 5, n_per_group)   # much larger variance
    
    df = pd.DataFrame({
        'value': np.concatenate([a, b, c]),
        'group': ['A'] * n_per_group + ['B'] * n_per_group + ['C'] * n_per_group
    })
    return df, 'value', 'group', 'welch_anova', 'three_groups_normal_unequal_var'


def three_conditions_repeated_nonnormal(n_subjects=20, seed=42):
    """3 repeated conditions, non-normal → Friedman"""
    np.random.seed(seed)
    baseline = np.random.exponential(1.0, n_subjects)
    cond1 = baseline + np.random.exponential(0.5, n_subjects)
    cond2 = baseline + np.random.exponential(1.0, n_subjects)
    subject_ids = np.arange(n_subjects)
    
    df = pd.DataFrame({
        'subject': np.tile(subject_ids, 3),
        'condition': ['A'] * n_subjects + ['B'] * n_subjects + ['C'] * n_subjects,
        'score': np.concatenate([baseline, cond1, cond2])
    })
    return df, 'score', 'condition', 'friedman', 'three_cond_repeated_nonnormal'


def zero_variance_guardrail(seed=42):
    """One group has zero variance → should be BLOCKED"""
    np.random.seed(seed)
    df = pd.DataFrame({
        'value': np.concatenate([
            np.random.normal(0, 1, 30),
            np.full(30, 5.0)             # identical values
        ]),
        'group': ['A'] * 30 + ['B'] * 30
    })
    return df, 'value', 'group', 'BLOCKED', 'zero_variance_guardrail'


def small_n_guardrail(seed=42):
    """n < 5 in one group → should be BLOCKED"""
    np.random.seed(seed)
    df = pd.DataFrame({
        'value': np.concatenate([
            np.random.normal(0, 1, 3),   # critically small
            np.random.normal(1, 1, 30)
        ]),
        'group': ['tiny'] * 3 + ['normal'] * 30
    })
    return df, 'value', 'group', 'BLOCKED', 'small_n_guardrail'


def neuro_reaction_times(
    n_per_condition: int = 40,
    seed: int = 42
) -> Tuple[pd.DataFrame, str, str, str, str]:
   
    np.random.seed(seed)
    
    # Log-normal parameters: exp(mean) gives median
    baseline = np.random.lognormal(np.log(300), 0.2, n_per_condition)
    cue = np.random.lognormal(np.log(250), 0.2, n_per_condition)
    no_cue = np.random.lognormal(np.log(350), 0.2, n_per_condition)
    
    df = pd.DataFrame({
        'reaction_time_ms': np.concatenate([baseline, cue, no_cue]),
        'condition': (
            ['baseline'] * n_per_condition +
            ['cue'] * n_per_condition +
            ['no_cue'] * n_per_condition
        )
    })
    
    return (
        df,
        'reaction_time_ms',
        'condition',
        'kruskal_wallis',
        'neuro_reaction_times_three_conditions'
    )


def get_all_scenarios() -> Dict[str, Any]:

    scenarios = {}
    
    # Two-group scenarios (3)
    scenarios.update(get_two_group_scenarios())
    
    # Three-group scenarios (2)
    df, target, group, test, name = three_groups_normal_homosc()
    scenarios[name] = {
        'df': df,
        'target_col': target,
        'group_col': group,
        'correct_test': test,
        'scenario_name': name,
        'description': 'Three normal groups, equal variance (one-way ANOVA)'
    }
    
    df, target, group, test, name = three_groups_nonnormal()
    scenarios[name] = {
        'df': df,
        'target_col': target,
        'group_col': group,
        'correct_test': test,
        'scenario_name': name,
        'description': 'Three exponential groups (Kruskal-Wallis)'
    }
    
    # Correlation scenarios (2)
    df, x_col, y_col, test, name = continuous_correlation_normal()
    scenarios[name] = {
        'df': df,
        'target_col': x_col,
        'group_col': y_col,  # Repurposed for correlation
        'correct_test': test,
        'scenario_name': name,
        'description': 'Linear correlation, both normal (Pearson)'
    }
    
    df, x_col, y_col, test, name = continuous_correlation_nonnormal()
    scenarios[name] = {
        'df': df,
        'target_col': x_col,
        'group_col': y_col,
        'correct_test': test,
        'scenario_name': name,
        'description': 'Monotonic correlation, non-normal (Spearman)'
    }
    
    # Neuroscience scenario
    df, target, group, test, name = neuro_reaction_times()
    scenarios[name] = {
        'df': df,
        'target_col': target,
        'group_col': group,
        'correct_test': test,
        'scenario_name': name,
        'description': 'Neuroscience RT data, 3 conditions, log-normal (Kruskal-Wallis)'
    }

    df, target, group, test, name = repeated_measures_normal()
    scenarios[name] = {
        'df': df, 'target_col': target, 'group_col': group,
        'correct_test': test, 'scenario_name': name,
        'description': 'Paired design, normal differences (paired t-test)',
        'design': 'paired',
        'subject_col': 'subject'
    }

    df, target, group, test, name = repeated_measures_nonnormal()
    scenarios[name] = {
        'df': df, 'target_col': target, 'group_col': group,
        'correct_test': test, 'scenario_name': name,
        'description': 'Paired design, non-normal differences (Wilcoxon)',
        'design': 'paired',
        'subject_col': 'subject'
    }

    df, target, group, test, name = three_groups_normal_unequal_var()
    scenarios[name] = {
        'df': df, 'target_col': target, 'group_col': group,
        'correct_test': test, 'scenario_name': name,
        'description': 'Three normal groups, unequal variance (Welch\'s ANOVA)'
    }

    df, target, group, test, name = three_conditions_repeated_nonnormal()
    scenarios[name] = {
        'df': df, 'target_col': target, 'group_col': group,
        'correct_test': test, 'scenario_name': name,
        'description': 'Three repeated conditions, non-normal (Friedman)',
        'design': 'repeated',
        'subject_col': 'subject'
    }

    df, target, group, test, name = zero_variance_guardrail()
    scenarios[name] = {
        'df': df, 'target_col': target, 'group_col': group,
        'correct_test': test, 'scenario_name': name,
        'description': 'Zero variance in one group (should be BLOCKED)',
        'score_type': 'guardrail'
    }

    df, target, group, test, name = small_n_guardrail()
    scenarios[name] = {
        'df': df, 'target_col': target, 'group_col': group,
        'correct_test': test, 'scenario_name': name,
        'description': 'Critically small n in one group (should be BLOCKED)',
        'score_type': 'guardrail'
    }

    return scenarios