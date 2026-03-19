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