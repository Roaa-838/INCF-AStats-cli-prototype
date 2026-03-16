import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Any


def test_homogeneity(
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
    

def test_normality(data: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:

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