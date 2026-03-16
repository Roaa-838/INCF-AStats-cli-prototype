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