from __future__ import annotations
import json
import subprocess
import shutil
import tempfile
import os
from typing import Any, Optional
import pandas as pd


def _r_is_available() -> bool:
    return shutil.which('Rscript') is not None


def _check_r_package(package: str) -> bool:
    if not _r_is_available():
        return False
    script = f'cat(requireNamespace("{package}", quietly=TRUE))'
    result = subprocess.run(
        ['Rscript', '-e', script],
        capture_output=True, text=True, timeout=10
    )
    return result.stdout.strip() == 'TRUE'


def run_r_script(r_code: str, timeout: int = 30) -> dict[str, Any]:

    if not _r_is_available():
        return {
            'success': False,
            'error': (
                'R is not installed or not on PATH. '
                'Install R from https://cran.r-project.org/ '
                'then install required packages: '
                'install.packages(c("lme4", "emmeans", "effectsize"))'
            )
        }
    
    # Write R code to temp file (avoids shell escaping issues)
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.R', delete=False, encoding='utf-8'
    ) as f:
        f.write(r_code)
        script_path = f.name
    
    try:
        result = subprocess.run(
            ['Rscript', '--vanilla', script_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode != 0:
            return {
                'success': False,
                'error': f'R script failed:\n{result.stderr.strip()}'
            }
        
        # Parse JSON from stdout
        output = result.stdout.strip()
        try:
            parsed = json.loads(output)
            parsed['success'] = True
            return parsed
        except json.JSONDecodeError:
            return {
                'success': False,
                'error': f'R output was not valid JSON:\n{output[:500]}'
            }
    
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': f'R script timed out after {timeout} seconds.'
        }
    finally:
        os.unlink(script_path)


def run_lmer(
    df: pd.DataFrame,
    outcome_col: str,
    fixed_effect_col: str,
    subject_col: str,
    random_slopes: bool = False
) -> dict[str, Any]:

    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.csv', delete=False, encoding='utf-8'
    ) as f:
        df[[outcome_col, fixed_effect_col, subject_col]].to_csv(f, index=False)
        data_path = f.name
    
    random_term = f"({fixed_effect_col}|{subject_col})" if random_slopes else f"(1|{subject_col})"
    
    r_code = f"""
suppressPackageStartupMessages({{
    library(lme4)
    library(lmerTest)    # adds p-values to lmer
    library(effectsize)
    library(emmeans)
    library(jsonlite)
}})

df <- read.csv("{data_path}")

# Fit mixed model
formula_str <- "{outcome_col} ~ {fixed_effect_col} + {random_term}"
model <- tryCatch(
    lmer(as.formula(formula_str), data=df, REML=TRUE),
    error = function(e) NULL
)

if (is.null(model)) {{
    cat(toJSON(list(
        success = FALSE,
        error = "lmer() failed to converge. Try simpler random effects structure."
    ), auto_unbox=TRUE))
    quit()
}}

# Extract fixed effects table (with p-values from lmerTest)
anova_table <- anova(model)
f_val <- anova_table["{fixed_effect_col}", "F value"]
p_val <- anova_table["{fixed_effect_col}", "Pr(>F)"]
df_num <- anova_table["{fixed_effect_col}", "NumDF"]
df_den <- anova_table["{fixed_effect_col}", "DenDF"]

# Effect size: partial eta-squared from F
partial_eta_sq <- (f_val * df_num) / (f_val * df_num + df_den)

# Random effects variance
re_var <- as.data.frame(VarCorr(model))
subject_var <- re_var[re_var$grp == "{subject_col}", "vcov"]
residual_var <- re_var[re_var$grp == "Residual", "vcov"]
icc <- subject_var / (subject_var + residual_var)

result <- list(
    test = "lmer",
    formula = formula_str,
    f_value = round(f_val, 4),
    p_value = round(p_val, 4),
    df_numerator = df_num,
    df_denominator = round(df_den, 2),
    partial_eta_squared = round(partial_eta_sq, 4),
    icc = round(icc, 4),
    n_subjects = length(unique(df${subject_col})),
    n_observations = nrow(df)
)

cat(toJSON(result, auto_unbox=TRUE))
"""
    
    try:
        result = run_r_script(r_code, timeout=60)
    finally:
        os.unlink(data_path)
    
    if result.get('success'):
        result['effect_type'] = 'partial_eta_squared'
        result['effect_size'] = result.get('partial_eta_squared', 0.0)
        result['note'] = (
            f"Linear mixed model: {outcome_col} ~ {fixed_effect_col} + {random_term}. "
            f"ICC = {result.get('icc', 'N/A')} (proportion of variance due to subjects)."
        )
    
    return result


def check_r_environment() -> dict[str, Any]:

    status = {
        'r_available': _r_is_available(),
        'packages': {}
    }
    
    if not status['r_available']:
        status['message'] = (
            'R not found. R-based tests (mixed models, advanced post-hoc) '
            'will be unavailable. Install R from https://cran.r-project.org/'
        )
        return status
    
    required_packages = ['lme4', 'lmerTest', 'emmeans', 'effectsize', 'jsonlite']
    missing = []
    
    for pkg in required_packages:
        available = _check_r_package(pkg)
        status['packages'][pkg] = available
        if not available:
            missing.append(pkg)
    
    if missing:
        status['message'] = (
            f"R found but missing packages: {', '.join(missing)}. "
            f"Install with: install.packages(c({', '.join(repr(p) for p in missing)}))"
        )
    else:
        status['message'] = 'R environment fully configured.'
    
    return status