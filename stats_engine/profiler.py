from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class StructureVerdict:
    verdict: str                          # 'independent' | 'repeated_measures' | 'unknown'
    subject_col: Optional[str]            # detected subject ID column
    condition_col: Optional[str]          # detected condition/group column
    wide_format: bool                     # True if data is one-row-per-subject
    n_subjects: Optional[int]             # unique subjects detected
    obs_per_subject: Optional[dict]       # how many obs each subject has
    confidence: str                       # 'high' | 'medium' | 'low'
    warnings: list = field(default_factory=list)
    needs_clarification: Optional[str] = None


def _is_likely_subject_col(series: pd.Series, df_len: int) -> bool:
 
    n_unique = series.nunique(dropna=True)
    unique_ratio = n_unique / df_len
    
    # Too many unique values → probably a measurement, not a subject ID
    if unique_ratio > 0.7:
        return False
    
    # Too few unique values → probably a treatment/condition column
    if unique_ratio < 0.02:
        return False
    
    # Check if values repeat (each subject appears multiple times)
    value_counts = series.value_counts()
    mean_reps = value_counts.mean()
    
    # Subject ID columns typically have each value appearing 2+ times
    return mean_reps >= 2.0


def _looks_like_subject_name(col_name: str) -> bool:

    name_lower = col_name.lower().strip()
    subject_keywords = [
        'subject', 'subj', 'participant', 'id', 'pid',
        'patient', 'animal', 'mouse', 'rat', 'monkey',
        'subject_id', 'sub', 'ppid', 'respondent'
    ]
    return any(kw in name_lower for kw in subject_keywords)


def _looks_like_condition_name(col_name: str) -> bool:

    name_lower = col_name.lower().strip()
    condition_keywords = [
        'condition', 'session', 'trial', 'time', 'day',
        'week', 'phase', 'period', 'block', 'run',
        'group', 'treatment', 'stimulus', 'cue'
    ]
    return any(kw in name_lower for kw in condition_keywords)


def _detect_wide_format(df: pd.DataFrame) -> tuple[bool, list[str]]:
 
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return False, []
    
    prefixes = {}
    for col in numeric_cols:
        for sep in ['_', '-', '.']:
            parts = col.split(sep)
            if len(parts) >= 2:
                prefix = sep.join(parts[:-1])
                prefixes.setdefault(prefix, []).append(col)
    
    wide_groups = {k: v for k, v in prefixes.items() if len(v) >= 2}
    
    if wide_groups:
        best_group = max(wide_groups.values(), key=len)
        return True, best_group
    
    return False, []


def infer_structure(df: pd.DataFrame) -> StructureVerdict:

    warnings = []
    df_len = len(df)
    
    if df_len < 4:
        return StructureVerdict(
            verdict='unknown',
            subject_col=None,
            condition_col=None,
            wide_format=False,
            n_subjects=None,
            obs_per_subject=None,
            confidence='low',
            warnings=['Dataset too small to infer structure reliably (n < 4).'],
            needs_clarification='Please specify the experimental design manually.'
        )
    
    # Step 1: Check for wide format
    is_wide, wide_cols = _detect_wide_format(df)
    if is_wide:
        # Find subject column (non-numeric, or with subject-like name)
        non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
        subject_col = None
        for col in non_numeric:
            if _looks_like_subject_name(col):
                subject_col = col
                break
        if subject_col is None and non_numeric:
            subject_col = non_numeric[0]  # best guess
        
        n_subjects = df[subject_col].nunique() if subject_col else len(df)
        
        return StructureVerdict(
            verdict='repeated_measures',
            subject_col=subject_col,
            condition_col=None,
            wide_format=True,
            n_subjects=n_subjects,
            obs_per_subject=None,
            confidence='medium',
            warnings=[
                f"Wide-format repeated measures detected. "
                f"Measurement columns: {wide_cols}. "
                f"Each row appears to represent one subject measured across conditions."
            ]
        )
    
    # Step 2: Scan columns for subject ID candidates
    subject_col_candidates = []
    condition_col_candidates = []
    
    for col in df.columns:
        series = df[col]
        
        # Skip numeric columns (subject IDs are usually strings or small integers)
        is_numeric = pd.api.types.is_numeric_dtype(series)
        
        has_subject_name = _looks_like_subject_name(col)
        has_condition_name = _looks_like_condition_name(col)
        looks_like_subject = _is_likely_subject_col(series, df_len)
        
        if has_subject_name and looks_like_subject:
            subject_col_candidates.append((col, 'high'))
        elif looks_like_subject and not is_numeric:
            subject_col_candidates.append((col, 'medium'))
        elif has_subject_name:
            subject_col_candidates.append((col, 'low'))
        
        if has_condition_name:
            condition_col_candidates.append(col)
    
    # Step 3: Determine verdict
    if not subject_col_candidates:
        # No subject column detected — assume independent design
        return StructureVerdict(
            verdict='independent',
            subject_col=None,
            condition_col=condition_col_candidates[0] if condition_col_candidates else None,
            wide_format=False,
            n_subjects=None,
            obs_per_subject=None,
            confidence='medium',
            warnings=[
                "No subject ID column detected. Assuming independent groups design. "
                "If this is a repeated measures study, specify the subject column manually."
            ]
        )
    
    # Pick the best subject column candidate
    high_confidence = [c for c, conf in subject_col_candidates if conf == 'high']
    medium_confidence = [c for c, conf in subject_col_candidates if conf == 'medium']
    
    subject_col = (
        high_confidence[0] if high_confidence
        else medium_confidence[0] if medium_confidence
        else subject_col_candidates[0][0]
    )
    
    confidence = 'high' if high_confidence else 'medium'
    
    # Step 4: Analyze observations per subject
    obs_counts = df[subject_col].value_counts().to_dict()
    n_subjects = len(obs_counts)
    
    # Check if observations are balanced (equal per subject)
    unique_counts = set(obs_counts.values())
    is_balanced = len(unique_counts) == 1
    obs_per_condition = list(unique_counts)[0] if is_balanced else None
    
    if not is_balanced:
        min_obs = min(obs_counts.values())
        max_obs = max(obs_counts.values())
        warnings.append(
            f"Unequal observations per subject detected "
            f"(min={min_obs}, max={max_obs}). "
            f"This may indicate missing data. "
            f"Paired tests will be downgraded to independent-group alternatives."
        )
    
    if n_subjects == 1:
        return StructureVerdict(
            verdict='unknown',
            subject_col=subject_col,
            condition_col=None,
            wide_format=False,
            n_subjects=1,
            obs_per_subject=obs_counts,
            confidence='low',
            warnings=warnings,
            needs_clarification=(
                "Only one unique subject detected. "
                "This may be a single-case study or a data format issue."
            )
        )
    
    # Multiple observations per subject → repeated measures
    if obs_per_condition and obs_per_condition >= 2:
        condition_col = condition_col_candidates[0] if condition_col_candidates else None
        
        return StructureVerdict(
            verdict='repeated_measures',
            subject_col=subject_col,
            condition_col=condition_col,
            wide_format=False,
            n_subjects=n_subjects,
            obs_per_subject=obs_counts,
            confidence=confidence,
            warnings=warnings
        )
    
    # Each subject appears only once → independent
    return StructureVerdict(
        verdict='independent',
        subject_col=None,
        condition_col=condition_col_candidates[0] if condition_col_candidates else None,
        wide_format=False,
        n_subjects=n_subjects,
        obs_per_subject=None,
        confidence='high',
        warnings=warnings
    )