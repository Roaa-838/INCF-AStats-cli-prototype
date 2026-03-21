import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from stats_engine.profiler import infer_structure

np.random.seed(42)


def test_detects_repeated_measures():
    """Sleepstudy-like: 18 subjects, 10 days each → repeated measures"""
    subjects = np.repeat(np.arange(18), 10)
    days = np.tile(np.arange(10), 18)
    reaction = np.random.normal(300, 50, 180)
    
    df = pd.DataFrame({
        'Subject': subjects,
        'Days': days,
        'Reaction': reaction
    })
    
    verdict = infer_structure(df)
    assert verdict.verdict == 'repeated_measures', (
        f"Expected repeated_measures, got {verdict.verdict}. "
        f"Subject column detected: {verdict.subject_col}"
    )
    assert verdict.subject_col == 'Subject'
    assert verdict.n_subjects == 18
    print(f"✓ Detected repeated measures: {verdict.n_subjects} subjects")


def test_detects_independent():
    """Standard two-group study: no repeated measures"""
    df = pd.DataFrame({
        'score': np.random.normal(0, 1, 100),
        'group': ['control'] * 50 + ['treatment'] * 50
    })
    
    verdict = infer_structure(df)
    assert verdict.verdict == 'independent', (
        f"Expected independent, got {verdict.verdict}"
    )
    print(f"✓ Correctly identified independent groups")


def test_detects_wide_format():
    """Pre/post design: one row per subject, columns for each timepoint"""
    df = pd.DataFrame({
        'participant_id': [f'P{i}' for i in range(30)],
        'score_pre': np.random.normal(50, 10, 30),
        'score_post': np.random.normal(55, 10, 30),
        'score_followup': np.random.normal(52, 10, 30)
    })
    
    verdict = infer_structure(df)
    assert verdict.wide_format == True, (
        f"Expected wide_format=True, got {verdict.wide_format}"
    )
    assert verdict.verdict == 'repeated_measures'
    print(f"✓ Detected wide-format repeated measures")


def test_warns_on_unequal_observations():
    """Missing data: some subjects have fewer observations"""
    subjects = list(np.repeat(np.arange(10), 5))  # 10 subjects × 5 days
    # Drop 3 observations from subject 0 (simulate missing data)
    subjects = subjects[3:]
    days = list(range(5)) * 10
    days = days[3:]
    reaction = np.random.normal(300, 50, len(subjects))
    
    df = pd.DataFrame({
        'Subject': subjects,
        'Day': days[:len(subjects)],
        'RT': reaction
    })
    
    verdict = infer_structure(df)
    has_unequal_warning = any('Unequal' in w for w in verdict.warnings)
    assert has_unequal_warning, (
        f"Expected unequal observations warning, got: {verdict.warnings}"
    )
    print(f"Warning issued for unequal observations: {verdict.warnings[0][:60]}...")


if __name__ == '__main__':
    test_detects_repeated_measures()
    test_detects_independent()
    test_detects_wide_format()
    test_warns_on_unequal_observations()
    print("\nAll profiler tests passed")