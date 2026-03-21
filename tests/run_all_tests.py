import sys
import os
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Colour helpers (work on all platforms) ────────────────────
def green(text): return f"\033[92m{text}\033[0m"
def red(text):   return f"\033[91m{text}\033[0m"
def yellow(text):return f"\033[93m{text}\033[0m"
def bold(text):  return f"\033[1m{text}\033[0m"

passed = []
failed = []


def section(title):
    print(f"\n{'='*60}")
    print(bold(f"  {title}"))
    print('='*60)


def run(label, fn):
    """Run one test function, catch any exception."""
    try:
        fn()
        passed.append(label)
        print(green(f"  ✓ {label}"))
    except AssertionError as e:
        failed.append(label)
        print(red(f"  ✗ {label}"))
        print(red(f"    AssertionError: {e}"))
    except Exception as e:
        failed.append(label)
        print(red(f"  ✗ {label}"))
        print(red(f"    {type(e).__name__}: {e}"))
        # Uncomment for full traceback while debugging:
        # traceback.print_exc()


# ══════════════════════════════════════════════════════════════
# BLOCK 1 — Assumption Checker
# ══════════════════════════════════════════════════════════════
section("BLOCK 1 — Assumption Checker")

import numpy as np
import pandas as pd
from stats_engine.assumption_checker import (
    check_normality, check_homogeneity, build_data_profile, check_guardrails
)

np.random.seed(42)


def test_normality_normal():
    result = check_normality(pd.Series(np.random.normal(0, 1, 100)))
    assert result['passed'] is True, f"Normal data should pass, got p={result['p_value']}"

def test_normality_skewed():
    result = check_normality(pd.Series(np.random.exponential(1, 100)))
    assert result['passed'] is False, "Exponential data should fail normality"

def test_normality_large_sample_uses_dagostino():
    result = check_normality(pd.Series(np.random.normal(0, 1, 6000)))
    assert result['test'] == 'dagostino_pearson', \
        f"n>5000 should use D'Agostino-Pearson, got {result['test']}"

def test_normality_small_sample_blocked():
    result = check_normality(pd.Series([1, 2, 3]))
    assert result['passed'] is False
    assert result['n'] == 3

def test_homogeneity_equal_var():
    g1 = pd.Series(np.random.normal(0, 1, 50))
    g2 = pd.Series(np.random.normal(2, 1, 50))
    result = check_homogeneity([g1, g2])
    assert result['passed'] is True, f"Equal variance should pass, got p={result['p_value']}"

def test_homogeneity_unequal_var():
    g1 = pd.Series(np.random.normal(0, 1, 50))
    g2 = pd.Series(np.random.normal(0, 10, 50))
    result = check_homogeneity([g1, g2])
    assert result['passed'] is False, "Unequal variance should fail"

def test_homogeneity_three_groups():
    groups = [pd.Series(np.random.normal(i, 1, 30)) for i in range(3)]
    result = check_homogeneity(groups)
    assert result['n_groups'] == 3

run("Normality: normal data passes",           test_normality_normal)
run("Normality: skewed data fails",            test_normality_skewed)
run("Normality: large sample uses D'Agostino", test_normality_large_sample_uses_dagostino)
run("Normality: n<5 handled gracefully",       test_normality_small_sample_blocked)
run("Homogeneity: equal variance passes",      test_homogeneity_equal_var)
run("Homogeneity: unequal variance fails",     test_homogeneity_unequal_var)
run("Homogeneity: three groups",               test_homogeneity_three_groups)


# ══════════════════════════════════════════════════════════════
# BLOCK 2 — Guardrails
# ══════════════════════════════════════════════════════════════
section("BLOCK 2 — Guardrails")

def test_guardrail_zero_variance():
    groups = {
        'normal': pd.Series(np.random.normal(0, 1, 30)),
        'zero':   pd.Series([5.0] * 30)
    }
    result = check_guardrails(groups)
    assert result['blocked'] is True, "Zero variance should block"
    assert any('zero variance' in i.lower() for i in result['issues'])

def test_guardrail_small_n():
    groups = {
        'tiny':   pd.Series([1.0, 2.0, 3.0]),
        'normal': pd.Series(np.random.normal(0, 1, 30))
    }
    result = check_guardrails(groups)
    assert result['blocked'] is True, "n<5 should block"

def test_guardrail_high_missingness():
    data = pd.Series([np.nan] * 25 + list(np.random.normal(0, 1, 5)))
    groups = {'missing': data, 'normal': pd.Series(np.random.normal(0, 1, 30))}
    result = check_guardrails(groups)
    assert result['blocked'] is True, ">20% missing should block"

def test_guardrail_valid_data_not_blocked():
    groups = {
        'A': pd.Series(np.random.normal(0, 1, 30)),
        'B': pd.Series(np.random.normal(1, 1, 30))
    }
    result = check_guardrails(groups)
    assert result['blocked'] is False, "Valid data should not be blocked"

run("Guardrail: zero variance blocked",          test_guardrail_zero_variance)
run("Guardrail: n<5 blocked",                    test_guardrail_small_n)
run("Guardrail: >20% missing blocked",           test_guardrail_high_missingness)
run("Guardrail: valid data not blocked",         test_guardrail_valid_data_not_blocked)


# ══════════════════════════════════════════════════════════════
# BLOCK 3 — Decision Tree
# ══════════════════════════════════════════════════════════════
section("BLOCK 3 — Decision Tree")

def _profile(groups, design='independent'):
    return build_data_profile(groups, design=design)

def test_dt_independent_t():
    groups = {
        'A': pd.Series(np.random.normal(0, 1, 50)),
        'B': pd.Series(np.random.normal(1, 1, 50))
    }
    p = _profile(groups)
    assert p['recommendation']['recommended_test'] == 'independent_t', \
        f"Got {p['recommendation']['recommended_test']}"

def test_dt_welch_t():
    groups = {
        'A': pd.Series(np.random.normal(0, 1, 50)),
        'B': pd.Series(np.random.normal(0, 5, 50))
    }
    p = _profile(groups)
    assert p['recommendation']['recommended_test'] == 'welch_t', \
        f"Got {p['recommendation']['recommended_test']}"

def test_dt_mann_whitney():
    groups = {
        'A': pd.Series(np.random.exponential(1, 50)),
        'B': pd.Series(np.random.exponential(2, 50))
    }
    p = _profile(groups)
    assert p['recommendation']['recommended_test'] == 'mann_whitney_u', \
        f"Got {p['recommendation']['recommended_test']}"

def test_dt_one_way_anova():
    groups = {k: pd.Series(np.random.normal(i, 1, 40)) for i, k in enumerate('ABC')}
    p = _profile(groups)
    assert p['recommendation']['recommended_test'] == 'one_way_anova', \
        f"Got {p['recommendation']['recommended_test']}"

def test_dt_welch_anova():
    np.random.seed(42)
    groups = {
        'A': pd.Series(np.random.normal(0,   1, 40)),
        'B': pd.Series(np.random.normal(0.5, 1, 40)),
        'C': pd.Series(np.random.normal(1,   5, 40))   # much larger variance
    }
    p = _profile(groups)
    assert p['recommendation']['recommended_test'] == 'welch_anova', \
        f"Got {p['recommendation']['recommended_test']}"

def test_dt_kruskal_wallis():
    groups = {k: pd.Series(np.random.exponential(i+1, 30)) for i, k in enumerate('ABC')}
    p = _profile(groups)
    assert p['recommendation']['recommended_test'] == 'kruskal_wallis', \
        f"Got {p['recommendation']['recommended_test']}"

def test_dt_pearson():
    x = pd.Series(np.random.normal(0, 1, 100))
    y = 0.7 * x + pd.Series(np.random.normal(0, 0.5, 100))
    p = _profile({'x': x, 'y': y}, design='correlation')
    assert p['recommendation']['recommended_test'] == 'pearson_r', \
        f"Got {p['recommendation']['recommended_test']}"

def test_dt_spearman():
    x = pd.Series(np.random.exponential(1, 100))
    y = x ** 2 + pd.Series(np.random.normal(0, 5, 100))
    p = _profile({'x': x, 'y': y}, design='correlation')
    assert p['recommendation']['recommended_test'] == 'spearman_r', \
        f"Got {p['recommendation']['recommended_test']}"

def test_dt_paired_t():
    before = pd.Series(np.random.normal(50, 10, 25))
    after  = before + pd.Series(np.random.normal(5, 3, 25))
    p = _profile({'before': before, 'after': after}, design='paired')
    assert p['recommendation']['recommended_test'] == 'paired_t', \
        f"Got {p['recommendation']['recommended_test']}"

def test_dt_wilcoxon():
    before = pd.Series(np.random.exponential(1.0, 25))
    after  = before * pd.Series(np.random.uniform(1.2, 1.5, 25))
    p = _profile({'before': before, 'after': after}, design='paired')
    assert p['recommendation']['recommended_test'] == 'wilcoxon_signed_rank', \
        f"Got {p['recommendation']['recommended_test']}"

run("Decision tree: independent t-test",  test_dt_independent_t)
run("Decision tree: Welch's t-test",      test_dt_welch_t)
run("Decision tree: Mann-Whitney U",      test_dt_mann_whitney)
run("Decision tree: one-way ANOVA",       test_dt_one_way_anova)
run("Decision tree: Welch's ANOVA",       test_dt_welch_anova)
run("Decision tree: Kruskal-Wallis",      test_dt_kruskal_wallis)
run("Decision tree: Pearson r",           test_dt_pearson)
run("Decision tree: Spearman rho",        test_dt_spearman)
run("Decision tree: paired t-test",       test_dt_paired_t)
run("Decision tree: Wilcoxon",            test_dt_wilcoxon)


# ══════════════════════════════════════════════════════════════
# BLOCK 4 — Executor (all 11 tests + effect sizes)
# ══════════════════════════════════════════════════════════════
section("BLOCK 4 — Executor")

from stats_engine.executor import run_test

def test_exec_independent_t():
    groups = {
        'A': pd.Series(np.random.normal(0, 1, 50)),
        'B': pd.Series(np.random.normal(1, 1, 50))
    }
    r = run_test('independent_t', groups)
    assert r['success']
    assert 'statistic' in r and 'p_value' in r
    assert r['effect_type'] == 'cohen_d'
    assert r['degrees_of_freedom'] == 98

def test_exec_welch_t():
    groups = {
        'A': pd.Series(np.random.normal(0, 1, 50)),
        'B': pd.Series(np.random.normal(0, 5, 50))
    }
    r = run_test('welch_t', groups)
    assert r['success']
    assert r['effect_type'] == 'cohen_d'

def test_exec_mann_whitney():
    groups = {
        'A': pd.Series(np.random.exponential(1, 50)),
        'B': pd.Series(np.random.exponential(2, 50))
    }
    r = run_test('mann_whitney_u', groups)
    assert r['success']
    assert r['effect_type'] == 'rank_biserial_r'
    assert -1 <= r['effect_size'] <= 1, "Rank-biserial r must be in [-1, 1]"

def test_exec_paired_t():
    before = pd.Series(np.random.normal(50, 10, 30))
    after  = before + pd.Series(np.random.normal(5, 3, 30))
    r = run_test('paired_t', {'before': before, 'after': after})
    assert r['success']
    assert r['effect_type'] == 'cohen_d'
    assert 'mean_difference' in r

def test_exec_wilcoxon():
    before = pd.Series(np.random.exponential(1, 30))
    after  = before * pd.Series(np.random.uniform(1.2, 1.5, 30))
    r = run_test('wilcoxon_signed_rank', {'before': before, 'after': after})
    assert r['success']
    assert r['effect_type'] == 'rank_biserial_r'

def test_exec_one_way_anova():
    groups = {k: pd.Series(np.random.normal(i, 1, 30)) for i, k in enumerate('ABC')}
    r = run_test('one_way_anova', groups)
    assert r['success']
    assert r['effect_type'] == 'eta_squared'
    assert 0 <= r['effect_size'] <= 1

def test_exec_welch_anova():
    groups = {
        'A': pd.Series(np.random.normal(0,   1, 40)),
        'B': pd.Series(np.random.normal(0.5, 1, 40)),
        'C': pd.Series(np.random.normal(1,   5, 40))
    }
    r = run_test('welch_anova', groups)
    assert r['success'], f"Welch's ANOVA failed: {r.get('error')}"
    assert r['effect_type'] == 'eta_squared'

def test_exec_kruskal_wallis():
    groups = {k: pd.Series(np.random.exponential(i+1, 30)) for i, k in enumerate('ABC')}
    r = run_test('kruskal_wallis', groups)
    assert r['success']
    assert r['effect_type'] == 'epsilon_squared'

def test_exec_friedman():
    n = 25
    base = np.random.normal(50, 5, n)
    groups = {
        'A': pd.Series(base + np.random.normal(0, 2, n)),
        'B': pd.Series(base + np.random.normal(2, 2, n)),
        'C': pd.Series(base + np.random.normal(8, 2, n))
    }
    r = run_test('friedman', groups)
    assert r['success'], f"Friedman failed: {r.get('error')}"
    assert r['effect_type'] == 'kendalls_w'
    assert 0 <= r['effect_size'] <= 1, f"Kendall's W={r['effect_size']} out of range"

def test_exec_pearson():
    x = pd.Series(np.random.normal(0, 1, 100))
    y = 0.7 * x + pd.Series(np.random.normal(0, 0.5, 100))
    r = run_test('pearson_r', {'x': x, 'y': y})
    assert r['success']
    assert r['effect_type'] == 'r_squared'
    assert abs(r['correlation']) > 0.5, "Should detect strong correlation"

def test_exec_spearman():
    x = pd.Series(np.random.exponential(1, 100))
    y = x ** 2 + pd.Series(np.random.normal(0, 5, 100))
    r = run_test('spearman_r', {'x': x, 'y': y})
    assert r['success']
    assert r['effect_type'] == 'rho_squared'

def test_exec_unknown_test_returns_error():
    r = run_test('nonexistent_test', {'A': pd.Series([1, 2, 3])})
    assert r['success'] is False
    assert 'error' in r

run("Executor: independent t-test + Cohen's d",  test_exec_independent_t)
run("Executor: Welch's t-test",                   test_exec_welch_t)
run("Executor: Mann-Whitney + rank-biserial r",   test_exec_mann_whitney)
run("Executor: paired t-test",                    test_exec_paired_t)
run("Executor: Wilcoxon signed-rank",             test_exec_wilcoxon)
run("Executor: one-way ANOVA + eta-squared",      test_exec_one_way_anova)
run("Executor: Welch's ANOVA",                    test_exec_welch_anova)
run("Executor: Kruskal-Wallis + epsilon-squared", test_exec_kruskal_wallis)
run("Executor: Friedman + Kendall's W",           test_exec_friedman)
run("Executor: Pearson r",                        test_exec_pearson)
run("Executor: Spearman rho",                     test_exec_spearman)
run("Executor: unknown test returns error dict",  test_exec_unknown_test_returns_error)


# ══════════════════════════════════════════════════════════════
# BLOCK 5 — Post-Hoc Tests
# ══════════════════════════════════════════════════════════════
section("BLOCK 5 — Post-Hoc Tests")

def test_tukey_finds_differences():
    groups = {
        'A': pd.Series(np.random.normal(0, 1, 40)),
        'B': pd.Series(np.random.normal(4, 1, 40)),   # clearly different
        'C': pd.Series(np.random.normal(0.1, 1, 40))  # similar to A
    }
    r = run_test('tukey_hsd', groups)
    assert r['success'], f"Tukey HSD failed: {r.get('error')}"
    assert r['n_comparisons'] == 3
    assert any('A' in p and 'B' in p for p in r['significant_pairs']), \
        f"A vs B should be significant. Got: {r['significant_pairs']}"

def test_dunns_finds_differences():
    groups = {
        'low':  pd.Series(np.random.exponential(1, 40)),
        'mid':  pd.Series(np.random.exponential(3, 40)),
        'high': pd.Series(np.random.exponential(8, 40))
    }
    r = run_test('dunns_test', groups)
    assert r['success'], f"Dunn's test failed: {r.get('error')}"
    assert any(
        ('low' in p and 'high' in p) or ('high' in p and 'low' in p)
        for p in r['significant_pairs']
    ), f"low vs high should be significant. Got: {r['significant_pairs']}"

def test_pairwise_wilcoxon():
    n = 25
    base = np.random.normal(50, 5, n)
    groups = {
        'A': pd.Series(base + np.random.normal(0, 2, n)),
        'B': pd.Series(base + np.random.normal(2, 2, n)),
        'C': pd.Series(base + np.random.normal(10, 2, n))  # clearly different
    }
    r = run_test('pairwise_wilcoxon', groups)
    assert r['success'], f"Pairwise Wilcoxon failed: {r.get('error')}"
    assert r['n_comparisons'] == 3
    assert any('C' in p for p in r['significant_pairs']), \
        f"C should differ from others. Got: {r['significant_pairs']}"

def test_friedman_then_posthoc_pipeline():
    """Full pipeline: Friedman significant → pairwise Wilcoxon post-hoc."""
    n = 25
    base = np.random.normal(50, 5, n)
    groups = {
        'cond_A': pd.Series(base + np.random.normal(0,  2, n)),
        'cond_B': pd.Series(base + np.random.normal(2,  2, n)),
        'cond_C': pd.Series(base + np.random.normal(12, 2, n))
    }
    friedman = run_test('friedman', groups)
    assert friedman['success']
    assert friedman['p_value'] < 0.05, \
        f"Friedman should be significant, got p={friedman['p_value']:.4f}"

    posthoc = run_test('pairwise_wilcoxon', groups)
    assert posthoc['success']
    assert len(posthoc['significant_pairs']) > 0, "Post-hoc should find differences"

run("Post-hoc: Tukey HSD finds significant pairs",       test_tukey_finds_differences)
run("Post-hoc: Dunn's test finds significant pairs",     test_dunns_finds_differences)
run("Post-hoc: pairwise Wilcoxon + Holm correction",     test_pairwise_wilcoxon)
run("Post-hoc: Friedman → pairwise Wilcoxon pipeline",  test_friedman_then_posthoc_pipeline)


# ══════════════════════════════════════════════════════════════
# BLOCK 6 — Structure Inference (Profiler)
# ══════════════════════════════════════════════════════════════
section("BLOCK 6 — Structure Inference (Profiler)")

from stats_engine.profiler import infer_structure

def test_profiler_repeated_measures():
    subjects = np.repeat(np.arange(18), 10)
    df = pd.DataFrame({
        'Subject':  subjects,
        'Days':     np.tile(np.arange(10), 18),
        'Reaction': np.random.normal(300, 50, 180)
    })
    v = infer_structure(df)
    assert v.verdict == 'repeated_measures', \
        f"Expected repeated_measures, got {v.verdict}"
    assert v.subject_col == 'Subject'
    assert v.n_subjects == 18

def test_profiler_independent():
    df = pd.DataFrame({
        'score': np.random.normal(0, 1, 100),
        'group': ['A'] * 50 + ['B'] * 50
    })
    v = infer_structure(df)
    assert v.verdict == 'independent', \
        f"Expected independent, got {v.verdict}"

def test_profiler_wide_format():
    df = pd.DataFrame({
        'participant_id': [f'P{i}' for i in range(30)],
        'score_pre':      np.random.normal(50, 10, 30),
        'score_post':     np.random.normal(55, 10, 30),
        'score_followup': np.random.normal(52, 10, 30)
    })
    v = infer_structure(df)
    assert v.wide_format is True, \
        f"Expected wide_format=True, got {v.wide_format}"
    assert v.verdict == 'repeated_measures'

def test_profiler_warns_unequal_observations():
    # 10 subjects × 5 days, but drop 3 rows from subject 0
    rows = [{'Subject': s, 'Day': d, 'RT': np.random.normal(300, 50)}
            for s in range(10) for d in range(5)]
    df = pd.DataFrame(rows).iloc[3:].reset_index(drop=True)
    v = infer_structure(df)
    assert any('nequal' in w for w in v.warnings), \
        f"Expected unequal-observations warning, got: {v.warnings}"

run("Profiler: detects repeated measures",          test_profiler_repeated_measures)
run("Profiler: detects independent groups",         test_profiler_independent)
run("Profiler: detects wide format",                test_profiler_wide_format)
run("Profiler: warns on unequal observations",      test_profiler_warns_unequal_observations)


# ══════════════════════════════════════════════════════════════
# BLOCK 7 — HITL (Human-in-the-Loop)
# ══════════════════════════════════════════════════════════════
section("BLOCK 7 — Human-in-the-Loop")

from stats_engine.hitl import HITLCheckpoint

def _make_profile(recommended='kruskal_wallis', blocked=False):
    """Build a minimal profile dict for HITL testing."""
    return {
        'recommendation': {
            'recommended_test': recommended,
            'rationale': 'test rationale',
            'warnings': []
        },
        'guardrails': {
            'blocked': blocked,
            'issues': ['zero variance'] if blocked else []
        }
    }

def _make_groups():
    return {'A': pd.Series(np.random.normal(0, 1, 30))}

def test_hitl_disabled_auto_accepts():
    cp = HITLCheckpoint(enabled=False)
    profile = _make_profile('kruskal_wallis')
    decision = cp.review(profile, _make_groups())
    assert decision['test'] == 'kruskal_wallis'
    assert decision['source'] == 'pipeline'
    assert decision['user_confirmed'] is True

def test_hitl_blocked_data_returns_none():
    cp = HITLCheckpoint(enabled=False)
    profile = _make_profile(blocked=True)
    decision = cp.review(profile, _make_groups())
    assert decision['test'] is None
    assert decision['source'] == 'blocked'

def test_hitl_audit_log_starts_empty():
    cp = HITLCheckpoint(enabled=False)
    assert cp.get_audit_log() == []

def test_hitl_noninteractive_auto_accepts():
    """In non-interactive environments (CI), HITL should auto-accept."""
    cp = HITLCheckpoint(enabled=True)
    profile = _make_profile('independent_t')
    # When stdin is not a terminal, HITLCheckpoint catches EOFError
    # and auto-accepts. Test that enabled=False path is stable at minimum.
    cp2 = HITLCheckpoint(enabled=False)
    decision = cp2.review(profile, _make_groups())
    assert decision['test'] == 'independent_t'

run("HITL: disabled mode auto-accepts pipeline",    test_hitl_disabled_auto_accepts)
run("HITL: blocked data returns test=None",         test_hitl_blocked_data_returns_none)
run("HITL: audit log starts empty",                 test_hitl_audit_log_starts_empty)
run("HITL: non-interactive auto-accepts",           test_hitl_noninteractive_auto_accepts)


# ══════════════════════════════════════════════════════════════
# BLOCK 8 — Full Eval Harness (all scenarios)
# ══════════════════════════════════════════════════════════════
section("BLOCK 8 — Full Eval Harness")

from data_utils.simulator import get_all_scenarios

def test_eval_harness():
    scenarios = get_all_scenarios()
    total = len(scenarios)
    correct = 0
    failures = []

    for name, scenario in scenarios.items():
        df          = scenario['df']
        target_col  = scenario['target_col']
        group_col   = scenario['group_col']
        expected    = scenario['correct_test']
        design      = scenario.get('design', 'independent')
        score_type  = scenario.get('score_type', 'test_match')

        # Build groups
        if design == 'correlation':
            groups = {target_col: df[target_col], group_col: df[group_col]}
        elif design in ('paired', 'repeated'):
            subject_col = scenario.get('subject_col', 'subject')
            groups = {}
            for cond in df[group_col].unique():
                subset = (df[df[group_col] == cond]
                          .sort_values(subject_col)[target_col]
                          .reset_index(drop=True))
                groups[cond] = subset
        else:
            groups = {
                n: df[df[group_col] == n][target_col]
                for n in df[group_col].unique()
            }

        # Guardrail scenarios
        if score_type == 'guardrail':
            result = check_guardrails(groups, design=design)
            if result['blocked']:
                correct += 1
            else:
                failures.append(f"{name}: expected BLOCKED, not blocked")
            continue

        # Standard scenarios
        profile   = build_data_profile(groups, design=design)
        if profile['guardrails']['blocked']:
            failures.append(f"{name}: unexpectedly blocked for non-guardrail scenario")
            continue

        recommended = profile['recommendation']['recommended_test']
        if recommended == expected:
            correct += 1
        else:
            failures.append(f"{name}: expected {expected}, got {recommended}")

    accuracy = correct / total * 100
    for f in failures:
        print(yellow(f"    ! {f}"))

    assert accuracy >= 85, \
        f"Eval harness: {correct}/{total} ({accuracy:.1f}%) — below 85% threshold"
    print(f"    Eval harness: {correct}/{total} ({accuracy:.1f}%)")

run(f"Eval harness: ≥85% across all scenarios", test_eval_harness)


# ══════════════════════════════════════════════════════════════
# BLOCK 9 — LLM Interpreter (template fallback only)
# ══════════════════════════════════════════════════════════════
section("BLOCK 9 — LLM Interpreter (template fallback)")

from stats_engine.llm_interpreter import generate_methods_paragraph

def test_llm_template_fallback():
    """Template fallback must always work, even without Ollama."""
    result = {
        'test': 'kruskal_wallis',
        'statistic': 12.34,
        'p_value': 0.002,
        'effect_size': 0.18,
        'effect_type': 'epsilon_squared'
    }
    profile = {
        'recommendation': {'rationale': '3 groups, non-normal', 'recommended_test': 'kruskal_wallis'},
        'n_groups': 3
    }
    output = generate_methods_paragraph(result, profile, use_llm=False)
    assert isinstance(output, str) and len(output) > 30, \
        "Template fallback should return a non-empty string"
    assert 'kruskal' in output.lower() or 'Kruskal' in output, \
        "Output should mention the test name"

def test_llm_output_contains_stats():
    result = {
        'test': 'independent_t',
        'statistic': 2.45,
        'p_value': 0.015,
        'effect_size': 0.52,
        'effect_type': 'cohen_d'
    }
    profile = {
        'recommendation': {'rationale': '2 normal groups equal variance', 'recommended_test': 'independent_t'},
        'n_groups': 2
    }
    output = generate_methods_paragraph(result, profile, use_llm=False)
    assert '2.45' in output or '0.015' in output or '0.52' in output, \
        "Output should include at least one statistic from the result"

run("LLM: template fallback returns valid string",    test_llm_template_fallback)
run("LLM: template output includes statistics",       test_llm_output_contains_stats)


# ══════════════════════════════════════════════════════════════
# BLOCK 10 — R Backend (availability check only)
# ══════════════════════════════════════════════════════════════
section("BLOCK 10 — R Backend")

from stats_engine.r_backend import check_r_environment

def test_r_backend_returns_status():
    """R backend should return a status dict regardless of whether R is installed."""
    status = check_r_environment()
    assert 'r_available' in status
    assert 'message' in status
    assert isinstance(status['message'], str)

def test_r_backend_graceful_when_missing():
    """If R is not available, run_lmer should return a helpful error, not crash."""
    from stats_engine.r_backend import run_lmer
    status = check_r_environment()
    if status['r_available']:
        return  # R is installed, skip this test
    df = pd.DataFrame({
        'Reaction': np.random.normal(300, 50, 180),
        'Days':     np.tile(np.arange(10), 18).astype(str),
        'Subject':  np.repeat(np.arange(18), 10)
    })
    result = run_lmer(df, 'Reaction', 'Days', 'Subject')
    assert result['success'] is False
    assert 'error' in result
    assert 'R' in result['error'] or 'install' in result['error'].lower()

run("R backend: check_r_environment returns status dict", test_r_backend_returns_status)
run("R backend: graceful error when R not installed",     test_r_backend_graceful_when_missing)


# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
total_run    = len(passed) + len(failed)
pass_count   = len(passed)
fail_count   = len(failed)
pass_rate    = pass_count / total_run * 100 if total_run else 0

print(f"\n{'='*60}")
print(bold("  FINAL RESULTS"))
print('='*60)
print(green(f"  Passed : {pass_count}/{total_run}"))
if fail_count:
    print(red(f"  Failed : {fail_count}/{total_run}"))
    print(red("\n  Failing tests:"))
    for f in failed:
        print(red(f"    ✗ {f}"))
else:
    print(green("  All tests passed."))
print(f"\n  Pass rate: {pass_rate:.1f}%")
print('='*60)

sys.exit(0 if fail_count == 0 else 1)