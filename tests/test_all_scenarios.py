import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stats_engine.assumption_checker import build_data_profile, check_guardrails
from data_utils.simulator import get_all_scenarios

scenarios = get_all_scenarios()
total = len(scenarios)
correct_count = 0
results_log = []


def build_groups(scenario):

    df = scenario['df']
    target_col = scenario['target_col']
    group_col = scenario['group_col']
    design = scenario.get('design', 'independent')

    if design == 'correlation':
        return {target_col: df[target_col], group_col: df[group_col]}, design

    if design in ('paired', 'repeated'):
        subject_col = scenario.get('subject_col', 'subject')
        # Pivot: one series per condition, aligned by subject
        groups = {}
        for cond in df[group_col].unique():
            subset = (
                df[df[group_col] == cond]
                .sort_values(subject_col)[target_col]
                .reset_index(drop=True)
            )
            groups[cond] = subset
        return groups, design

    # Default: independent
    groups = {
        name: df[df[group_col] == name][target_col]
        for name in df[group_col].unique()
    }
    return groups, 'independent'


for scenario_name, scenario in scenarios.items():
    description = scenario['description']
    expected_test = scenario['correct_test']
    score_type = scenario.get('score_type', 'test_match')

    groups, design = build_groups(scenario)

    # ── Guardrail scenarios ────────────────────────────────────
    if score_type == 'guardrail':
        guardrail = check_guardrails(groups, design=design)
        blocked = guardrail['blocked']
        correct = blocked  # correct means the pipeline blocked it

        status = 'CORRECT (blocked as expected)' if correct else 'WRONG (should have been blocked)'
        print(f"Scenario : {description}")
        print(f"Expected : BLOCKED")
        print(f"Result   : {'BLOCKED' if blocked else 'NOT blocked'}")
        print(f"{status}")
        if not blocked:
            print(f"  Issues found: {guardrail['issues']}")
        print()

        if correct:
            correct_count += 1
        results_log.append({
            'name': scenario_name,
            'expected': 'BLOCKED',
            'got': 'BLOCKED' if blocked else 'NOT_BLOCKED',
            'correct': correct,
            'score_type': 'guardrail'
        })
        continue

    # ── Standard test-match scenarios ─────────────────────────
    profile = build_data_profile(groups, design=design)

    # Check guardrails first — if blocked, that's a failure for
    # non-guardrail scenarios (we shouldn't block valid analyses)
    if profile['guardrails']['blocked']:
        print(f"Scenario : {description}")
        print(f"Expected : {expected_test}")
        print(f"Result   : BLOCKED (unexpected)")
        print(f"WRONG — pipeline blocked a valid scenario")
        print(f"  Issues: {profile['guardrails']['issues']}")
        print()
        results_log.append({
            'name': scenario_name,
            'expected': expected_test,
            'got': 'BLOCKED',
            'correct': False,
            'score_type': 'test_match'
        })
        continue

    recommended = profile['recommendation']['recommended_test']
    correct = (recommended == expected_test)

    if correct:
        correct_count += 1

    status = 'CORRECT' if correct else 'WRONG'
    print(f"Scenario : {description}")
    print(f"Expected : {expected_test}")
    print(f"Got      : {recommended}")
    print(f"{status}")
    print(f"Rationale: {profile['recommendation']['rationale']}")
    print()

    results_log.append({
        'name': scenario_name,
        'expected': expected_test,
        'got': recommended,
        'correct': correct,
        'score_type': 'test_match'
    })


# ── Summary ────────────────────────────────────────────────────
accuracy = (correct_count / total) * 100
print("=" * 60)
print(f"RESULTS: {correct_count}/{total} ({accuracy:.1f}%)")
print("=" * 60)

# Break down by type so you can see where failures are
test_match_results = [r for r in results_log if r['score_type'] == 'test_match']
guardrail_results  = [r for r in results_log if r['score_type'] == 'guardrail']

if test_match_results:
    tm_correct = sum(1 for r in test_match_results if r['correct'])
    print(f"  Test selection : {tm_correct}/{len(test_match_results)}")

if guardrail_results:
    gr_correct = sum(1 for r in guardrail_results if r['correct'])
    print(f"  Guardrails     : {gr_correct}/{len(guardrail_results)}")

failures = [r for r in results_log if not r['correct']]
if failures:
    print(f"\nFailed scenarios:")
    for f in failures:
        print(f"  {f['name']}: expected '{f['expected']}', got '{f['got']}'")

print()
if accuracy == 100:
    print("All scenarios passing.")
elif accuracy >= 85:
    print("PASSING — above 85% threshold")
elif accuracy >= 70:
    print("PASSING — above 70% threshold but investigate failures")
else:
    print("FAILING — below 70%, debug needed")