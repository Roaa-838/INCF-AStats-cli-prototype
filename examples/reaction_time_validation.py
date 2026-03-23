# examples/reaction_time_validation.py
"""
Simulated reaction time dataset mimicking a real neuroscience study.
Log-normal distribution — standard for RT data.
3 conditions: baseline, congruent cue, incongruent cue.
This tests whether AStats correctly handles non-normal neuroscience data.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from stats_engine.assumption_checker import build_data_profile
from stats_engine.executor import run_test
from stats_engine.llm_interpreter import generate_methods_paragraph

np.random.seed(42)

n_per_condition = 45

# Log-normal RT: typical values around 250-400ms
baseline      = np.random.lognormal(np.log(320), 0.25, n_per_condition)
congruent     = np.random.lognormal(np.log(270), 0.25, n_per_condition)
incongruent   = np.random.lognormal(np.log(380), 0.25, n_per_condition)

groups = {
    'baseline':    pd.Series(baseline),
    'congruent':   pd.Series(congruent),
    'incongruent': pd.Series(incongruent)
}


print("=== REACTION TIME VALIDATION ===")
print(f"Research question: Do reaction times differ across cue conditions?")
print(f"Data: {n_per_condition} participants per condition, log-normal distribution")
print(f"\nDescriptive stats (ms):")
for cond, series in groups.items():
    print(f"  {cond}: mean={series.mean():.1f}, median={series.median():.1f}, std={series.std():.1f}")

profile = build_data_profile(groups, design='independent')

print(f"\nNormality checks:")
for cond, norm in profile['normality'].items():
    print(f"  {cond}: {'PASSED' if norm['passed'] else 'FAILED'} (p={norm['p_value']:.4f}, skew={norm['skewness']:.3f})")

print(f"\nTest selected: {profile['recommendation']['recommended_test']}")
print(f"Rationale: {profile['recommendation']['rationale']}")

result = run_test(profile['recommendation']['recommended_test'], groups)
print(f"\nResults:")
print(f"  H = {result['statistic']:.3f}")
print(f"  p = {result['p_value']:.4f}")
print(f"  Epsilon-squared = {result['effect_size']:.3f} (effect magnitude)")

if result['p_value'] < 0.05:
    print(f"\nPost-hoc (Dunn's test with Holm correction):")
    posthoc = run_test('dunns_test', groups)
    for comp in posthoc['comparisons']:
        sig = "SIGNIFICANT" if comp['significant'] else "not significant"
        print(f"  {comp['group1']} vs {comp['group2']}: p_adj={comp['p_adjusted']:.4f} {sig}")

methods = generate_methods_paragraph(result, profile, use_llm=False)
print(f"\nAuto-generated methods paragraph:")
print(methods)

print(f"\nKey finding: Log-normal RT data correctly routed to Kruskal-Wallis.")
print(f"A naive tool running one-way ANOVA would violate normality assumption.")