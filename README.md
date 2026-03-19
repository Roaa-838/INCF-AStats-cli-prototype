# AStats CLI Prototype

Statistical test selection pipeline that checks assumptions and picks the right test automatically.

Built for GSoC 2026 - INCF Project #33.

**Core components:**
- Assumption checking (Shapiro-Wilk, Levene's test)
- Decision tree for test selection
- Test execution (9 tests via scipy)
- Effect size calculations

**Current status:** The eval harness shows 8/8 on synthetic benchmarks, and I tested it on the Iris dataset to make sure it works on real data too.

## Quick Demo
```python
from data_utils.simulator import two_groups_normal_equal_var
from stats_engine.assumption_checker import build_data_profile
from stats_engine.executor import run_test

# Generate test data
df, target, group, expected_test, _ = two_groups_normal_equal_var()

# Prepare groups
groups = {
    name: df[df[group] == name][target]
    for name in df[group].unique()
}

# Run pipeline
profile = build_data_profile(groups, design='independent')
result = run_test(profile['recommendation']['recommended_test'], groups)

# Check result
print(f"Expected: {expected_test}")
print(f"Got: {profile['recommendation']['recommended_test']}")
print(f"p-value: {result['p_value']:.4f}")
print(f"Effect size: {result['effect_size']:.4f}")
```

## Tests Implemented

| Test | When to Use | Effect Size |
|------|-------------|-------------|
| Independent t-test | 2 groups, normal, equal variance | Cohen's d |
| Welch's t-test | 2 groups, normal, unequal variance | Cohen's d |
| Mann-Whitney U | 2 groups, non-normal | Rank-biserial r |
| Paired t-test | Paired data, normal differences | Cohen's d |
| Wilcoxon signed-rank | Paired data, non-normal | Rank-biserial r |
| One-way ANOVA | 3+ groups, normal, equal variance | Eta-squared |
| Kruskal-Wallis | 3+ groups, non-normal | Epsilon-squared |
| Pearson correlation | Both variables normal | r-squared |
| Spearman correlation | Non-normal data | rho-squared |

## Eval Harness Results

Testing on synthetic scenarios (where I know the right answer):
```
Scenario: Two normal groups, equal variance
  Expected: independent_t
  Got: independent_t 

Scenario: Two normal groups, unequal variance  
  Expected: welch_t
  Got: welch_t 

Scenario: Two non-normal groups
  Expected: mann_whitney_u
  Got: mann_whitney_u 

... (5 more scenarios)

RESULTS: 8/8 (100%)
```

Also tested on Iris dataset - correctly identified that variance was unequal across species and routed to Kruskal-Wallis (p < 0.001, large effect size).

## How the Decision Tree Works

The routing is conservative - if ANY group fails an assumption, use the robust alternative:
```
For 2 independent groups:
  - Both normal + equal variance → Independent t-test
  - Both normal + unequal variance → Welch's t-test  
  - Either/both non-normal → Mann-Whitney U

For paired data:
  - Normal differences → Paired t-test
  - Non-normal differences → Wilcoxon signed-rank

For 3+ groups:
  - All normal + equal variance → One-way ANOVA
  - Otherwise → Kruskal-Wallis

For correlation:
  - Both normal → Pearson
  - Either non-normal → Spearman
```

I went with conservative routing because I'd rather lose 5% statistical power than give someone invalid p-values.

## Project Structure
```
INCF-AStats-cli-prototype/
├── stats_engine/
│   ├── assumption_checker.py  # Checks normality, variance, recommends test
│   └── executor.py             # Runs the actual scipy tests
├── data_utils/
│   └── simulator.py            # Generates labeled test scenarios
├── examples/
│   └── real_data_example.py    # Iris dataset demo
├── tests/
│   ├── test_normality.py
│   ├── test_homogeneity.py
│   ├── test_decision_tree.py
│   ├── test_executor.py
│   ├── test_all_scenarios.py   # The eval harness
│   └── test_full_pipeline.py
└── study_notes/
    ├── effect_sizes.md
    └── synthetic_data.md
```

## Study Notes

While building this, I had to learn some statistical concepts properly:

- **Why Levene's test over Bartlett's?** Bartlett assumes normality, which defeats the purpose if you're checking variance *because* you suspect non-normality. Levene with median is more robust.

- **Why use Mann-Whitney if only ONE group is non-normal?** The t-test assumes normality in BOTH groups. If one fails, the test statistic doesn't follow a t-distribution anymore, so p-values become unreliable.

- **What's the point of effect sizes?** p-values tell you IF there's a difference, effect sizes tell you HOW BIG. With large sample sizes, tiny meaningless differences can be "significant" - effect sizes prevent that trap.

All my notes are in `study_notes/` if anyone's interested.

## Next Steps

Things I'm planning to add:

- [ ] Agents layer for plain-language interpretation
- [ ] CLI with `--human-in-loop` flag for when the LLM disagrees with the rule-based path
- [ ] More real dataset examples (neuroscience data would be cool)
- [ ] CI/CD pipeline
- [ ] Post-hoc tests (Tukey HSD for ANOVA, Dunn's test for Kruskal-Wallis)

## Running Tests
```bash
# All tests should pass
python tests/test_normality.py
python tests/test_homogeneity.py
python tests/test_executor.py
python tests/test_all_scenarios.py

# Real dataset example
python examples/real_data_example.py
```

## Why I Built This

The project description mentioned that practitioners often make assumption errors - I've definitely seen this in my data science courses. People run t-tests on skewed data, or ANOVA with unequal variance, and get misleading results. 

My approach is: deterministic decision tree for reliability (because stats assumptions are math, not opinions), but designed to integrate with an LLM layer for:
- Upstream: parsing natural language queries
- Downstream: domain-specific interpretation
- Validation: LLM can challenge the recommendation but doesn't override it

Basically: let the rule-based system handle the math, let the LLM handle the natural language.

---

Built by Roaa Raafat for GSoC 2026 - INCF Project #33  
Repo: https://github.com/Roaa-838/INCF-AStats-cli-prototype
