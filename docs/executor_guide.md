# Executor Module Guide

## Overview
The executor module implements 9 statistical tests with proper effect sizes.

## Test Selection Quick Reference

| Scenario | Data Type | Test |
|----------|-----------|------|
| 2 independent groups, normal, equal var | Continuous | independent_t |
| 2 independent groups, normal, unequal var | Continuous | welch_t |
| 2 independent groups, non-normal | Continuous/Ordinal | mann_whitney_u |
| 2 paired groups, normal differences | Continuous | paired_t |
| 2 paired groups, non-normal differences | Continuous/Ordinal | wilcoxon_signed_rank |
| 3+ groups, all normal, equal var | Continuous | one_way_anova |
| 3+ groups, non-normal or unequal var | Continuous/Ordinal | kruskal_wallis |
| 2 variables, both normal, linear | Continuous | pearson_r |
| 2 variables, non-normal or monotonic | Continuous/Ordinal | spearman_r |

## Effect Size Interpretation

### Cohen's d (t-tests, ANOVA)
- Small: 0.2
- Medium: 0.5
- Large: 0.8

### Rank-biserial r (Mann-Whitney, Wilcoxon)
- Small: 0.1
- Medium: 0.3
- Large: 0.5

### Eta-squared / Epsilon-squared (ANOVA, Kruskal)
- Small: 0.01 (1% variance explained)
- Medium: 0.06 (6% variance explained)
- Large: 0.14 (14% variance explained)

### Correlation r² / ρ²
- Small: 0.01 (1% variance explained)
- Medium: 0.09 (9% variance explained)
- Large: 0.25 (25% variance explained)

## Common Questions

### "When should I use Welch's t-test instead of Student's t-test?"
Use Welch when:
- Levene's test shows p < 0.05 (unequal variance)
- Sample sizes differ greatly (n1/n2 > 2 or < 0.5)
- Standard deviations differ greatly (SD1/SD2 > 2 or < 0.5)

### "Why use Mann-Whitney even if only ONE group is non-normal?"
Parametric tests (t-test) assume normality in **BOTH** groups. If even one group violates this, the test statistic no longer follows a t-distribution, making p-values unreliable.

### "What's the difference between paired t-test and independent t-test?"
Paired t-test:
- Same subjects measured twice (before/after)
- Tests if **mean difference** ≠ 0
- More powerful (controls for individual differences)
- Requires same n in both groups

Independent t-test:
- Different subjects in each group
- Tests if **group means** differ
- Less powerful (more between-subject variability)
- Can have different n

### "When should I use ANOVA vs multiple t-tests?"
ALWAYS use ANOVA for 3+ groups, never multiple t-tests.

Why?
- Multiple t-tests inflate Type I error (false positives)
- With 3 groups: 3 t-tests = 14% chance of false positive (not 5%)
- With 4 groups: 6 t-tests = 26% chance of false positive!
- ANOVA controls this at 5% for the entire family of comparisons