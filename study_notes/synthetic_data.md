# Synthetic Data Generation Strategy

## Goal
Create datasets where we KNOW the ground truth:
- Which test SHOULD be used
- Whether groups differ
- What the effect size should be approximately

## Design Principles

### 1. Normal Data
- Use np.random.normal(mean, std, size)
- Can verify with Shapiro-Wilk (should pass)

### 2. Non-Normal Data
- Exponential: np.random.exponential(scale, size)
  - Always right-skewed (skew > 1)
  - Shapiro-Wilk will reject
- Uniform: np.random.uniform(low, high, size)
  - Platykurtic (light tails)
  - For small n, may pass normality

### 3. Equal Variance
- Use same std for both groups
- np.random.normal(0, 1, n) vs np.random.normal(2, 1, n)

### 4. Unequal Variance
- Use different std
- np.random.normal(0, 1, n) vs np.random.normal(0, 5, n)
- Ratio of 1:5 is clearly unequal

## Scenario Design

### Scenario 1: Normal + Equal Var → independent_t
- Group 1: N(0, 1)
- Group 2: N(0.5, 1)  # Small effect, d ≈ 0.5
- Expected: Both pass normality, equal variance, use independent_t

### Scenario 2: Normal + Unequal Var → welch_t
- Group 1: N(0, 1)
- Group 2: N(0, 5)  # Same mean (H0 true), but SD 5x larger
- Expected: Both pass normality, unequal variance, use welch_t

### Scenario 3: Non-Normal → mann_whitney_u
- Group 1: Exp(1)
- Group 2: Exp(2)  # Different scale
- Expected: Both fail normality, use mann_whitney_u

### Scenario 4: Three Groups Normal → one_way_anova
- Group A: N(0, 1)
- Group B: N(0.5, 1)
- Group C: N(1, 1)
- Expected: All pass normality, equal variance, use one_way_anova

### Scenario 5: Three Groups Non-Normal → kruskal_wallis
- Group A: Exp(1)
- Group B: Exp(1.5)
- Group C: Exp(2)
- Expected: All fail normality, use kruskal_wallis

### Scenario 6: Correlation Normal → pearson_r
- X: N(0, 1)
- Y: 0.7*X + noise  # r ≈ 0.7
- Expected: Both normal, linear, use pearson_r

### Scenario 7: Correlation Non-Normal → spearman_r
- X: Exp(1)
- Y: log(X) + noise  # Monotonic but not linear
- Expected: At least one non-normal, use spearman_r