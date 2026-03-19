# Effect Sizes Study Notes

## Why Effect Sizes Matter
- p-value: "Is there a difference?" (statistical significance)
- Effect size: "How big is that difference?" (practical significance)
- With N=10,000, even tiny effects become "significant" but meaningless
- Effect sizes are comparable across studies

## Cohen's d (for t-tests)
Formula: d = (M1 - M2) / SD_pooled

**For equal variance (pooled SD):**
SD_pooled = sqrt(((n1-1)*SD1² + (n2-1)*SD2²) / (n1+n2-2))

**For unequal variance:**
SD_pooled = sqrt((SD1² + SD2²) / 2)

**Interpretation (Cohen, 1988):**
- Small effect: d = 0.2
- Medium effect: d = 0.5
- Large effect: d = 0.8

**Example:**
- M1 = 100, M2 = 115, SD = 15
- d = (100 - 115) / 15 = -1.0
- Interpretation: "1 standard deviation difference" = LARGE effect

## Rank-Biserial r (for Mann-Whitney U)
Formula: r = 1 - (2U)/(n1*n2)

**Why this formula?**
- U statistic ranges from 0 to n1*n2
- U = 0 means all group 1 values < all group 2 values
- U = n1*n2 means all group 1 values > all group 2 values
- r rescales this to -1 to +1 range

**Interpretation:**
- Small: |r| = 0.1
- Medium: |r| = 0.3
- Large: |r| = 0.5

## Eta-squared η² (for ANOVA)
Formula: η² = SS_between / SS_total

**From F-statistic:**
η² = (df_between * F) / (df_between * F + df_within)

**Interpretation:**
- Small: η² = 0.01 (1% variance explained)
- Medium: η² = 0.06 (6% variance explained)
- Large: η² = 0.14 (14% variance explained)

## Key Insight
**Power vs. Robustness Trade-off:**
- Mann-Whitney has ~95% power efficiency vs t-test when normality holds
- But t-test can have inflated Type I error when normality violated
- 5% power loss is acceptable price for robustness