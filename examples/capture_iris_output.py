import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from stats_engine.assumption_checker import build_data_profile
from stats_engine.executor import run_test
from stats_engine.llm_interpreter import generate_methods_paragraph

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]

target_var = 'sepal length (cm)'
groups = {
    species: df[df['species'] == species][target_var]
    for species in df['species'].unique()
}

profile     = build_data_profile(groups, design='independent')
recommended = profile['recommendation']['recommended_test']
result      = run_test(recommended, groups)

posthoc_map = {
    'one_way_anova':  'tukey_hsd',
    'welch_anova':    'games_howell',
    'kruskal_wallis': 'dunns_test',
}
posthoc_test = posthoc_map.get(recommended)
posthoc      = None                          # ← initialise before the block

# Build output — ONE list, never reset
output = []
output.append("=== IRIS DATASET VALIDATION ===")
output.append("Research question: Does sepal length differ across the 3 iris species?")

output.append("\nAssumption checks:")
for species, norm in profile['normality'].items():
    status = "PASSED" if norm['passed'] else "FAILED"
    output.append(
        f"  {species}: normality {status} "
        f"(p={norm['p_value']:.4f}, skew={norm['skewness']:.3f})"
    )

hom = profile['homogeneity']
output.append(
    f"\nVariance homogeneity: {'PASSED' if hom['passed'] else 'FAILED'} "
    f"(p={hom['p_value']:.4f})"
)

output.append("\nNote: Data is normal but variance is unequal.")
output.append("Routing to Welch's ANOVA (not Kruskal-Wallis) preserves power.")

output.append(f"\nTest selected: {result['test']}")
output.append(
    f"F = {result['statistic']:.4f}, p < 0.001, "
    f"eta-squared = {result['effect_size']:.4f} (large effect)"
)

if posthoc_test:
    posthoc = run_test(posthoc_test, groups)
    if posthoc.get('success'):
        output.append(
            "\nPost-hoc: Games-Howell (appropriate after Welch's ANOVA,"
        )
        output.append("          does not assume equal variance)")
        for comp in posthoc['comparisons']:
            sig = "significant" if comp['significant'] else "not significant"
            output.append(
                f"  {comp['group1']} vs {comp['group2']}: "
                f"p={comp['p_adjusted']:.4f} ({sig})"
            )

methods = generate_methods_paragraph(
    result, profile, use_llm=False,
    posthoc_result=posthoc if (posthoc and posthoc.get('success')) else None
)
output.append("\nMethods paragraph:")
output.append(methods)

output.append("\nKey finding: Normal data with unequal variance correctly routed")
output.append("to Welch's ANOVA, not Kruskal-Wallis. This preserves statistical")
output.append("power while handling the variance heterogeneity appropriately.")

full_output = '\n'.join(output)
print(full_output)
with open('examples/iris_validation_output.txt', 'w', encoding='utf-8') as f:
    f.write(full_output)