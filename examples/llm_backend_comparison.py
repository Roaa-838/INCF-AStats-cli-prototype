# examples/llm_backend_comparison.py
"""
Demonstrates the mentor's suggestion: comparing API-based LLMs
(Claude, GPT-4) with local open-weight models (Ollama) for
statistical methods paragraph generation.

The mentor noted these models "know quite a bit about statistics
and are able to work agentically quite well."
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from stats_engine.assumption_checker import build_data_profile
from stats_engine.executor import run_test
from stats_engine.llm_interpreter import generate_methods_paragraph, compare_backends

np.random.seed(42)

# Use sleepstudy-style repeated measures result
groups = {
    'Day_0': pd.Series(np.random.normal(250, 30, 18)),
    'Day_5': pd.Series(np.random.normal(280, 35, 18)),
    'Day_9': pd.Series(np.random.normal(320, 40, 18)),
}

profile  = build_data_profile(groups, design='repeated')
recommended = profile['recommendation']['recommended_test']

# Use friedman as fallback since repeated_anova isn't implemented yet
test_to_run = 'friedman' if recommended in ('repeated_anova', 'friedman') else recommended
result = run_test(test_to_run, groups)

print(f"Pipeline recommends: {recommended}")
print(f"Running: {test_to_run} (repeated_anova not yet implemented — using Friedman)")

print("=" * 65)
print("LLM BACKEND COMPARISON: Methods Paragraph Generation")
print("Scenario: Friedman test on repeated measures RT data")
print("=" * 65)
print(f"Friedman result: chi2={result['statistic']:.2f}, "
      f"p={result['p_value']:.4f}, W={result['effect_size']:.3f}")
print()

outputs = compare_backends(result, profile)

for backend, text in outputs.items():
    print(f"--- {backend.upper()} ---")
    print(text)
    print()

print("=" * 65)
print("Observation: API models produce more natural phrasing.")
print("Template fallback ensures the tool works without any API key.")
print("The auto backend selects the best available option.")