# examples/agentic_llm_experiment.py
"""
Mentor suggested testing whether Claude/GPT-4 can work agentically
on statistical analysis tasks.

This experiment gives the same datasets to:
1. AStats deterministic pipeline
2. Claude acting as a statistical agent

Then compares:
- Correctness of test selection
- Correctness of assumption checking
- Quality of reasoning
- Whether the LLM commits pseudoreplication on the sleepstudy data
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import urllib.request
import numpy as np
import pandas as pd
from stats_engine.profiler import infer_structure
from stats_engine.assumption_checker import build_data_profile
from stats_engine.executor import run_test


def ask_llm_as_agent(dataset_description: str,
                     data_sample: str,
                     api_key: str,
                     model: str = "claude-haiku-4-5-20251001") -> str:
    """
    Ask Claude to act as a statistical analysis agent.
    Give it the same information AStats has and see what it decides.
    """
    prompt = f"""You are a statistical analysis expert. 
A researcher has given you a dataset and wants to know 
what statistical test to run.

Dataset description:
{dataset_description}

First 10 rows of data:
{data_sample}

Please:
1. Identify the data structure (independent groups or repeated measures?)
2. State what assumption checks you would run
3. Select the appropriate statistical test and explain why
4. Note any concerns about the data

Be specific and justify each decision."""

    payload = json.dumps({
        "model": model,
        "max_tokens": 600,
        "messages": [{"role": "user", "content": prompt}]
    }).encode('utf-8')

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            result = json.loads(r.read().decode('utf-8'))
            return result['content'][0]['text'].strip()
    except Exception as e:
        return f"API call failed: {e}"


def run_astats_pipeline(df: pd.DataFrame,
                        group_col: str,
                        value_col: str,
                        design: str = 'independent') -> dict:
    """Run the full AStats deterministic pipeline."""
    groups = {
        name: df[df[group_col] == name][value_col]
        for name in df[group_col].unique()
    }
    profile  = build_data_profile(groups, design=design)
    verdict  = infer_structure(df)
    recommended = profile['recommendation']['recommended_test']

    if not profile['guardrails']['blocked']:
        result = run_test(recommended, groups)
        return {
            'structure_detected': verdict.verdict,
            'test_selected': recommended,
            'rationale': profile['recommendation']['rationale'],
            'p_value': result.get('p_value'),
            'effect_size': result.get('effect_size'),
            'effect_type': result.get('effect_type')
        }
    return {
        'structure_detected': verdict.verdict,
        'test_selected': 'BLOCKED',
        'rationale': str(profile['guardrails']['issues'])
    }


# ── Test Case 1: The Pseudoreplication Trap ───────────────────
print("=" * 65)
print("TEST 1: Sleepstudy — Does LLM commit pseudoreplication?")
print("=" * 65)

import urllib.request as ur
import io

url = ("https://raw.githubusercontent.com/vincentarelbundock/"
       "Rdatasets/master/csv/lme4/sleepstudy.csv")
with ur.urlopen(url) as r:
    df_sleep = pd.read_csv(io.StringIO(r.read().decode('utf-8')))
if df_sleep.columns[0] not in {'Reaction', 'Days', 'Subject'}:
    df_sleep = df_sleep.iloc[:, 1:]
df_sleep['Days'] = df_sleep['Days'].astype(str)

# AStats answer
astats_result = run_astats_pipeline(df_sleep, 'Days', 'Reaction')
print(f"\nAStats:")
print(f"  Structure detected: {astats_result['structure_detected']}")
print(f"  Test selected:      {astats_result['test_selected']}")
print(f"  Rationale:          {astats_result['rationale']}")

# LLM answer (only runs if API key set)
api_key = os.environ.get('ANTHROPIC_API_KEY', '')
if api_key:
    sample = df_sleep.head(15).to_string(index=False)
    description = """
Dataset: sleepstudy
Rows: 180
Columns: Reaction (milliseconds), Days (0-9), Subject (integer ID)
Research question: Does reaction time change across days of sleep deprivation?
"""
    llm_answer = ask_llm_as_agent(description, sample, api_key)
    print(f"\nClaude (agentic):")
    print(llm_answer)

    print(f"\nKey question: Did Claude detect the repeated measures structure?")
    print(f"Did it avoid treating 180 rows as 180 independent observations?")
else:
    print("\n[Set ANTHROPIC_API_KEY to run LLM comparison]")


# ── Test Case 2: Non-Normal Data ──────────────────────────────
print("\n" + "=" * 65)
print("TEST 2: Log-normal RT data — Does LLM suggest correct test?")
print("=" * 65)

np.random.seed(42)
n = 40
df_rt = pd.DataFrame({
    'condition': ['baseline']*n + ['treatment']*n,
    'rt_ms': np.concatenate([
        np.random.lognormal(np.log(300), 0.3, n),
        np.random.lognormal(np.log(250), 0.3, n)
    ])
})

astats_rt = run_astats_pipeline(df_rt, 'condition', 'rt_ms')
print(f"\nAStats:")
print(f"  Test selected: {astats_rt['test_selected']}")
print(f"  Rationale:     {astats_rt['rationale']}")

if api_key:
    sample_rt = df_rt.head(10).to_string(index=False)
    desc_rt = """
Dataset: Reaction time study
Rows: 80 (40 per condition)
Columns: condition (baseline/treatment), rt_ms (reaction time in milliseconds)
Research question: Is reaction time different between baseline and treatment?
Note: reaction time data often follows a log-normal distribution.
"""
    llm_rt = ask_llm_as_agent(desc_rt, sample_rt, api_key)
    print(f"\nClaude (agentic):")
    print(llm_rt)

print("\n" + "=" * 65)
print("INTERPRETATION")
print("=" * 65)
print("""
This experiment addresses the mentor's suggestion to explore whether
API-based LLMs 'know quite a bit about statistics and are able to 
work agentically quite well.'

Findings this experiment helps answer:
- Does Claude correctly identify repeated measures structure?
- Does Claude recommend non-parametric tests for non-normal data?
- Does Claude's reasoning match AStats' deterministic logic?
- Where does the LLM approach fail that the deterministic approach catches?

These findings inform the hybrid design: deterministic guarantees for
correctness, LLM for natural language and interpretation.
""")