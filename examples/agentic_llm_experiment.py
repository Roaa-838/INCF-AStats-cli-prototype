# examples/agentic_llm_experiment.py
"""
Mentor suggested testing whether LLMs can work agentically
on statistical analysis tasks.

This experiment gives the same datasets to:
1. AStats deterministic pipeline
2. An LLM acting as a statistical agent (Groq/Gemini/Claude)

Then compares:
- Correctness of test selection
- Correctness of assumption checking
- Quality of reasoning
- Whether the LLM commits pseudoreplication on the sleepstudy data
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

import json
import io
import urllib.request
import numpy as np
import pandas as pd
from stats_engine.profiler import infer_structure
from stats_engine.assumption_checker import build_data_profile
from stats_engine.executor import run_test


def ask_llm_as_agent(dataset_description: str,
                     data_sample: str) -> tuple[str, str]:
    """
    Ask an available LLM to act as a statistical analysis agent.
    Tries Groq first (free), then Gemini (free), then Claude (paid).
    Returns (response_text, backend_name).
    """
    prompt = f"""You are a statistical analysis expert.
A researcher has given you a dataset and wants to know
what statistical test to run.

Dataset description:
{dataset_description}

First 15 rows of data:
{data_sample}

Please:
1. Identify the data structure (independent groups or repeated measures?)
2. State what assumption checks you would run
3. Select the appropriate statistical test and explain why
4. Note any concerns about the data

Be specific and justify each decision."""

    # Try Groq first (free)
    groq_key = os.environ.get('GROQ_API_KEY')
    if groq_key:
        payload = json.dumps({
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 600,
            "temperature": 0
        }).encode('utf-8')
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {groq_key}"
            },
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                result = json.loads(r.read().decode('utf-8'))
                text = result['choices'][0]['message']['content'].strip()
                return text, "Groq/Llama-3.3-70b"
        except Exception as e:
            print(f"Groq failed: {e}")

    # Try Gemini (free)
    gemini_key = os.environ.get('GEMINI_API_KEY')
    if gemini_key:
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}]
        }).encode('utf-8')
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"gemini-2.0-flash:generateContent?key={gemini_key}")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                result = json.loads(r.read().decode('utf-8'))
                text = (result['candidates'][0]['content']
                              ['parts'][0]['text'].strip())
                return text, "Gemini-1.5-Flash"
        except Exception as e:
            print(f"Gemini failed: {e}")

    # Try Claude (paid — kept for GSoC)
    claude_key = os.environ.get('ANTHROPIC_API_KEY')
    if claude_key:
        payload = json.dumps({
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 600,
            "messages": [{"role": "user", "content": prompt}]
        }).encode('utf-8')
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": claude_key,
                "anthropic-version": "2023-06-01"
            },
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                result = json.loads(r.read().decode('utf-8'))
                text = result['content'][0]['text'].strip()
                return text, "Claude-Haiku"
        except Exception as e:
            print(f"Claude failed: {e}")

    return ("No API key available. Set GROQ_API_KEY or GEMINI_API_KEY "
            "in your .env file."), "none"


def run_astats_pipeline(df: pd.DataFrame,
                        group_col: str,
                        value_col: str) -> dict:
    """Run the full AStats deterministic pipeline with auto structure detection."""
    groups = {
        name: df[df[group_col] == name][value_col]
        for name in df[group_col].unique()
    }

    # Use profiler to detect design first
    verdict = infer_structure(df)
    detected_design = (
        'repeated' if verdict.verdict == 'repeated_measures'
        else 'independent'
    )

    # Pass detected design to profile
    profile     = build_data_profile(groups, design=detected_design)
    recommended = profile['recommendation']['recommended_test']

    if not profile['guardrails']['blocked']:
        result = run_test(recommended, groups)
        return {
            'structure_detected': verdict.verdict,
            'design_used':        detected_design,
            'test_selected':      recommended,
            'rationale':          profile['recommendation']['rationale'],
            'p_value':            result.get('p_value'),
            'effect_size':        result.get('effect_size'),
            'effect_type':        result.get('effect_type')
        }
    return {
        'structure_detected': verdict.verdict,
        'test_selected':      'BLOCKED',
        'rationale':          str(profile['guardrails']['issues'])
    }


# ── Test Case 1: The Pseudoreplication Trap ───────────────────
print("=" * 65)
print("TEST 1: Sleepstudy — Does LLM commit pseudoreplication?")
print("=" * 65)

url = ("https://raw.githubusercontent.com/vincentarelbundock/"
       "Rdatasets/master/csv/lme4/sleepstudy.csv")
with urllib.request.urlopen(url) as r:
    df_sleep = pd.read_csv(io.StringIO(r.read().decode('utf-8')))
if df_sleep.columns[0] not in {'Reaction', 'Days', 'Subject'}:
    df_sleep = df_sleep.iloc[:, 1:]
df_sleep['Days'] = df_sleep['Days'].astype(str)

# AStats answer
astats_result = run_astats_pipeline(df_sleep, 'Days', 'Reaction')
print(f"\nAStats:")
print(f"  Structure detected: {astats_result['structure_detected']}")
print(f"  Design used:        {astats_result.get('design_used', 'N/A')}")
print(f"  Test selected:      {astats_result['test_selected']}")
print(f"  Rationale:          {astats_result['rationale']}")

# LLM answer
sample = df_sleep.head(15).to_string(index=False)
description = """
Dataset: sleepstudy
Rows: 180
Columns: Reaction (milliseconds), Days (0-9), Subject (integer ID)
Research question: Does reaction time change across days of sleep deprivation?
"""
llm_answer, backend_name = ask_llm_as_agent(description, sample)
print(f"\nLLM agent ({backend_name}):")
print(llm_answer)
print(f"\nKey question: Did the LLM detect repeated measures structure?")
print(f"Did it avoid treating 180 rows as 180 independent observations?")


# ── Test Case 2: Non-Normal Data ──────────────────────────────
print("\n" + "=" * 65)
print("TEST 2: Log-normal RT data — Does LLM suggest correct test?")
print("=" * 65)

np.random.seed(42)
n = 40
df_rt = pd.DataFrame({
    'condition': ['baseline'] * n + ['treatment'] * n,
    'rt_ms': np.concatenate([
        np.random.lognormal(np.log(300), 0.3, n),
        np.random.lognormal(np.log(250), 0.3, n)
    ])
})

astats_rt = run_astats_pipeline(df_rt, 'condition', 'rt_ms')
print(f"\nAStats:")
print(f"  Test selected: {astats_rt['test_selected']}")
print(f"  Rationale:     {astats_rt['rationale']}")

sample_rt = df_rt.head(10).to_string(index=False)
desc_rt = """
Dataset: Reaction time study
Rows: 80 (40 per condition)
Columns: condition (baseline/treatment), rt_ms (reaction time in milliseconds)
Research question: Is reaction time different between baseline and treatment?
Note: reaction time data often follows a log-normal distribution.
"""
llm_rt, backend_rt = ask_llm_as_agent(desc_rt, sample_rt)
print(f"\nLLM agent ({backend_rt}):")
print(llm_rt)

print("\n" + "=" * 65)
print("INTERPRETATION")
print("=" * 65)
print("""
This experiment addresses the mentor's suggestion to explore whether
API-based LLMs 'know quite a bit about statistics and are able to
work agentically quite well.'

Findings this experiment helps answer:
- Does the LLM correctly identify repeated measures structure?
- Does the LLM recommend non-parametric tests for non-normal data?
- Does the LLM reasoning match AStats deterministic logic?
- Where does the LLM approach fail that the deterministic approach catches?

These findings inform the hybrid design: deterministic guarantees for
correctness, LLM for natural language and interpretation.
""")