# stats_engine/llm_interpreter.py
"""
LLM Interpretation Layer.

Supports three backends (in order of preference):
1. Anthropic Claude API  — best statistical reasoning, low cost
2. OpenAI GPT-4 API     — strong statistical reasoning, widely available
3. Ollama (local)        — free, works offline, no API key needed
4. Template fallback     — always works, no LLM required

The mentor specifically suggested Codex/Claude as inexpensive options
that work well agentically for statistical tasks.
"""
from __future__ import annotations
import json
import os
import urllib.request
from typing import Optional


TEST_DISPLAY_NAMES = {
    'independent_t':        "an independent samples t-test",
    'welch_t':              "Welch's t-test",
    'mann_whitney_u':       "a Mann-Whitney U test",
    'paired_t':             "a paired samples t-test",
    'wilcoxon_signed_rank': "a Wilcoxon signed-rank test",
    'one_way_anova':        "a one-way ANOVA",
    'welch_anova':          "Welch's one-way ANOVA",
    'kruskal_wallis':       "a Kruskal-Wallis test",
    'friedman':             "a Friedman test",
    'pearson_r':            "Pearson's correlation",
    'spearman_r':           "Spearman's rank correlation",
    'tukey_hsd':            "Tukey's HSD post-hoc test",
    'games_howell':         "Games-Howell post-hoc test",
    'dunns_test':           "Dunn's test with Holm-Bonferroni correction",
    'pairwise_wilcoxon':    "pairwise Wilcoxon signed-rank tests with Holm correction",
}

def _get_test_display_name(test_name: str) -> str:
    return TEST_DISPLAY_NAMES.get(test_name, 
                                   f"a {test_name.replace('_', ' ')} test")

def _effect_magnitude(effect_size: float, effect_type: str) -> str:
    if effect_type == 'cohen_d':
        thresholds = [(0.8, 'large'), (0.5, 'medium'), (0.2, 'small')]
    elif effect_type in ('eta_squared', 'epsilon_squared'):
        thresholds = [(0.14, 'large'), (0.06, 'medium'), (0.01, 'small')]
    elif effect_type == 'kendalls_w':
        thresholds = [(0.5, 'large'), (0.3, 'medium'), (0.1, 'small')]
    else:
        thresholds = [(0.5, 'large'), (0.3, 'medium'), (0.1, 'small')]
    for threshold, label in thresholds:
        if abs(effect_size) >= threshold:
            return label
    return 'negligible'

def _build_prompt(test_result: dict, profile: dict) -> str:
    rec         = profile.get('recommendation', {})
    test_name   = test_result.get('test', 'unknown')
    statistic   = test_result.get('statistic', 0) or 0
    p_value     = test_result.get('p_value', 1) or 1
    effect_size = test_result.get('effect_size', 0) or 0
    effect_type = test_result.get('effect_type', 'effect size') or 'effect size'

    return f"""You are a scientific writing assistant for neuroscience research.
Convert this statistical result into ONE publication-ready methods sentence.
Use APA 7th edition format. Be precise and concise.

Test: {test_name}
Statistic: {statistic:.3f}
p-value: {p_value:.4f}
Effect size: {effect_type} = {effect_size:.3f}
Rationale for test selection: {rec.get('rationale', '')}
Number of groups: {profile.get('n_groups', 2)}

Output ONE sentence only. No preamble, no explanation, no markdown."""

# ── Backend 1: Anthropic Claude API ──────────────────────────

def _call_claude(prompt: str, model: str = "claude-haiku-4-5-20251001") -> Optional[str]:

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return None

    payload = json.dumps({
        "model": model,
        "max_tokens": 200,
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
    except Exception:
        return None


# ── Backend 2: OpenAI GPT-4 API ──────────────────────────────

def _call_openai(prompt: str, model: str = "gpt-4o-mini") -> Optional[str]:

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return None

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0
    }).encode('utf-8')

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            result = json.loads(r.read().decode('utf-8'))
            return result['choices'][0]['message']['content'].strip()
    except Exception:
        return None


# ── Backend 3: Ollama (local open-weight) ────────────────────

def _ollama_available(base_url: str = "http://localhost:11434") -> bool:
    try:
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.status < 400
    except Exception:
        return False


def _call_ollama(prompt: str, model: str = "qwen2.5",
                  base_url: str = "http://localhost:11434") -> Optional[str]:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0}
    }).encode('utf-8')

    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            result = json.loads(r.read().decode('utf-8'))
            return str(result.get('response', '')).strip()
    except Exception:
        return None


# ── Template fallback ─────────────────────────────────────────
def _template_fallback(test_result: dict, profile: dict,
                        posthoc_result: dict = None) -> str:
    test_name   = test_result.get('test', 'unknown')
    statistic   = test_result.get('statistic', 0.0)
    p_value     = test_result.get('p_value', 1.0)
    effect_size = test_result.get('effect_size', 0.0)
    effect_type = test_result.get('effect_type', 'effect size')
    n_groups    = profile.get('n_groups', 2)

    test_display = _get_test_display_name(test_name)
    p_str      = "< 0.001" if p_value < 0.001 else f"= {p_value:.3f}"
    conclusion = ("a statistically significant difference was detected"
                  if p_value < 0.05
                  else "no statistically significant difference was detected")
    magnitude  = _effect_magnitude(effect_size, effect_type)
    effect_display = {
        'cohen_d':         "Cohen's d",
        'eta_squared':     "eta-squared",
        'epsilon_squared': "epsilon-squared",
        'kendalls_w':      "Kendall's W",
        'rank_biserial_r': "rank-biserial r",
        'r_squared':       "r-squared",
        'rho_squared':     "rho-squared"
    }.get(effect_type, effect_type)

    base = (
        f"Data from {n_groups} groups were analyzed using "
        f"{test_display}. "
        f"Results indicated {conclusion} "
        f"(statistic = {statistic:.3f}, p {p_str}, "
        f"{effect_display} = {effect_size:.3f}, {magnitude} effect)."
    )

    # Append post-hoc summary if provided and significant
    if posthoc_result and posthoc_result.get('success'):
        sig_pairs = posthoc_result.get('significant_pairs', [])
        posthoc_name = _get_test_display_name(posthoc_result.get('test', ''))
        if sig_pairs:
            base += (
                f" {posthoc_name} identified {len(sig_pairs)} significant "
                f"pairwise difference(s): {', '.join(sig_pairs)}."
            )

    base += "\n[Generated by template fallback — install Ollama or set an API key for enhanced output]"
    return base

# ── Main entry point ──────────────────────────────────────────

def generate_methods_paragraph(
    test_result: dict,
    profile: dict,
    use_llm: bool = True,
    backend: str = "auto",
    model: Optional[str] = None,
    posthoc_result: dict = None 
) -> str:

    if not use_llm:
        return _template_fallback(test_result, profile, posthoc_result)

    prompt = _build_prompt(test_result, profile)
    output = None
    backend_used = "template"

    if backend == "auto":
        # Try Claude first
        output = _call_claude(prompt, model or "claude-haiku-4-5-20251001")
        if output:
            backend_used = "claude-haiku"
        else:
            # Try OpenAI
            output = _call_openai(prompt, model or "gpt-4o-mini")
            if output:
                backend_used = "gpt-4o-mini"
            else:
                # Try Ollama
                if _ollama_available():
                    output = _call_ollama(prompt, model or "qwen2.5")
                    if output:
                        backend_used = f"ollama/{model or 'qwen2.5'}"

    elif backend == "claude":
        output = _call_claude(prompt, model or "claude-haiku-4-5-20251001")
        backend_used = "claude"

    elif backend == "openai":
        output = _call_openai(prompt, model or "gpt-4o-mini")
        backend_used = "openai"

    elif backend == "ollama":
        if _ollama_available():
            output = _call_ollama(prompt, model or "qwen2.5")
            backend_used = "ollama"

    if output and len(output) > 20:
        return f"{output}\n[Generated by {backend_used}]"

    return _template_fallback(test_result, profile, posthoc_result)


def compare_backends(test_result: dict, profile: dict) -> dict:

    prompt = _build_prompt(test_result, profile)
    results = {}

    claude_out = _call_claude(prompt)
    results['claude'] = claude_out if claude_out else "API key not set or unavailable"

    openai_out = _call_openai(prompt)
    results['openai'] = openai_out if openai_out else "API key not set or unavailable"

    if _ollama_available():
        ollama_out = _call_ollama(prompt)
        results['ollama'] = ollama_out if ollama_out else "Ollama call failed"
    else:
        results['ollama'] = "Ollama not running"

    results['template'] = _template_fallback(test_result, profile)

    return results