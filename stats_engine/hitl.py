
from __future__ import annotations
from typing import Optional
import pandas as pd


class HITLCheckpoint:
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.override_log = []   # track all human overrides for audit
    
    def review(
        self,
        profile: dict,
        groups: dict,
        verbose: bool = True
    ) -> dict:

        recommendation = profile.get('recommendation', {})
        recommended_test = recommendation.get('recommended_test')
        rationale = recommendation.get('rationale', '')
        warnings = recommendation.get('warnings', [])
        guardrails = profile.get('guardrails', {})
        
        # If guardrails blocked analysis, no HITL — show blocking reason
        if guardrails.get('blocked'):
            print("\n[AStats] Analysis blocked — data quality issues:")
            for issue in guardrails.get('issues', []):
                print(f"  ✗ {issue}")
            return {
                'test': None,
                'source': 'blocked',
                'reasoning': guardrails.get('issues', []),
                'user_confirmed': False
            }
        
        if not self.enabled:
            # Automated mode: run pipeline recommendation silently
            return {
                'test': recommended_test,
                'source': 'pipeline',
                'reasoning': [rationale],
                'user_confirmed': True   # implicitly approved
            }
        
        print("\n" + "="*60)
        print("[AStats] Pipeline Recommendation")
        print("="*60)
        print(f"Recommended test : {recommended_test}")
        print(f"Rationale        : {rationale}")
        
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f" {w}")
        
        # Show group summary
        print(f"\nData summary:")
        for name, series in groups.items():
            clean = series.dropna()
            print(f"  {name}: n={len(clean)}, "
                  f"mean={clean.mean():.2f}, "
                  f"median={clean.median():.2f}")
        
        print("\nOptions:")
        print("  [Enter]        Accept this recommendation")
        print("  [o] Override   Choose a different test")
        print("  [w] Why        Explain the decision in detail")
        print("  [s] Skip       Skip HITL for this analysis")
        print("-"*60)
        
        while True:
            try:
                choice = input("Your choice: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                # Non-interactive environment: auto-accept
                print("  [Non-interactive mode: auto-accepting recommendation]")
                choice = ''
            
            if choice == '' or choice == 'a':
                print(f"  Accepted: {recommended_test}")
                return {
                    'test': recommended_test,
                    'source': 'pipeline',
                    'reasoning': [rationale],
                    'user_confirmed': True
                }
            
            elif choice == 'o':
                return self._handle_override(recommended_test, rationale)
            
            elif choice == 'w':
                self._explain_reasoning(profile)
                # Loop back to show options again
            
            elif choice == 's':
                self.enabled = False
                print("  HITL disabled for this session.")
                return {
                    'test': recommended_test,
                    'source': 'pipeline',
                    'reasoning': [rationale],
                    'user_confirmed': True
                }
            
            else:
                print("  Please enter one of: [Enter], o, w, s")
    
    def _handle_override(
        self,
        recommended: str,
        rationale: str
    ) -> dict:
        """Let user specify a different test with a reason."""
        available_tests = [
            'independent_t', 'welch_t', 'mann_whitney_u',
            'paired_t', 'wilcoxon_signed_rank',
            'one_way_anova', 'welch_anova', 'kruskal_wallis',
            'friedman', 'pearson_r', 'spearman_r'
        ]
        
        print("\nAvailable tests:")
        for i, t in enumerate(available_tests, 1):
            marker = " ← (pipeline choice)" if t == recommended else ""
            print(f"  {i:2}. {t}{marker}")
        
        while True:
            try:
                choice = input("\nEnter test name or number: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("  Keeping pipeline recommendation.")
                return {
                    'test': recommended,
                    'source': 'pipeline',
                    'reasoning': [rationale],
                    'user_confirmed': True
                }
            
            # Accept number
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(available_tests):
                    chosen = available_tests[idx]
                else:
                    print(f"  Please enter 1-{len(available_tests)}")
                    continue
            elif choice in available_tests:
                chosen = choice
            else:
                print(f"  Unknown test '{choice}'. Use a name from the list.")
                continue
            
            try:
                reason = input(
                    "Brief reason for override (helps audit trail): "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                reason = "user override (no reason given)"
            
            # Log the override
            self.override_log.append({
                'pipeline_recommendation': recommended,
                'user_choice': chosen,
                'reason': reason
            })
            
            print(f"\n  Override accepted: {chosen}")
            if chosen != recommended:
                print(f"  Note: pipeline recommended '{recommended}' based on:")
                print(f"    {rationale}")
                print(f"    Your override will run '{chosen}' instead.")
            
            return {
                'test': chosen,
                'source': 'user_override',
                'reasoning': [f"User override: {reason}"],
                'pipeline_recommendation': recommended,
                'user_confirmed': True
            }
    
    def _explain_reasoning(self, profile: dict) -> None:
        """Detailed explanation of why the pipeline chose this test."""
        rec = profile.get('recommendation', {})
        
        print("\n[Why this test?]")
        print(f"  Test: {rec.get('recommended_test')}")
        print(f"  Reason: {rec.get('rationale')}")
        
        print("\n[Normality results:]")
        for group, result in profile.get('normality', {}).items():
            status = "PASSED ✓" if result.get('passed') else "FAILED ✗"
            print(f"  {group}: {status} (p={result.get('p_value', 'N/A')})")
        
        homogeneity = profile.get('homogeneity', {})
        if homogeneity:
            status = "PASSED " if homogeneity.get('passed') else "FAILED ✗"
            print(f"\n[Variance homogeneity]: {status} "
                  f"(p={homogeneity.get('p_value', 'N/A')})")
        
        print("\n[Conservative routing policy]:")
        print("  If ANY group fails normality → non-parametric test.")
        print("  This trades ~5% statistical power for valid p-values.")
        print("  You can override this if you have domain reasons to proceed.")
        print()
    
    def get_audit_log(self) -> list:
        return self.override_log