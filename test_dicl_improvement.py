#!/usr/bin/env python3

import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent))

from pr_agent.evaluation import (
    KubernetesTestCases,
    MetricsCalculator,
    ErrorDetectionParser,
)

class DICLImprovementTester:
    def __init__(self):
        self.significant_threshold = 0.05

    async def test_dicl_improvement(self, test_subset_size: int = 15) -> Dict[str, Any]:
        test_cases = self._get_diverse_test_cases(test_subset_size)

        baseline_results = await self._evaluate_without_context(test_cases)
        baseline_report = MetricsCalculator.aggregate_results(baseline_results)
        baseline_report.model_name = "GPT-4 (No Context)"

        enhanced_results = await self._evaluate_with_dicl_context(test_cases)
        enhanced_report = MetricsCalculator.aggregate_results(enhanced_results)
        enhanced_report.model_name = "GPT-4 (With DICL)"

        f1_improvement = enhanced_report.overall_f1_score - baseline_report.overall_f1_score
        precision_improvement = enhanced_report.overall_precision - baseline_report.overall_precision
        significant = f1_improvement >= self.significant_threshold
        improvement_percentage = (
            (f1_improvement / baseline_report.overall_f1_score * 100)
            if baseline_report.overall_f1_score > 0 else 0
        )
        conclusion = self._generate_conclusion(f1_improvement, self.significant_threshold)

        return {
            "test_cases": test_subset_size,
            "f1_improvement": f1_improvement,
            "precision_improvement": precision_improvement,
            "conclusion": conclusion,
        }

    def _get_diverse_test_cases(self, count: int) -> List:
        all_cases = KubernetesTestCases.get_all_test_cases()
        cases_by_category = {}
        for case in all_cases:
            for error in case.known_errors:
                category = error.category.value
                if category not in cases_by_category:
                    cases_by_category[category] = []
                if case not in cases_by_category[category]:
                    cases_by_category[category].append(case)
        selected_cases = []
        categories = list(cases_by_category.keys())
        cases_per_category = max(1, count // len(categories))
        for category in categories:
            category_cases = cases_by_category[category][:cases_per_category]
            selected_cases.extend(category_cases)
            if len(selected_cases) >= count:
                break
        return selected_cases[:count]

    async def _evaluate_without_context(self, test_cases: List) -> List:
        results = []
        for i, test_case in enumerate(test_cases, 1):
            print(f"  [{i}/{len(test_cases)}] Testing {test_case.id} (no context)")
            review_output = await self._get_basic_gpt4_review(test_case)
            detected_errors = ErrorDetectionParser.parse_review_output(
                review_output, "GPT-4 (No Context)"
            )
            result = MetricsCalculator.evaluate_test_case(
                test_case, detected_errors, "GPT-4 (No Context)"
            )
            results.append(result)
            print(
                f"TP: {len(result.true_positives)}, "
                f"FP: {len(result.false_positives)}, "
                f"FN: {len(result.false_negatives)}"
            )
        return results

    async def _evaluate_with_dicl_context(self, test_cases: List) -> List:
        results = []
        for i, test_case in enumerate(test_cases, 1):
            print(f"  [{i}/{len(test_cases)}] Testing {test_case.id} (with DICL)")
            review_output = await self._get_dicl_enhanced_review(test_case)
            detected_errors = ErrorDetectionParser.parse_review_output(
                review_output, "GPT-4 (With DICL)"
            )
            result = MetricsCalculator.evaluate_test_case(
                test_case, detected_errors, "GPT-4 (With DICL)"
            )
            results.append(result)
            print(
                f"    TP: {len(result.true_positives)}, "
                f"FP: {len(result.false_positives)}, "
                f"FN: {len(result.false_negatives)}"
            )
        return results

    async def _get_basic_gpt4_review(self, test_case) -> str:
        from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler
        basic_prompt = f"""URGENT: Analyze this Kubernetes configuration for CRITICAL SECURITY and RELIABILITY issues.
```yaml
{test_case.code_content}
```

🎯 MANDATORY CHECKS - Find ALL instances of:
1. **SECURITY VIOLATIONS:**
   - privileged: true (allows container escape)
   - runAsUser: 0 (root access)
   - hardcoded secrets/passwords in env vars
   - missing securityContext
   - wildcard RBAC permissions (*)

2. **RELIABILITY ISSUES:**
   - missing resource limits (memory/cpu)
   - port mismatches between service/container
   - maxUnavailable: 100% (total downtime risk)
   - missing readiness/liveness probes

3. **CONFIGURATION ERRORS:**
   - invalid field names or values
   - missing required fields

⚠️ CRITICAL: Specify EXACT line number for each issue found.
Format: "Line X: [SEVERITY] Issue description"
Example: "Line 15: CRITICAL - privileged: true allows container escape to host system"""
        ai_handler = LiteLLMAIHandler()
        response, _ = await ai_handler.chat_completion(
            model="gpt-4",
            temperature=0.1,
            system="You are a CRITICAL SECURITY AUDITOR for Kubernetes. Your job is to find EVERY security violation and reliability issue with EXACT line numbers.",
            user=basic_prompt,
        )
        return response

    async def _get_dicl_enhanced_review(self, test_case) -> str:
        from pr_agent.dicl.sdk import DICL
        pr_data = {
            "title": test_case.title,
            "description": test_case.description,
            "changed_files": [test_case.file_path],
            "language": test_case.language,
            "pr_id": f"eval_{test_case.id}",
            "diff": test_case.code_content,
        }
        base_prompt = f"""URGENT: Analyze this Kubernetes configuration for CRITICAL SECURITY and RELIABILITY issues.
```yaml
{test_case.code_content}
```

🎯 MANDATORY CHECKS - Find ALL instances of:
1. **SECURITY VIOLATIONS:**
   - privileged: true (allows container escape)
   - runAsUser: 0 (root access)
   - hardcoded secrets/passwords in env vars
   - missing securityContext
   - wildcard RBAC permissions (*)

2. **RELIABILITY ISSUES:**
   - missing resource limits (memory/cpu)
   - port mismatches between service/container
   - maxUnavailable: 100% (total downtime risk)
   - missing readiness/liveness probes

3. **CONFIGURATION ERRORS:**
   - invalid field names or values
   - missing required fields

⚠️ CRITICAL: Specify EXACT line number for each issue found.
Format: "Line X: [SEVERITY] Issue description"
Example: "Line 15: CRITICAL - privileged: true allows container escape to host system"""
        enhanced_review = await DICL.evolve(pr_data, base_prompt)
        return enhanced_review

    def _generate_conclusion(self, f1_improvement: float, threshold: float) -> str:
        if f1_improvement >= threshold:
            return f"✅ DICL significantly improves GPT-4 performance by {f1_improvement:.3f} F1 points"
        elif f1_improvement > 0:
            return f"🟡 DICL shows modest improvement of {f1_improvement:.3f} F1 points (below significance threshold)"
        elif abs(f1_improvement) < 0.01:
            return f"➖ DICL shows no meaningful change in performance ({f1_improvement:+.3f} F1 points)"
        else:
            return f"❌ DICL appears to hurt performance by {abs(f1_improvement):.3f} F1 points"

async def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Test DICL Context Impact on GPT-4 Performance"
    )
    parser.add_argument(
        "--test-cases",
        type=int,
        default=15,
        help="Number of test cases to use (default: 15)",
    )
    args = parser.parse_args()
    tester = DICLImprovementTester()
    print("🎯 DICL Context Impact Evaluation")
    start_time = time.time()
    analysis = await tester.test_dicl_improvement(args.test_cases)
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"📊 Test Cases: {analysis['test_cases']}")
    print()
    print(f"F1 Score Improvement: {analysis['f1_improvement']:+.3f}")
    print(f"Precision Improvement: {analysis['precision_improvement']:+.3f}")
    print()
    print(f"{analysis['conclusion']}")
    print()

if __name__ == "__main__":
    asyncio.run(main())
