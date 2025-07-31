import asyncio
import time
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from .test_cases import KubernetesTestCases, TestCase, ErrorCategory
from .metrics import (
    ErrorDetectionParser, MetricsCalculator,
    ModelPerformanceReport, EvaluationReporter
)

class ModelEvaluationRunner:
    def __init__(self, results_dir: str = "evaluation_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.test_cases = KubernetesTestCases.get_all_test_cases()
        
    async def evaluate_model(self, model_name: str, test_subset: Optional[List[str]] = None,
                           categories: Optional[List[ErrorCategory]] = None) -> ModelPerformanceReport:
        print(f"\n🧪 Starting evaluation for {model_name}")
        print("=" * 50)
        
        test_cases_to_run = self._filter_test_cases(test_subset, categories)
        print(f"Running {len(test_cases_to_run)} test cases")
        
        results = []
        total_time = 0
        
        for i, test_case in enumerate(test_cases_to_run, 1):
            print(f"\n[{i}/{len(test_cases_to_run)}] Testing {test_case.id}: {test_case.title}")
            
            start_time = time.time()
            
            try:
                review_output = await self._get_model_review(model_name, test_case)
                detected_errors = ErrorDetectionParser.parse_review_output(review_output, model_name)
                result = MetricsCalculator.evaluate_test_case(test_case, detected_errors, model_name)
                result.execution_time = time.time() - start_time
                
                results.append(result)
                
                print(f"  ✅ Precision: {result.precision:.3f}, Recall: {result.recall:.3f}, F1: {result.f1_score:.3f}")
                print(f"  📊 TP: {len(result.true_positives)}, FP: {len(result.false_positives)}, FN: {len(result.false_negatives)}")
                
                total_time += result.execution_time
                
            except Exception as e:
                print(f"  ❌ Error evaluating {test_case.id}: {e}")
                continue
        
        report = MetricsCalculator.aggregate_results(results)
        
        print(f"\n🎯 Evaluation Complete for {model_name}")
        print(f"Overall F1 Score: {report.overall_f1_score:.3f}")
        print(f"Total Time: {total_time:.1f}s")
        
        self._save_evaluation_results(report)
        
        return report
    
    async def compare_models(self, model_names: List[str], test_subset: Optional[List[str]] = None,
                           categories: Optional[List[ErrorCategory]] = None) -> Dict[str, ModelPerformanceReport]:
        print(f"\n🏁 Starting model comparison: {', '.join(model_names)}")
        print("=" * 60)
        
        reports = {}
        
        for model_name in model_names:
            reports[model_name] = await self.evaluate_model(model_name, test_subset, categories)
        
        comparison_report = EvaluationReporter.compare_models(list(reports.values()))
        
        print(f"\n📊 Model Comparison Complete")
        print("\n" + "=" * 60)
        print(comparison_report)
        
        return reports
    
    async def run_regression_test(self, baseline_model: str, new_model: str, 
                                improvement_threshold: float = 0.05) -> Dict[str, Any]:
        print(f"\n🔄 Running regression test: {new_model} vs {baseline_model}")
        
        baseline_report = await self.evaluate_model(baseline_model)
        new_report = await self.evaluate_model(new_model)
        
        f1_improvement = new_report.overall_f1_score - baseline_report.overall_f1_score
        precision_improvement = new_report.overall_precision - baseline_report.overall_precision
        recall_improvement = new_report.overall_recall - baseline_report.overall_recall
        
        regression_results = {
            "baseline_model": baseline_model,
            "new_model": new_model,
            "baseline_f1": baseline_report.overall_f1_score,
            "new_f1": new_report.overall_f1_score,
            "f1_improvement": f1_improvement,
            "precision_improvement": precision_improvement,
            "recall_improvement": recall_improvement,
            "improvement_threshold": improvement_threshold,
            "passed": f1_improvement >= improvement_threshold,
            "timestamp": datetime.now().isoformat()
        }
        
        category_improvements = {}
        for category in baseline_report.category_performance:
            if category in new_report.category_performance:
                baseline_f1 = baseline_report.category_performance[category]['f1_score']
                new_f1 = new_report.category_performance[category]['f1_score']
                category_improvements[category] = new_f1 - baseline_f1
        
        regression_results["category_improvements"] = category_improvements
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        regression_file = self.results_dir / f"regression_test_{timestamp}.json"
        with open(regression_file, 'w') as f:
            json.dump(regression_results, f, indent=2)
        
        status = "✅ PASSED" if regression_results["passed"] else "❌ FAILED"
        print(f"\n{status} Regression Test Results:")
        print(f"F1 Score Change: {f1_improvement:+.3f} (threshold: {improvement_threshold:+.3f})")
        print(f"Precision Change: {precision_improvement:+.3f}")
        print(f"Recall Change: {recall_improvement:+.3f}")
        
        return regression_results
    
    async def _get_model_review(self, model_name: str, test_case: TestCase) -> str:
        from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler
        
        pr_data = {
            "title": test_case.title,
            "description": test_case.description,
            "changed_files": [test_case.file_path],
            "language": test_case.language,
            "diff_text": f"--- /dev/null\n+++ {test_case.file_path}\n@@ -0,0 +1,{len(test_case.code_content.split())} @@\n" + 
                       "\n".join([f"+{line}" for line in test_case.code_content.split('\n')])
        }
        
        review_prompt = f"""You are a senior DevOps engineer reviewing Kubernetes YAML configurations. Analyze the following code for potential issues.

## File: {test_case.file_path}
## Context: {test_case.description}

```yaml
{test_case.code_content}
```

## Instructions:
1. Review the YAML configuration line by line
2. Identify any security, performance, reliability, or configuration issues
3. For each issue found, specify:
   - The exact line number where the issue occurs
   - A clear description of the problem
   - The severity level (critical/high/medium/low)
   - A suggested fix

## Focus Areas:
- Security vulnerabilities (privileged containers, hardcoded secrets, missing security contexts)
- Resource management issues (missing limits, excessive requests)
- Networking problems (port mismatches, DNS issues)
- Configuration errors (invalid values, missing required fields)
- Performance issues (inefficient settings, slow storage)
- Reliability concerns (missing disruption budgets, single points of failure)
- Observability gaps (missing monitoring, poor logging)
- RBAC issues (excessive permissions)

Provide your findings in a clear, structured format with line numbers."""
        
        ai_handler = LiteLLMAIHandler()
        
        model_mapping = {
            'gpt-4': 'gpt-4',
            'gpt-4o': 'gpt-4o',
            'gpt-4o-mini': 'gpt-4o-mini',
            'o3-mini': 'o3-mini',
            'claude-3-sonnet': 'claude-3-sonnet-20240229',
            'claude-3-haiku': 'claude-3-haiku-20240307'
        }
        
        actual_model = model_mapping.get(model_name, model_name)
        
        response, _ = await ai_handler.chat_completion(
            model=actual_model,
            temperature=0.1,
            system="You are an expert Kubernetes security and configuration reviewer. Focus on finding real issues with specific line numbers.",
            user=review_prompt
        )
        
        return response
    
    def _filter_test_cases(self, test_subset: Optional[List[str]] = None,
                          categories: Optional[List[ErrorCategory]] = None) -> List[TestCase]:
        test_cases = self.test_cases
        
        if test_subset:
            test_cases = [tc for tc in test_cases if tc.id in test_subset]
        
        if categories:
            filtered_cases = []
            for tc in test_cases:
                if any(error.category in categories for error in tc.known_errors):
                    filtered_cases.append(tc)
            test_cases = filtered_cases
        
        return test_cases
    
    def _save_evaluation_results(self, report: ModelPerformanceReport):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe_name = report.model_name.replace("-", "_").replace(":", "_")
        json_file = self.results_dir / f"{model_safe_name}_results_{timestamp}.json"
        EvaluationReporter.save_results_json(report, str(json_file))
        print(f"📁 Results saved: {json_file}")

class QuickTestRunner:
    @staticmethod
    async def test_single_case(model_name: str, test_case_id: str):
        runner = ModelEvaluationRunner()
        test_case = KubernetesTestCases.get_test_case_by_id(test_case_id)
        
        print(f"🔍 Testing {test_case_id} with {model_name}")
        print(f"Description: {test_case.description}")
        print(f"Known errors: {len(test_case.known_errors)}")
        
        review_output = await runner._get_model_review(model_name, test_case)
        print(f"\n📝 Model Review:\n{review_output}")
        
        detected_errors = ErrorDetectionParser.parse_review_output(review_output, model_name)
        print(f"\n🎯 Detected {len(detected_errors)} errors:")
        for error in detected_errors:
            print(f"  - Line {error.line_number}: {error.error_description}")
        
        result = MetricsCalculator.evaluate_test_case(test_case, detected_errors, model_name)
        print(f"\n📊 Results:")
        print(f"  - Precision: {result.precision:.3f}")
        print(f"  - Recall: {result.recall:.3f}")
        print(f"  - F1 Score: {result.f1_score:.3f}")
        print(f"  - True Positives: {len(result.true_positives)}")
        print(f"  - False Positives: {len(result.false_positives)}")
        print(f"  - False Negatives: {len(result.false_negatives)}")
        
        return result
    
    @staticmethod
    async def quick_comparison(model1: str, model2: str, num_cases: int = 5):
        runner = ModelEvaluationRunner()
        
        test_cases = KubernetesTestCases.get_all_test_cases()[:num_cases]
        test_ids = [tc.id for tc in test_cases]
        
        print(f"🏃‍♂️ Quick comparison: {model1} vs {model2} on {num_cases} cases")
        
        reports = await runner.compare_models([model1, model2], test_subset=test_ids)
        
        return reports

# CLI Interface for easy usage
async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='PR Agent Model Evaluation System')
    parser.add_argument('--model', type=str, help='Model to evaluate')
    parser.add_argument('--compare', nargs='+', help='Models to compare')
    parser.add_argument('--test-case', type=str, help='Single test case ID to run')
    parser.add_argument('--category', type=str, choices=[c.value for c in ErrorCategory], 
                       help='Filter by error category')
    parser.add_argument('--quick', action='store_true', help='Quick test with 5 cases')
    parser.add_argument('--regression', nargs=2, metavar=('BASELINE', 'NEW'), 
                       help='Run regression test')
    
    args = parser.parse_args()
    
    runner = ModelEvaluationRunner()
    
    if args.test_case and args.model:   
        await QuickTestRunner.test_single_case(args.model, args.test_case)
    
    elif args.compare:
        if args.quick:
            await QuickTestRunner.quick_comparison(args.compare[0], args.compare[1])
        else:
            categories = [ErrorCategory(args.category)] if args.category else None
            await runner.compare_models(args.compare, categories=categories)
    
    elif args.regression:
        await runner.run_regression_test(args.regression[0], args.regression[1])
    
    elif args.model:
        categories = [ErrorCategory(args.category)] if args.category else None
        await runner.evaluate_model(args.model, categories=categories)
    
    else:
        print("Please specify --model, --compare, --test-case, or --regression")
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())