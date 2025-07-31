
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Set, Tuple
import json
from datetime import datetime

from .test_cases import TestCase, KnownError, ErrorCategory, ErrorSeverity

@dataclass
class DetectedError:
    line_number: int
    error_description: str
    severity_mentioned: str = None
    confidence: float = 1.0
    model_name: str = None

@dataclass
class EvaluationResult:
    test_case_id: str
    model_name: str
    true_positives: List[Tuple[KnownError, DetectedError]]
    false_positives: List[DetectedError]
    false_negatives: List[KnownError]
    precision: float
    recall: float
    f1_score: float
    category_scores: Dict[str, Dict[str, float]]
    severity_scores: Dict[str, Dict[str, float]]
    execution_time: float = 0.0

@dataclass
class ModelPerformanceReport:
    model_name: str
    total_test_cases: int
    overall_precision: float
    overall_recall: float
    overall_f1_score: float
    category_performance: Dict[str, Dict[str, float]]
    severity_performance: Dict[str, Dict[str, float]]
    detailed_results: List[EvaluationResult]
    strengths: List[str]
    weaknesses: List[str]
    improvement_suggestions: List[str]

class ErrorDetectionParser:
    
    @staticmethod
    def parse_review_output(review_text: str, model_name: str = "unknown") -> List[DetectedError]:
        detected_errors = []
        
        patterns = [
            r'(\w+\.ya?ml):(\d+)\s*[-:]?\s*(.+?)(?:\n|$)',
            r'[Ll]ine\s+(\d+):?\s*(.+?)(?:\n|$)',
            r'at\s+line\s+(\d+)[,:]\s*(.+?)(?:\n|$)',
            r'\(line\s+(\d+)\)\s*(.+?)(?:\n|$)',
            r'(?:issue|problem|error|warning|concern|risk)[:\s]+(.+?)(?:\n|$)',
        ]
        
        lines = review_text.split('\n')
        
        for line_text in lines:
            line_text = line_text.strip()
            if not line_text or len(line_text) < 10:
                continue
                
            for pattern in patterns[:4]:
                matches = re.finditer(pattern, line_text, re.IGNORECASE)
                for match in matches:
                    if len(match.groups()) == 3:
                        line_num = int(match.group(2))
                        description = match.group(3).strip()
                    else:
                        line_num = int(match.group(1))
                        description = match.group(2).strip()
                    
                    if description and len(description) > 5:
                        severity = ErrorDetectionParser._extract_severity(line_text)
                        detected_errors.append(DetectedError(
                            line_number=line_num,
                            error_description=description,
                            severity_mentioned=severity,
                            model_name=model_name
                        ))
                        break
            
            if not any(re.search(pattern, line_text, re.IGNORECASE) for pattern in patterns[:4]):
                for pattern in patterns[4:]:
                    matches = re.finditer(pattern, line_text, re.IGNORECASE)
                    for match in matches:
                        description = match.group(1).strip()
                        if description and len(description) > 10:
                            line_num = ErrorDetectionParser._extract_line_number(description)
                            severity = ErrorDetectionParser._extract_severity(line_text)
                            detected_errors.append(DetectedError(
                                line_number=line_num or 0,
                                error_description=description,
                                severity_mentioned=severity,
                                model_name=model_name
                            ))
                            break
        
        return ErrorDetectionParser._deduplicate_errors(detected_errors)
    
    @staticmethod
    def _extract_severity(text: str) -> str:
        text = text.lower()
        if any(word in text for word in ['critical', 'severe', 'high']):
            return 'high'
        elif any(word in text for word in ['medium', 'moderate']):
            return 'medium'
        elif any(word in text for word in ['low', 'minor', 'info']):
            return 'low'
        return None
    
    @staticmethod
    def _extract_line_number(text: str) -> int:
        line_patterns = [
            r'line[:\s]+(\d+)',
            r':\s*(\d+)',
            r'@\s*(\d+)',
        ]
        
        for pattern in line_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None
    
    @staticmethod
    def _deduplicate_errors(errors: List[DetectedError]) -> List[DetectedError]:
        if not errors:
            return []
        
        deduplicated = []
        seen_combinations = set()
        
        for error in errors:
            desc_key = ErrorDetectionParser._normalize_description(error.error_description)
            key = (error.line_number, desc_key)
            
            if key not in seen_combinations:
                seen_combinations.add(key)
                deduplicated.append(error)
        
        return deduplicated
    
    @staticmethod
    def _normalize_description(description: str) -> str:
        normalized = re.sub(r'^(error|warning|issue|problem)[:\s]*', '', description.lower())
        normalized = re.sub(r'\s+', ' ', normalized.strip())
        return normalized[:50]

class MetricsCalculator:
    
    @staticmethod
    def evaluate_test_case(test_case: TestCase, detected_errors: List[DetectedError], 
                          model_name: str) -> EvaluationResult:
        true_positives = []
        false_positives = []
        false_negatives = []
        
        known_errors_matched = set()
        detected_errors_matched = set()
        
        for i, detected in enumerate(detected_errors):
            best_match = None
            best_score = 0
            
            for j, known in enumerate(test_case.known_errors):
                if j in known_errors_matched:
                    continue
                    
                match_score = MetricsCalculator._calculate_match_score(detected, known)
                if match_score > best_score and match_score >= 0.3:
                    best_score = match_score
                    best_match = j
            
            if best_match is not None:
                true_positives.append((test_case.known_errors[best_match], detected))
                known_errors_matched.add(best_match)
                detected_errors_matched.add(i)
        
        for i, detected in enumerate(detected_errors):
            if i not in detected_errors_matched:
                false_positives.append(detected)
        
        for j, known in enumerate(test_case.known_errors):
            if j not in known_errors_matched:
                false_negatives.append(known)
        
        precision = len(true_positives) / len(detected_errors) if detected_errors else 0.0
        recall = len(true_positives) / len(test_case.known_errors) if test_case.known_errors else 1.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        category_scores = MetricsCalculator._calculate_category_scores(
            true_positives, false_positives, false_negatives
        )
        severity_scores = MetricsCalculator._calculate_severity_scores(
            true_positives, false_positives, false_negatives
        )
        
        return EvaluationResult(
            test_case_id=test_case.id,
            model_name=model_name,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            category_scores=category_scores,
            severity_scores=severity_scores
        )
    
    @staticmethod
    def _calculate_match_score(detected: DetectedError, known: KnownError) -> float:
        score = 0.0
        
        if detected.line_number > 0:
            line_diff = abs(detected.line_number - known.line_number)
            if line_diff == 0:
                score += 0.4
            elif line_diff <= 2:
                score += 0.3
            elif line_diff <= 5:
                score += 0.2
        
        desc_similarity = MetricsCalculator._calculate_text_similarity(
            detected.error_description.lower(),
            known.description.lower()
        )
        score += 0.5 * desc_similarity
        
        if known.error_type.lower() in detected.error_description.lower():
            score += 0.1
        
        return min(score, 1.0)
    
    @staticmethod
    def _calculate_text_similarity(text1: str, text2: str) -> float:
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    @staticmethod  
    def _calculate_category_scores(true_positives: List[Tuple[KnownError, DetectedError]], 
                                 false_positives: List[DetectedError],
                                 false_negatives: List[KnownError]) -> Dict[str, Dict[str, float]]:
        category_stats = {}
        
        for known_error, _ in true_positives:
            category = known_error.category.value
            if category not in category_stats:
                category_stats[category] = {'tp': 0, 'fp': 0, 'fn': 0}
            category_stats[category]['tp'] += 1
        
        for known_error in false_negatives:
            category = known_error.category.value
            if category not in category_stats:
                category_stats[category] = {'tp': 0, 'fp': 0, 'fn': 0}
            category_stats[category]['fn'] += 1
        
        category_scores = {}
        for category, stats in category_stats.items():
            tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            category_scores[category] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        return category_scores
    
    @staticmethod
    def _calculate_severity_scores(true_positives: List[Tuple[KnownError, DetectedError]], 
                                 false_positives: List[DetectedError],
                                 false_negatives: List[KnownError]) -> Dict[str, Dict[str, float]]:
        severity_stats = {}
        
        for known_error, _ in true_positives:
            severity = known_error.severity.value
            if severity not in severity_stats:
                severity_stats[severity] = {'tp': 0, 'fp': 0, 'fn': 0}
            severity_stats[severity]['tp'] += 1
        
        for known_error in false_negatives:
            severity = known_error.severity.value
            if severity not in severity_stats:
                severity_stats[severity] = {'tp': 0, 'fp': 0, 'fn': 0}
            severity_stats[severity]['fn'] += 1
        
        severity_scores = {}
        for severity, stats in severity_stats.items():
            tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            severity_scores[severity] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        return severity_scores
    
    @staticmethod
    def aggregate_results(results: List[EvaluationResult]) -> ModelPerformanceReport:
        if not results:
            return ModelPerformanceReport(
                model_name="unknown",
                total_test_cases=0,
                overall_precision=0.0,
                overall_recall=0.0,
                overall_f1_score=0.0,
                category_performance={},
                severity_performance={},
                detailed_results=[],
                strengths=[],
                weaknesses=[],
                improvement_suggestions=[]
            )
        
        model_name = results[0].model_name
        total_cases = len(results)
        
        total_weight = 0
        weighted_precision = 0
        weighted_recall = 0
        weighted_f1 = 0
        
        category_aggregates = {}
        severity_aggregates = {}
        
        for result in results:
            weight = len(result.true_positives) + len(result.false_negatives)
            if weight == 0:
                weight = 1
                
            total_weight += weight
            weighted_precision += result.precision * weight
            weighted_recall += result.recall * weight
            weighted_f1 += result.f1_score * weight
            
            for category, scores in result.category_scores.items():
                if category not in category_aggregates:
                    category_aggregates[category] = {'total_weight': 0, 'weighted_precision': 0, 
                                                   'weighted_recall': 0, 'weighted_f1': 0}
                category_aggregates[category]['total_weight'] += weight
                category_aggregates[category]['weighted_precision'] += scores['precision'] * weight
                category_aggregates[category]['weighted_recall'] += scores['recall'] * weight
                category_aggregates[category]['weighted_f1'] += scores['f1_score'] * weight
            
            for severity, scores in result.severity_scores.items():
                if severity not in severity_aggregates:
                    severity_aggregates[severity] = {'total_weight': 0, 'weighted_precision': 0, 
                                                   'weighted_recall': 0, 'weighted_f1': 0}
                severity_aggregates[severity]['total_weight'] += weight
                severity_aggregates[severity]['weighted_precision'] += scores['precision'] * weight
                severity_aggregates[severity]['weighted_recall'] += scores['recall'] * weight
                severity_aggregates[severity]['weighted_f1'] += scores['f1_score'] * weight
        
        overall_precision = weighted_precision / total_weight if total_weight > 0 else 0.0
        overall_recall = weighted_recall / total_weight if total_weight > 0 else 0.0
        overall_f1 = weighted_f1 / total_weight if total_weight > 0 else 0.0
        
        category_performance = {}
        for category, agg in category_aggregates.items():
            w = agg['total_weight']
            category_performance[category] = {
                'precision': agg['weighted_precision'] / w if w > 0 else 0.0,
                'recall': agg['weighted_recall'] / w if w > 0 else 0.0,
                'f1_score': agg['weighted_f1'] / w if w > 0 else 0.0
            }
        
        severity_performance = {}
        for severity, agg in severity_aggregates.items():
            w = agg['total_weight']
            severity_performance[severity] = {
                'precision': agg['weighted_precision'] / w if w > 0 else 0.0,
                'recall': agg['weighted_recall'] / w if w > 0 else 0.0,
                'f1_score': agg['weighted_f1'] / w if w > 0 else 0.0
            }
        
        strengths, weaknesses, suggestions = MetricsCalculator._analyze_performance(
            category_performance, severity_performance, overall_precision, overall_recall, overall_f1
        )
        
        return ModelPerformanceReport(
            model_name=model_name,
            total_test_cases=total_cases,
            overall_precision=overall_precision,
            overall_recall=overall_recall,
            overall_f1_score=overall_f1,
            category_performance=category_performance,
            severity_performance=severity_performance,
            detailed_results=results,
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_suggestions=suggestions
        )
    
    @staticmethod
    def _analyze_performance(category_perf: Dict, severity_perf: Dict, 
                           precision: float, recall: float, f1: float) -> Tuple[List[str], List[str], List[str]]:
        strengths = []
        weaknesses = []
        suggestions = []
        
        if f1 >= 0.8:
            strengths.append(f"Excellent overall performance (F1: {f1:.3f})")
        elif f1 >= 0.6:
            strengths.append(f"Good overall performance (F1: {f1:.3f})")
        elif f1 < 0.4:
            weaknesses.append(f"Low overall performance (F1: {f1:.3f})")
            suggestions.append("Consider improving error detection patterns and training")
        
        if precision > recall + 0.2:
            strengths.append("High precision - few false positives")
            suggestions.append("Focus on improving recall to catch more issues")
        elif recall > precision + 0.2:
            strengths.append("High recall - catches most issues")
            suggestions.append("Focus on improving precision to reduce false positives")
        elif abs(precision - recall) < 0.1:
            strengths.append("Well-balanced precision and recall")
        
        best_categories = []
        worst_categories = []
        
        for category, scores in category_perf.items():
            if scores['f1_score'] >= 0.7:
                best_categories.append(f"{category} (F1: {scores['f1_score']:.3f})")
            elif scores['f1_score'] < 0.3:
                worst_categories.append(f"{category} (F1: {scores['f1_score']:.3f})")
        
        if best_categories:
            strengths.append(f"Strong performance in: {', '.join(best_categories)}")
        
        if worst_categories:
            weaknesses.append(f"Weak performance in: {', '.join(worst_categories)}")
            suggestions.append(f"Improve detection patterns for {', '.join([c.split(' (')[0] for c in worst_categories])}")
        
        for severity, scores in severity_perf.items():
            if severity == 'critical' and scores['recall'] < 0.8:
                weaknesses.append(f"Missing critical issues (recall: {scores['recall']:.3f})")
                suggestions.append("Prioritize detection of critical security and reliability issues")
            elif severity == 'low' and scores['precision'] < 0.6:
                weaknesses.append(f"Too many false positives in low-severity issues")
                suggestions.append("Reduce noise by improving low-severity issue detection")
        
        return strengths, weaknesses, suggestions

class EvaluationReporter:
    
    @staticmethod
    def save_results_json(report: ModelPerformanceReport, filepath: str):
        data = {
            "model_name": report.model_name,
            "timestamp": datetime.now().isoformat(),
            "total_test_cases": report.total_test_cases,
            "overall_metrics": {
                "precision": report.overall_precision,
                "recall": report.overall_recall,
                "f1_score": report.overall_f1_score
            },
            "category_performance": report.category_performance,
            "severity_performance": report.severity_performance,
            "strengths": report.strengths,
            "weaknesses": report.weaknesses,
            "improvement_suggestions": report.improvement_suggestions,
            "detailed_results": [
                {
                    "test_case_id": result.test_case_id,
                    "precision": result.precision,
                    "recall": result.recall,
                    "f1_score": result.f1_score,
                    "true_positives_count": len(result.true_positives),
                    "false_positives_count": len(result.false_positives),
                    "false_negatives_count": len(result.false_negatives)
                }
                for result in report.detailed_results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def compare_models(reports: List[ModelPerformanceReport]) -> str:
        if len(reports) < 2:
            return "Need at least 2 models for comparison"
        
        lines = []
        lines.append("# Model Comparison Report")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        lines.append("## Overall Performance Comparison")
        lines.append("| Model | Precision | Recall | F1 Score |")
        lines.append("|-------|-----------|--------|----------|")
        
        for report in sorted(reports, key=lambda r: r.overall_f1_score, reverse=True):
            lines.append(f"| {report.model_name} | {report.overall_precision:.3f} | "
                        f"{report.overall_recall:.3f} | {report.overall_f1_score:.3f} |")
        lines.append("")
        
        best_model = max(reports, key=lambda r: r.overall_f1_score)
        worst_model = min(reports, key=lambda r: r.overall_f1_score)
        
        lines.append("## Key Findings")
        lines.append(f"- **Best Overall**: {best_model.model_name} (F1: {best_model.overall_f1_score:.3f})")
        lines.append(f"- **Needs Improvement**: {worst_model.model_name} (F1: {worst_model.overall_f1_score:.3f})")
        
        all_categories = set()
        for report in reports:
            all_categories.update(report.category_performance.keys())
        
        lines.append("")
        lines.append("## Category Leaders")
        for category in sorted(all_categories):
            category_scores = []
            for report in reports:
                if category in report.category_performance:
                    category_scores.append((report.model_name, report.category_performance[category]['f1_score']))
            
            if category_scores:
                best_in_category = max(category_scores, key=lambda x: x[1])
                lines.append(f"- **{category.replace('_', ' ').title()}**: {best_in_category[0]} ({best_in_category[1]:.3f})")
        
        return "\n".join(lines)