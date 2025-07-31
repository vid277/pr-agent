from typing import Dict, List, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
import asyncio
from pr_agent.log import get_logger
import concurrent.futures
from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler
from pr_agent.algo.rag_handler import RAGHandler


@dataclass
class ModelComparison:
    gpt4_review: str
    o3_review: str
    differences: List[str]
    unique_to_gpt4: List[str]
    unique_to_o3: List[str]
    agreement_score: float
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class AutoLearningEngine:
    def __init__(self):
        self.logger = get_logger()

    def compare_models(self, gpt4_review: str, o3_review: str) -> ModelComparison:
        gpt4_issues = self._extract_issues(gpt4_review)
        o3_issues = self._extract_issues(o3_review)

        unique_to_gpt4 = [
            issue
            for issue in gpt4_issues
            if not self._similar_issue_exists(issue, o3_issues)
        ]
        unique_to_o3 = [
            issue
            for issue in o3_issues
            if not self._similar_issue_exists(issue, gpt4_issues)
        ]

        common_issues = (
            len(gpt4_issues) + len(o3_issues) - len(unique_to_gpt4) - len(unique_to_o3)
        )
        total_issues = len(gpt4_issues) + len(o3_issues)

        agreement_score = (
            1.0 if total_issues == 0 else min(1.0, (2 * common_issues) / total_issues)
        )

        return ModelComparison(
            gpt4_review=gpt4_review,
            o3_review=o3_review,
            differences=unique_to_gpt4 + unique_to_o3,
            unique_to_gpt4=unique_to_gpt4,
            unique_to_o3=unique_to_o3,
            agreement_score=agreement_score,
        )

    def _extract_issues(self, review_text: str) -> List[str]:
        indicators = [
            "issue:",
            "problem:",
            "concern:",
            "warning:",
            "error:",
            "missing:",
            "should:",
            "consider:",
            "recommend:",
            "fix:",
        ]
        return [
            line[:200]
            for line in review_text.split("\n")
            if any(indicator in line.strip().lower() for indicator in indicators)
            and len(line.strip()) > 10
        ]

    def _similar_issue_exists(self, issue: str, issue_list: List[str]) -> bool:
        key_words = [word for word in issue.lower().split() if len(word) > 3]
        return any(
            sum(1 for word in key_words if word in existing.lower()) >= 2
            for existing in issue_list
        )

    async def gen_insights(
        self, comparison: ModelComparison, pr_context: Dict[str, Any]
    ) -> List[str]:
        if not comparison.gpt4_review or not comparison.o3_review:
            return []

        if not comparison.gpt4_review.strip() or not comparison.o3_review.strip():
            self.logger.info(
                "No substantial reviews from either model, skipping insight generation"
            )
            return []

        insights = await self.gen_insights_with_judge(comparison, pr_context)

        quality_insights = [
            insight
            for insight in insights
            if len(insight) > 50
            and not any(
                term in insight.lower()
                for term in [
                    "gpt-4",
                    "o3",
                    "model",
                    "correctly identified",
                    "missed",
                ]
            )
        ]

        self.logger.info(
            f"LLM Judge generated {len(insights)} insights, {len(quality_insights)} passed quality filter"
        )
        return quality_insights

    async def gen_insights_with_judge(
        self, comparison: ModelComparison, pr_context: Dict[str, Any]
    ) -> List[str]:
        judge_prompt = f"""Analyze these two code reviews to extract direct learning instructions for improving future reviews.

## Context
- Language: {pr_context.get('language', 'unknown')}
- Files: {', '.join(pr_context.get('changed_files', [])[:3])}

## GPT-4 Review:
{comparison.gpt4_review[:2000]}

## O3 Review:
{comparison.o3_review[:2000]}

## Task
Where O3 found issues that GPT-4 missed, extract direct instructions for what GPT-4 should do differently next time.

Generate insights as direct instructions in this format:
"When reviewing [code pattern], always check for [specific issue type] by [specific action]. Look for [warning signs] and prioritize [specific remediation]."

Focus on extracting O3's superior detection methods for:
- Security vulnerabilities
- Performance issues  
- Testing gaps
- Logic errors
- Resource management problems

Each insight should be a direct instruction that teaches GPT-4 to replicate O3's superior analysis.

Output format: One direct instruction per line."""

        ai_handler = LiteLLMAIHandler()
        response, _ = await ai_handler.chat_completion(
            model="gpt-4o-mini",
            temperature=0.1,
            system="""You are extracting direct learning instructions to improve code review performance. Focus on what the superior model did that should be replicated.

Generate direct instructions that follow this pattern:
"When reviewing [code pattern], always [specific action] to detect [issue type]. Look for [warning signs] and prioritize [remediation approach]."

Requirements:
- Direct Instructions: Tell GPT-4 exactly what to do differently
- Actionable Methods: Extract O3's superior detection techniques
- Specific Actions: Focus on concrete steps, not abstract concepts
- No Meta-Commentary: Avoid mentioning model names or comparisons

Create instructions that directly improve future review capabilities by teaching O3's superior methods.""",
            user=judge_prompt,
        )

        return [
            line.strip()
            for line in response.strip().split("\n")
            if len(line.strip()) > 40
        ]

    def store_insights(self, insights: List[str], pr_context: Dict[str, Any]) -> bool:
        if not insights:
            self.logger.info("No insights to store, skipping RAG initialization")
            return False

        rag = RAGHandler()
        stored_count = sum(
            1
            for insight in insights
            if rag.add_learning_insight(
                insight_text=insight,
                pr_id=pr_context.get("pr_id", "auto_learning"),
                agreement_score=0.8,
                metadata={
                    "learning_type": "automated_comparison",
                    "timestamp": datetime.now().isoformat(),
                    "pr_language": pr_context.get("language", "unknown"),
                    "pr_files": pr_context.get("changed_files", [])[:10],
                    "automated": True,
                },
            )
        )
        self.logger.info(
            f"Stored {stored_count}/{len(insights)} automated learning insights"
        )
        return stored_count > 0


class DualModelReviewer:
    def __init__(self):
        self.learning_engine = AutoLearningEngine()
        self.logger = get_logger()

    async def dual_review(
        self, pr_data: Dict[str, Any], base_prompt: str
    ) -> Tuple[str, int]:
        enhanced_prompts = await self.get_new_prompts(pr_data, base_prompt)

        print("\n" + "=" * 120)
        print("===== GPT-4 PROMPT: =====")
        print("=" * 120)
        print(enhanced_prompts["gpt4"])
        print("\n" + "=" * 120)
        print("===== O3 PROMPT: =====")
        print("=" * 120)
        print(enhanced_prompts["o3"])
        print("=" * 120)

        gpt4_review = await self.get_gpt4_review(pr_data, enhanced_prompts["gpt4"])
        o3_review = await self.get_o3_review(pr_data, enhanced_prompts["o3"])

        comparison = self.learning_engine.compare_models(gpt4_review, o3_review)
        insights = await self.learning_engine.gen_insights(comparison, pr_data)
        self.learning_engine.store_insights(insights, pr_data)

        final_review = self.synthesize_reviews(gpt4_review, o3_review, comparison)
        self.logger.info(
            f"Dual model review completed. Agreement: {comparison.agreement_score:.2f}, New insights: {len(insights)}"
        )

        return final_review, len(insights)

    async def get_new_prompts(
        self, pr_data: Dict[str, Any], base_prompt: str
    ) -> Dict[str, str]:
        rag = RAGHandler()
        insights = await rag.get_relevant_learning_insights_with_context(
            pr_title=pr_data.get("title", ""),
            pr_description=pr_data.get("description", ""),
            changed_files=pr_data.get("changed_files", []),
            language=pr_data.get("language", ""),
        )

        if not insights:
            print("⚠️  No insights found, using base prompts only")
            return {"gpt4": base_prompt, "o3": base_prompt}

        learning_section = self.format_insights(insights)
        gpt4_base = (
            base_prompt[:2000] + "..." if len(base_prompt) > 2000 else base_prompt
        )

        return {
            "gpt4": f"{gpt4_base}\n\n{learning_section}",
            "o3": f"{base_prompt}\n\n{learning_section}",
        }

    def format_insights(self, insights: List[Dict[str, Any]]) -> str:
        if not insights:
            return ""

        high_relevance_insights = [
            insight
            for insight in insights
            if isinstance(insight, dict)
            and "similarity_score" in insight
            and insight.get("similarity_score", 0.0) > 0.85
            or not isinstance(insight, dict)
        ]

        if not high_relevance_insights:
            return ""

        learning_section = "\n## 🧠 LEARNING PATTERNS\n**Apply these proven detection methods ONLY if clearly relevant:**\n\n"

        for i, insight in enumerate(high_relevance_insights[:2], 1):
            if isinstance(insight, dict) and "insight" in insight:
                score = insight.get("similarity_score", 0.0)
                learning_section += f"**{i}.** ({score:.2f}) {insight['insight']}\n\n"
            elif isinstance(insight, str):
                learning_section += f"**{i}.** {insight}\n\n"

        return (
            learning_section
            + "**🎯 CRITICAL:** Only apply patterns that clearly match this specific code. If patterns don't apply, ignore them completely.\n\n"
        )

    async def get_gpt4_review(self, _: Dict[str, Any], prompt: str) -> str:
        ai_handler = LiteLLMAIHandler()
        response, _ = await ai_handler.chat_completion(
            model="gpt-4",
            temperature=0.1,
            system="""You are a senior software engineer and code review expert with 15+ years of experience, enhanced with continuous learning from previous expert reviews.

PRIORITY: Apply learning patterns carefully - they guide focus, but don't lower your standards for what constitutes a genuine issue.

ANALYSIS FRAMEWORK:
1. **Critical Issues Only**: Focus on genuine security/reliability risks, not best practices
2. Security vulnerabilities with actual exploit potential
3. Performance bottlenecks causing measurable impact
4. Configuration errors that break functionality
5. Resource management problems causing outages

ENHANCED METHODOLOGY:
- **Pattern-Guided Focus**: Use learning patterns to know WHERE to look, not WHAT to find
- **Evidence-Based**: Only flag issues with clear technical justification
- **Severity Discipline**: Reserve CRITICAL for actual critical issues (container escape, DoS, etc.)

OUTPUT REQUIREMENTS:
- **Be Conservative**: Better to miss a minor issue than create false positives
- Only report issues that have clear technical impact
- Use appropriate severity: CRITICAL = system compromise, HIGH = service impact, MEDIUM = maintainability
- Provide specific file:line references for each issue
- When learning patterns guide you to an area, verify the issue exists before reporting

Focus on genuine problems that impact security, reliability, or functionality. Learning patterns help you look in the right places, but maintain high standards for what constitutes a real issue.""",
            user=prompt,
        )
        return response

    async def get_o3_review(self, _: Dict[str, Any], prompt: str) -> str:
        ai_handler = LiteLLMAIHandler()
        response, _ = await ai_handler.chat_completion(
            model="o3-mini",
            temperature=0.05,
            system="""You are an elite code auditor with deep expertise in software architecture and security, enhanced with continuous learning from expert review patterns.

PRIORITY: Use learning patterns as investigation guides, but maintain strict standards for genuine security/reliability issues.

PRIMARY FOCUS AREAS:
1. **High-Impact Issues Only**: Focus on problems that cause actual system failures or security breaches
2. Logic flaws that break core functionality
3. Security vulnerabilities with clear exploit paths
4. Resource management problems causing service outages
5. Configuration errors that prevent deployment/operation

ENHANCED REVIEW METHODOLOGY:
- **Targeted Investigation**: Use learning patterns to focus analysis on high-risk areas
- **Evidence Required**: Only report issues with clear technical justification
- **Impact Assessment**: Verify each issue has measurable business/security impact
- **Conservative Reporting**: Avoid flagging best practices as critical issues

OUTPUT FORMAT:
- **Quality over Quantity**: Better to find 2 real issues than 5 questionable ones
- Reserve CRITICAL for actual system/security compromises
- Use HIGH only for service-impacting problems
- Provide concrete evidence for each reported issue
- When patterns guide investigation, validate findings independently

Focus on finding genuine problems that require immediate attention. Learning patterns help you investigate effectively, but don't lower the bar for what constitutes a real issue.""",
            user=prompt,
        )
        return response

    def synthesize_reviews(
        self, gpt4_review: str, o3_review: str, comparison: ModelComparison
    ) -> str:
        if comparison.agreement_score > 0.8:
            return gpt4_review if len(gpt4_review) > len(o3_review) else o3_review

        combined = f"## 🧬 DICL Enhanced Review (Multi-Model Analysis)\n\n"
        combined += f"### 🔍 Primary Analysis (GPT-4):\n{gpt4_review}\n\n"

        if comparison.unique_to_o3:
            combined += f"### 🎯 Specialized Insights (O3):\n"
            for insight in comparison.unique_to_o3[:10]:
                combined += f"- {insight}\n"

        combined += """\n### 📊 Model Consensus Analysis\n"""
        combined += f"- Agreement Score: {comparison.agreement_score:.2f}/1.0\n"
        combined += f"- Unique GPT-4 Findings: {len(comparison.unique_to_gpt4)}\n"
        combined += f"- Unique O3 Findings: {len(comparison.unique_to_o3)}\n"
        combined += f"- Analysis Quality: {'High consensus' if comparison.agreement_score >= 0.8 else 'Multi-perspective analysis required'}\n"

        return combined
