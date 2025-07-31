from typing import Dict, Any
from pr_agent.dicl.auto_learning import DualModelReviewer
import asyncio
from pr_agent.algo.rag_handler import RAGHandler
import concurrent.futures

class DICL:
    @staticmethod
    async def evolve(pr_data: Dict[str, Any], base_prompt: str) -> str:
        print("Retrieving insights from previous reviews...")

        dual_reviewer = DualModelReviewer()
        enhanced_review, insights_count = await dual_reviewer.dual_review(
            pr_data, base_prompt
        )

        if insights_count > 0:
            print(f"Generated {insights_count} new insights stored for future reviews")
            print("New insights will enhance future reviews")
        else:
            print("No new insights generated (high model agreement)")

        return enhanced_review

    @staticmethod
    def regular(pr_data: Dict[str, Any], base_prompt: str) -> str:
        try:
            rag = RAGHandler()
            title = pr_data.get("title", "")
            description = pr_data.get("description", "")
            changed_files = pr_data.get("changed_files", [])
            language = pr_data.get("language", None)

            try:
                asyncio.get_running_loop()

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        rag.get_relevant_learning_insights_with_context(
                            pr_title=title,
                            pr_description=description,
                            changed_files=changed_files,
                            language=language,
                        ),
                    )
                    insights = future.result(timeout=15)
            except RuntimeError:
                insights = asyncio.run(
                    rag.get_relevant_learning_insights_with_context(
                        pr_title=title,
                        pr_description=description,
                        changed_files=changed_files,
                        language=language,
                    )
                )
            except Exception:
                search_query = f"{title} {description}"
                if changed_files:
                    search_query += f" {' '.join(changed_files[:3])}"
                if language:
                    search_query += f" {language}"
                insights = rag.get_relevant_learning_insights(search_query)

        except Exception:
            return base_prompt

        if not insights:
            return base_prompt

        learning_section = f"CRITICAL LEARNING PATTERNS FROM PREVIOUS REVIEWS\\n"
        learning_section += f"PRIORITY INSTRUCTIONS: Use these {len(insights)} proven patterns to identify issues other reviewers missed:\\n"

        for i, insight in enumerate(insights[:5], 1):
            if isinstance(insight, dict) and "insight" in insight:
                score = insight.get("similarity_score", 0.0)
                learning_section += f"**PATTERN {i}** (relevance: {score:.2f}):\\n{insight['insight']}\\n\\n"
            elif isinstance(insight, str):
                learning_section += f"**PATTERN {i}:**\\n{insight}\\n\\n"

        learning_section += """
        APPLICATION MANDATE: For each file you review:
            1. Check for ALL patterns listed above that apply to this code type
            2. Reference specific line numbers when you find matches
            3. Prioritize issues that these patterns specifically warn about
            4. If you don't find pattern-related issues, explicitly state why they don't apply
        """

        return base_prompt + learning_section
