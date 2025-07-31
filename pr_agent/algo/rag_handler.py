import json
import time
import openai
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pinecone import Pinecone, ServerlessSpec
from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler
from pr_agent.config_loader import get_settings
from pr_agent.log import get_logger


@dataclass
class RAGContext:
    similar_diffs: List[Dict[str, Any]]
    similarity_scores: List[float]
    query_embedding: Optional[List[float]] = None
    learning_insights: Optional[List[Dict[str, Any]]] = None

    def get_formatted_context(self, max_context: int = 5) -> str:
        context_parts = []
        for i, (diff_data, score) in enumerate(
            zip(self.similar_diffs[:max_context], self.similarity_scores[:max_context])
        ):
            context_parts.append(
                f"### Similar Change #{i+1} (similarity: {score:.3f})\n"
                f"**Title:** {diff_data.get('title', 'N/A')}\n"
                f"**Summary:** {diff_data.get('diff_summary', 'N/A')}\n"
                f"**Files:** {', '.join(diff_data.get('changed_files', [])[:3])}\n"
                f"**Language:** {diff_data.get('language', 'N/A')}\n"
            )

        if self.learning_insights:
            context_parts.append("\n### Learning Insights from Previous Analysis:")
            for insight in self.learning_insights[:3]:
                context_parts.append(f"• {insight.get('insight', 'N/A')}")

        return "\n".join(context_parts)


class VectorDatabase:
    def __init__(self):
        self.logger = get_logger()
        api_key = get_settings().pinecone.api_key
        if not api_key:
            raise ValueError("Pinecone API key must be configured")

        self.pc = Pinecone(api_key=api_key)
        self.index_name = "pr-agent-diffs"

        if self.index_name not in [idx.name for idx in self.pc.list_indexes()]:
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        self.index = self.pc.Index(self.index_name)

    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        try:
            openai_key = get_settings().openai.key
            if not openai_key:
                self.logger.error("OpenAI API key not configured for embeddings")
                return None

            client = openai.OpenAI(api_key=openai_key)
            response = client.embeddings.create(
                model="text-embedding-ada-002", input=text
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            return None

    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        try:
            vectors_to_upsert = []
            for doc in documents:
                diff_summary = (
                    doc.get("diff_summary")
                    if isinstance(doc, dict)
                    else getattr(doc, "diff_summary", "")
                )
                embedding = self._generate_embedding(diff_summary)
                if embedding is None:
                    continue

                if isinstance(doc, dict):
                    metadata = {
                        "pr_url": doc.get("pr_url", ""),
                        "title": doc.get("title", ""),
                        "diff_summary": doc.get("diff_summary", ""),
                        "language": doc.get("language", ""),
                        "changed_files": json.dumps(doc.get("changed_files", [])),
                        "author": doc.get("author", ""),
                        "created_at": doc.get("created_at", ""),
                    }
                    vector_id = f"doc_{hash(str(doc))}"
                else:
                    metadata = {
                        "pr_url": getattr(doc, "pr_url", ""),
                        "title": getattr(doc, "title", ""),
                        "diff_summary": getattr(doc, "diff_summary", ""),
                        "language": getattr(doc, "language", ""),
                        "changed_files": json.dumps(getattr(doc, "changed_files", [])),
                        "author": getattr(doc, "author", ""),
                        "created_at": getattr(doc, "created_at", ""),
                    }
                    vector_id = f"doc_{hash(str(doc))}"

                vectors_to_upsert.append((vector_id, embedding, metadata))

            if vectors_to_upsert:
                self.index.upsert(vectors_to_upsert)
                self.logger.info(
                    f"Added {len(vectors_to_upsert)} documents to Pinecone"
                )
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to add documents: {e}")
            return False

    def search(
        self, query_embedding: List[float], k: int = 5
    ) -> Tuple[List[Dict], List[float]]:
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True,
            )

            documents = []
            scores = []
            for match in results.matches:
                metadata = match.metadata
                if metadata.get("changed_files"):
                    metadata["changed_files"] = json.loads(
                        metadata.get("changed_files", "[]")
                    )
                documents.append(metadata)
                scores.append(float(match.score))

            return documents, scores
        except Exception as e:
            self.logger.error(f"Failed to search: {e}")
            return [], []

    def get_stats(self) -> Dict[str, Any]:
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_documents": stats.total_vector_count,
                "index_name": self.index_name,
                "dimension": 1536,
            }
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {}


class RAGHandler:
    def __init__(self):
        self.logger = get_logger()
        self.ai_handler = LiteLLMAIHandler()
        self.vector_db = VectorDatabase()
        self.logger.info("Initialized RAG handler with Pinecone")

    def add_pr_diffs(self, pr_diffs: List[Dict[str, Any]]) -> bool:
        return self.vector_db.add_documents(pr_diffs)

    def get_similar_contexts(
        self,
        query_text: str,
        k: int = 5,
        language: str = None,
        include_learning: bool = True,
    ) -> RAGContext:
        try:
            query_embedding = self.vector_db._generate_embedding(query_text)
            if query_embedding is None:
                return RAGContext([], [])

            documents, scores = self.vector_db.search(
                query_embedding, k
            )

            if language:
                filtered_docs = []
                filtered_scores = []
                for doc, score in zip(documents, scores):
                    if doc.get("language", "").lower() == language.lower():
                        filtered_docs.append(doc)
                        filtered_scores.append(score)
                documents, scores = filtered_docs, filtered_scores

            learning_insights = []
            if include_learning:
                learning_insights = self.get_relevant_learning_insights(query_text)

            return RAGContext(
                similar_diffs=documents,
                similarity_scores=scores,
                query_embedding=query_embedding,
                learning_insights=learning_insights,
            )
        except Exception as e:
            self.logger.error(f"Failed to get similar contexts: {e}")
            return RAGContext([], [])

    def enhance_prompt_with_context(
        self, base_prompt: str, context: RAGContext, max_context: int = 3
    ) -> str:
        if not context.similar_diffs:
            return base_prompt

        context_section = f"""
## Similar Historical Changes (for reference):
{context.get_formatted_context(max_context)}

## Current Analysis:
"""
        return context_section + base_prompt

    def get_database_stats(self) -> Dict[str, Any]:
        return self.vector_db.get_stats()

    def add_learning_insight(
        self,
        insight_text: str,
        pr_id: str,
        agreement_score: float = 0.0,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        insight_data = {
            "insight": insight_text,
            "pr_id": pr_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "agreement_score": agreement_score,
            **(metadata or {}),
        }
        return self.add_learning_insights_to_rag([insight_data])

    async def gen_search_query(
        self,
        pr_title: str,
        pr_description: str,
        changed_files: List[str] = None,
        language: str = None,
    ) -> str:
        context = (
            f"Title: {pr_title}\nDescription: {pr_description or 'No description'}"
        )
        if changed_files:
            context += f"\nFiles: {', '.join(changed_files[:5])}"
        if language:
            context += f"\nLanguage: {language}"

        search_prompt = f"""Analyze this PR and generate search terms to find relevant code review learning insights.

## PR Context:
{context}

## Task:
Generate 2-3 descriptive phrases that would help find learning insights relevant to this PR. Focus on:
- Technical domains (authentication, testing, performance, security, database, API, etc.)
- Code patterns and practices
- Common issue types
- Risk categories

Transform specific PR details into general technical concepts that learning insights would address.

Examples:
- "authentication middleware security validation" 
- "database connection resource management"
- "test coverage gap detection"
- "API error handling patterns"

Output only the search phrases, one per line, no explanations."""

        response, _ = await self.ai_handler.chat_completion(
            model="gpt-4o-mini",
            temperature=0.1,
            system="You are an expert at mapping specific code changes to general technical insight categories. Generate search terms that will find relevant learning patterns.",
            user=search_prompt,
        )

        search_terms = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if len(line) > 10:
                search_terms.append(line)

        search_query = " ".join(search_terms[:3])
        self.logger.info(f"Generated insight search query: {search_query}")
        return search_query

    async def get_relevant_learning_insights_with_context(
        self,
        pr_title: str,
        pr_description: str = None,
        changed_files: List[str] = None,
        language: str = None,
    ) -> List[Dict[str, Any]]:
        search_query = await self.gen_search_query(
            pr_title=pr_title,
            pr_description=pr_description,
            changed_files=changed_files,
            language=language,
        )

        return self.get_relevant_learning_insights(search_query, 5)

    def get_relevant_learning_insights(
        self,
        query_text: str,
        max_insights: int = 8,
    ) -> List[Dict[str, Any]]:
        query_embedding = self.vector_db._generate_embedding(query_text)
        if query_embedding is None:
            return []

        documents, scores = self.vector_db.search(
            query_embedding, k=max_insights
        )

        learning_insights = []
        for doc, score in zip(documents, scores):
            learning_insights.append(
                {
                    "insight": doc.get("insight", doc.get("diff_summary", "")),
                    "pr_id": doc.get("pr_id", "unknown"),
                    "timestamp": doc.get("timestamp", ""),
                    "agreement_score": doc.get("agreement_score", 0.0),
                    "similarity_score": score,
                }
            )
        return learning_insights

    def add_learning_insights_to_rag(self, insights: List[Dict[str, Any]]) -> bool:
        learning_docs = []
        for insight in insights:
            doc = {
                "pr_id": f"learning_{insight.get('pr_id', 'unknown')}_{hash(insight.get('insight', ''))}",
                "pr_url": "",
                "title": f"Learning Insight: {insight.get('insight', '')[:50]}",
                "diff_summary": f"Learning point: {insight.get('insight', '')}",
                "language": "learning",
                "changed_files": [],
                "author": "system",
                "created_at": insight.get("timestamp", ""),
            }
            learning_docs.append(doc)

        if learning_docs:
            return self.vector_db.add_documents(learning_docs)
        return False
