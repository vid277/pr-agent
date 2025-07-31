import json
import time
import openai
import lancedb
import pyarrow as pa
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pinecone import Pinecone, ServerlessSpec
from pr_agent.algo.ai_handlers.base_ai_handler import BaseAiHandler
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
                f"""
### Similar Change #{i+1} (similarity: {score:.3f})
**Title:** {diff_data.get('title', 'N/A')}
**Summary:** {diff_data.get('diff_summary', 'N/A')}
**Files:** {', '.join(diff_data.get('changed_files', [])[:3])}
**Language:** {diff_data.get('language', 'N/A')}
"""
            )

        if self.learning_insights:
            context_parts.append("\n### Learning Insights from Previous Analysis:")
            for insight in self.learning_insights[:3]:
                context_parts.append(f"• {insight.get('insight', 'N/A')}")

        return "\n".join(context_parts)


class VectorDatabase(ABC):
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        pass

    @abstractmethod
    def search(
        self, query_embedding: List[float], k: int = 5, document_type: str = None
    ) -> Tuple[List[Dict], List[float]]:
        pass

    @abstractmethod
    def delete_by_hash(self, embedding_hash: str) -> bool:
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        pass


class LanceDBHandler(VectorDatabase):
    def __init__(self, uri: str = None):
        self.uri = uri or get_settings().lancedb.uri
        self.db = lancedb.connect(self.uri)
        self.table_name = "pr_diffs"
        self.logger = get_logger()

        self.schema = None

        self._ensure_table_exists()

    def _ensure_table_exists(self):
        try:
            self.table = self.db.open_table(self.table_name)
        except FileNotFoundError:
            self.table = None
            self.logger.info(
                f"Will create LanceDB table on first data insertion: {self.table_name}"
            )
        except Exception as e:
            self.logger.error(f"Error checking for table: {e}")
            self.table = None

    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        try:
            data_to_insert = []

            for doc in documents:
                embedding = self._generate_embedding(doc.diff_summary or "empty")
                if embedding is None:
                    self.logger.warning(
                        f"Skipping document {doc.pr_id} - no embedding generated"
                    )
                    continue

                record = {
                    "id": str(doc.pr_id or "unknown"),
                    "pr_url": str(doc.pr_url or ""),
                    "title": str(doc.title or ""),
                    "diff_summary": str(doc.diff_summary or ""),
                    "language": str(doc.language or "unknown"),
                    "changed_files": doc.changed_files if doc.changed_files else [],
                    "author": str(doc.author or "unknown"),
                    "created_at": str(doc.created_at or "1970-01-01T00:00:00Z"),
                    "embedding_hash": str(doc.embedding_hash or ""),
                    "vector": embedding,
                }

                self.logger.debug(
                    f"Adding record for PR {record['id']}: {record['title'][:50]}..."
                )
                data_to_insert.append(record)

            if data_to_insert:
                if self.table is None:
                    self.table = self.db.create_table(self.table_name, data_to_insert)
                    self.logger.info(
                        f"Created LanceDB table with {len(data_to_insert)} documents"
                    )
                else:
                    self.table.add(data_to_insert)
                    self.logger.info(
                        f"Added {len(data_to_insert)} documents to LanceDB"
                    )
                return True

            return False

        except Exception as e:
            self.logger.error(f"Failed to add documents to LanceDB: {e}")
            return False

    def search(
        self, query_embedding: List[float], k: int = 5, document_type: str = None
    ) -> Tuple[List[Dict], List[float]]:
        try:
            search_query = self.table.search(query_embedding).limit(k)

            if document_type:
                search_query = search_query.where(f"language = '{document_type}'")

            results = search_query.to_pandas()

            documents = []
            scores = []

            for _, row in results.iterrows():
                doc = {
                    "pr_id": row["id"],
                    "pr_url": row["pr_url"],
                    "title": row["title"],
                    "diff_summary": row["diff_summary"],
                    "language": row["language"],
                    "changed_files": row["changed_files"],
                    "author": row["author"],
                    "created_at": row["created_at"],
                    "embedding_hash": row["embedding_hash"],
                }
                documents.append(doc)
                scores.append(float(row.get("_distance", 0.0)))

            return documents, scores

        except Exception as e:
            self.logger.error(f"Failed to search LanceDB: {e}")
            return [], []

    def delete_by_hash(self, embedding_hash: str) -> bool:
        try:
            self.table.delete(f"embedding_hash = '{embedding_hash}'")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete document: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        try:
            count = len(self.table.to_pandas())
            return {
                "total_documents": count,
                "table_name": self.table_name,
                "database_uri": self.uri,
            }
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {}

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


class PineconeHandler(VectorDatabase):
    def __init__(self):
        self.logger = get_logger()

        api_key = get_settings().pinecone.api_key

        if not api_key:
            raise ValueError("Pinecone API key must be configured")

        self.pc = Pinecone(api_key=api_key)

        self.index_name = "pr-agent-diffs"
        self.dimension = 1536

        if self.index_name not in [idx.name for idx in self.pc.list_indexes()]:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        self.index = self.pc.Index(self.index_name)

    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        try:
            vectors_to_upsert = []

            for doc in documents:
                embedding = self._generate_embedding(doc.diff_summary)
                if embedding is None:
                    continue

                metadata = {
                    "pr_url": doc.pr_url,
                    "title": doc.title,
                    "diff_summary": doc.diff_summary,
                    "language": doc.language,
                    "changed_files": json.dumps(doc.changed_files),
                    "author": doc.author,
                    "created_at": doc.created_at,
                    "embedding_hash": doc.embedding_hash,
                    "document_type": "pr_diff",
                }

                vectors_to_upsert.append((doc.embedding_hash, embedding, metadata))

            if vectors_to_upsert:
                self.index.upsert(vectors_to_upsert)
                self.logger.info(
                    f"Added {len(vectors_to_upsert)} documents to Pinecone"
                )
                return True

            return False

        except Exception as e:
            self.logger.error(f"Failed to add documents to Pinecone: {e}")
            return False

    def add_learning_insights(self, insights: List[Dict[str, Any]]) -> bool:
        try:
            vectors_to_upsert = []

            for insight in insights:
                insight_text = insight.get("insight", "")
                embedding = self._generate_embedding(insight_text)
                if embedding is None:
                    continue

                insight_id = (
                    f"learning_{insight.get('pr_id', 'unknown')}_{hash(insight_text)}"
                )
                metadata = {
                    "document_type": "learning_insight",
                    "insight": insight_text,
                    "pr_id": insight.get("pr_id", "unknown"),
                    "timestamp": insight.get("timestamp", ""),
                    "agreement_score": insight.get("agreement_score", 0.0),
                    "title": f"Learning: {insight_text[:50]}...",
                    "language": "learning",
                    "author": "system",
                    "pr_url": "",
                    "diff_summary": insight_text,
                    "changed_files": "[]",
                    "embedding_hash": insight_id,
                }

                vectors_to_upsert.append((insight_id, embedding, metadata))

            if vectors_to_upsert:
                self.index.upsert(vectors_to_upsert)
                self.logger.info(
                    f"Added {len(vectors_to_upsert)} learning insights to Pinecone"
                )
                return True

            return False

        except Exception as e:
            self.logger.error(f"Failed to add learning insights to Pinecone: {e}")
            return False

    def search(
        self, query_embedding: List[float], k: int = 5, document_type: str = None
    ) -> Tuple[List[Dict], List[float]]:
        try:
            filter_dict = {}
            if document_type:
                filter_dict["document_type"] = {"$eq": document_type}

            results = self.index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None,
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
            self.logger.error(f"Failed to search Pinecone: {e}")
            return [], []

    def delete_by_hash(self, embedding_hash: str) -> bool:
        try:
            self.index.delete(ids=[embedding_hash])
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete document: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_documents": stats.total_vector_count,
                "index_name": self.index_name,
                "dimension": self.dimension,
            }
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {}

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


class RAGHandler:
    def __init__(self, vector_db_type: str = "pinecone"):
        self.logger = get_logger()
        self.ai_handler = LiteLLMAIHandler()

        if vector_db_type.lower() == "pinecone":
            self.vector_db = PineconeHandler()
        elif vector_db_type.lower() == "lancedb":
            self.vector_db = LanceDBHandler()
        else:
            raise ValueError(
                f"Unsupported vector database type: {vector_db_type}. Use 'pinecone' or 'lancedb'"
            )

        self.logger.info(f"Initialized RAG handler with {vector_db_type}")

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
            query_embedding = self._generate_embedding(query_text)
            if query_embedding is None:
                return RAGContext([], [])

            documents, scores = self.vector_db.search(
                query_embedding, k, document_type="pr_diff"
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

    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        return self.vector_db._generate_embedding(text)

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

        success = self.add_learning_insights_to_rag([insight_data])
        if success:
            self.logger.info(
                f"Added learning insight for PR {pr_id}: {insight_text[:50]}..."
            )
        return success

    async def generate_insight_search_query(
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
        max_insights: int = 3,
    ) -> List[Dict[str, Any]]:
        search_query = await self.generate_insight_search_query(
            pr_title=pr_title,
            pr_description=pr_description,
            changed_files=changed_files,
            language=language,
        )

        return self.get_relevant_learning_insights(search_query, max_insights)

    def get_relevant_learning_insights(
        self, query_text: str, max_insights: int = 3
    ) -> List[Dict[str, Any]]:
        query_embedding = self._generate_embedding(query_text)
        if query_embedding is None:
            self.logger.warning(
                "Could not generate embedding for learning insights query"
            )
            return []

        documents, scores = self.vector_db.search(
            query_embedding, k=max_insights, document_type="learning_insight"
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
        if hasattr(self.vector_db, "add_learning_insights"):
            return self.vector_db.add_learning_insights(insights)

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
                "embedding_hash": f"learning_{hash(insight.get('insight', ''))}",
            }
            learning_docs.append(doc)

        if learning_docs:
            return self.vector_db.add_documents(learning_docs)
        return False


class HybridSearchRAG(RAGHandler):
    def __init__(self, vector_db_type: str = "lancedb"):
        super().__init__(vector_db_type)
        self.keyword_weight = 0.3
        self.semantic_weight = 0.7
        self.semantic_weight = 0.7
        self.semantic_weight = 0.7
