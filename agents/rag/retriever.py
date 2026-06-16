from __future__ import annotations

"""
Vector store retrieval layer.

Supports two backends selectable via EMBEDDING_BACKEND / RAG_BACKEND settings:
  - local:    sentence-transformers + ChromaDB  (dev / small deployments)
  - vertex_ai: Vertex AI Embeddings + Vertex AI Vector Search (GCP production)
"""

import logging
import os
from typing import Any

# Must be set before chromadb is imported, otherwise telemetry fires at import time
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_ANONYMIZED_TELEMETRY"] = "false"

from config.settings import get_settings

logger = logging.getLogger(__name__)


class Document:
    """Lightweight document container returned by the retriever."""

    def __init__(self, doc_id: str, content: str, metadata: dict[str, Any], score: float = 0.0):
        self.doc_id = doc_id
        self.content = content
        self.metadata = metadata
        self.score = score
        self.source: str = metadata.get("source", doc_id)


class BaseRetriever:
    async def query(self, text: str, top_k: int = 5) -> list[Document]:
        raise NotImplementedError

    async def add_documents(self, documents: list[Document]) -> None:
        raise NotImplementedError

    async def delete_document(self, doc_id: str) -> None:
        raise NotImplementedError


class ChromaRetriever(BaseRetriever):
    """Local ChromaDB retriever with sentence-transformers embeddings."""

    COLLECTION_NAME = "store_knowledge"

    def __init__(self):
        self._settings = get_settings()
        self._collection = None
        self._embedding_fn = None
        self._init()

    def _init(self) -> None:
        try:
            import chromadb
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

            client = chromadb.PersistentClient(
                path=self._settings.chromadb_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            self._embedding_fn = SentenceTransformerEmbeddingFunction(
                model_name=self._settings.embedding_model
            )
            self._collection = client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=self._embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "ChromaDB collection '%s' ready (%d docs)",
                self.COLLECTION_NAME,
                self._collection.count(),
            )
        except Exception as exc:
            logger.warning("ChromaDB init failed, RAG disabled: %s", exc)
            self._collection = None

    async def query(self, text: str, top_k: int = 5) -> list[Document]:
        if self._collection is None:
            return []

        import asyncio

        def _sync() -> list[Document] | None:
            count = self._collection.count()
            if count == 0:
                return []
            results = self._collection.query(
                query_texts=[text],
                n_results=min(top_k, max(1, count)),
                include=["documents", "metadatas", "distances"],
            )
            docs = []
            for doc_id, content, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                docs.append(Document(
                    doc_id=doc_id,
                    content=content,
                    metadata=meta or {},
                    score=1.0 - float(dist),
                ))
            return docs

        return await asyncio.to_thread(_sync)

    async def add_documents(self, documents: list[Document]) -> None:
        if self._collection is None:
            return

        import asyncio

        def _sync() -> None:
            self._collection.upsert(
                ids=[d.doc_id for d in documents],
                documents=[d.content for d in documents],
                metadatas=[d.metadata for d in documents],
            )
            logger.info("Added/updated %d documents in ChromaDB", len(documents))

        await asyncio.to_thread(_sync)

    async def delete_document(self, doc_id: str) -> None:
        if self._collection:
            import asyncio
            await asyncio.to_thread(self._collection.delete, ids=[doc_id])


class VertexAIRetriever(BaseRetriever):
    """
    Vertex AI Vector Search retriever.
    Requires google-cloud-aiplatform and the index endpoint deployed in GCP.
    """

    def __init__(self):
        self._settings = get_settings()
        self._endpoint = None
        self._embedding_client = None
        self._init()

    def _init(self) -> None:
        try:
            from google.cloud import aiplatform
            from vertexai.language_models import TextEmbeddingModel

            aiplatform.init(
                project=self._settings.gcp_project_id,
                location=self._settings.gcp_region,
            )
            self._endpoint = aiplatform.MatchingEngineIndexEndpoint(
                index_endpoint_name=self._settings.vertex_ai_index_endpoint
            )
            self._embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko-multilingual@001")
            logger.info("Vertex AI Vector Search endpoint ready")
        except ImportError as exc:
            logger.error("google-cloud-aiplatform not installed: %s", exc)
            raise

    async def _embed(self, text: str) -> list[float]:
        embeddings = self._embedding_model.get_embeddings([text])
        return embeddings[0].values

    async def query(self, text: str, top_k: int = 5) -> list[Document]:
        embedding = await self._embed(text)
        response = self._endpoint.find_neighbors(
            deployed_index_id=self._settings.vertex_ai_index_id,
            queries=[embedding],
            num_neighbors=top_k,
        )
        docs = []
        for neighbor in response[0]:
            docs.append(Document(
                doc_id=neighbor.id,
                content=neighbor.restricts[0].allow_tokens[0] if neighbor.restricts else "",
                metadata={},
                score=neighbor.distance,
            ))
        return docs

    async def add_documents(self, documents: list[Document]) -> None:
        logger.warning("Vertex AI Vector Search indexing must be done via batch upsert job")

    async def delete_document(self, doc_id: str) -> None:
        logger.warning("Vertex AI Vector Search deletion must be done via batch job")


_chroma_singleton: "ChromaRetriever | None" = None


def build_retriever() -> BaseRetriever:
    global _chroma_singleton
    settings = get_settings()
    if settings.rag_backend == "vertex_ai":
        return VertexAIRetriever()
    # Singleton: one PersistentClient per process avoids SQLite locking when the
    # indexer and RAGAgent both open the same ChromaDB directory simultaneously.
    if _chroma_singleton is None:
        _chroma_singleton = ChromaRetriever()
    return _chroma_singleton
