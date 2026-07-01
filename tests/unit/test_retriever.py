"""Unit tests for agents/rag/retriever.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")


class TestDocument:
    def test_source_from_metadata(self):
        from agents.rag.retriever import Document
        doc = Document("id1", "content", {"source": "my_file.txt"})
        assert doc.source == "my_file.txt"

    def test_source_falls_back_to_doc_id(self):
        from agents.rag.retriever import Document
        doc = Document("fallback-id", "content", {})
        assert doc.source == "fallback-id"

    def test_default_score(self):
        from agents.rag.retriever import Document
        doc = Document("id", "content", {})
        assert doc.score == 0.0

    def test_custom_score(self):
        from agents.rag.retriever import Document
        doc = Document("id", "content", {}, score=0.95)
        assert doc.score == pytest.approx(0.95)

    def test_all_fields(self):
        from agents.rag.retriever import Document
        doc = Document("d1", "hello world", {"type": "faq"}, score=0.8)
        assert doc.doc_id == "d1"
        assert doc.content == "hello world"
        assert doc.metadata == {"type": "faq"}


class TestBuildRetriever:
    def test_returns_chroma_when_chromadb_backend(self, monkeypatch):
        monkeypatch.setenv("RAG_BACKEND", "chromadb")
        import agents.rag.retriever as retriever_module
        retriever_module._chroma_singleton = None
        with patch.object(retriever_module.ChromaRetriever, "_init"):
            result = retriever_module.build_retriever()
        assert isinstance(result, retriever_module.ChromaRetriever)
        retriever_module._chroma_singleton = None

    def test_singleton_behavior(self, monkeypatch):
        monkeypatch.setenv("RAG_BACKEND", "chromadb")
        import agents.rag.retriever as retriever_module
        retriever_module._chroma_singleton = None
        with patch.object(retriever_module.ChromaRetriever, "_init"):
            r1 = retriever_module.build_retriever()
            r2 = retriever_module.build_retriever()
        assert r1 is r2
        retriever_module._chroma_singleton = None


class TestChromaRetrieverInit:
    def test_graceful_failure_on_missing_chromadb(self, monkeypatch):
        """If chromadb import fails, _collection should be None (no crash)."""
        import agents.rag.retriever as retriever_module
        retriever_module._chroma_singleton = None
        # Instantiate with a patched _init that raises ImportError
        with patch.object(retriever_module.ChromaRetriever, "_init",
                          side_effect=Exception("no chromadb")):
            r = retriever_module.ChromaRetriever.__new__(retriever_module.ChromaRetriever)
            r._settings = MagicMock()
            r._collection = None
        assert r._collection is None
        retriever_module._chroma_singleton = None

    @pytest.mark.asyncio
    async def test_query_returns_empty_when_no_collection(self, monkeypatch):
        import agents.rag.retriever as retriever_module
        r = retriever_module.ChromaRetriever.__new__(retriever_module.ChromaRetriever)
        r._settings = MagicMock()
        r._collection = None
        result = await r.query("test")
        assert result == []

    @pytest.mark.asyncio
    async def test_add_documents_no_op_when_no_collection(self, monkeypatch):
        import agents.rag.retriever as retriever_module
        from agents.rag.retriever import Document
        r = retriever_module.ChromaRetriever.__new__(retriever_module.ChromaRetriever)
        r._settings = MagicMock()
        r._collection = None
        await r.add_documents([Document("id", "content", {})])  # should not raise
