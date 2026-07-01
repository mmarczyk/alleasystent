"""Unit tests for agents/rag/indexer.py pure functions."""
from __future__ import annotations

import hashlib

import pytest


class TestChunkText:
    def _chunk(self, text, chunk_size=800, overlap=100):
        from agents.rag.indexer import _chunk_text
        return _chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    def test_short_text_returns_one_chunk(self):
        text = "Hello world"
        chunks = self._chunk(text)
        assert chunks == [text]

    def test_exactly_chunk_size_returns_one_chunk(self):
        text = "a" * 800
        chunks = self._chunk(text, chunk_size=800)
        assert len(chunks) == 1

    def test_long_text_splits(self):
        text = "a" * 1600
        chunks = self._chunk(text, chunk_size=800, overlap=100)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 800

    def test_overlap(self):
        text = "a" * 900
        chunks = self._chunk(text, chunk_size=800, overlap=100)
        # Second chunk starts at 800-100=700
        assert len(chunks) == 2
        assert chunks[1] == text[700:]

    def test_empty_text(self):
        chunks = self._chunk("", chunk_size=800)
        assert chunks == [""]

    def test_chunks_cover_full_text(self):
        text = "x" * 2000
        chunks = self._chunk(text, chunk_size=800, overlap=0)
        # With zero overlap all chars covered
        reconstructed = "".join(chunks)
        assert reconstructed == text


class TestMakeId:
    def _make_id(self, content, source):
        from agents.rag.indexer import _make_id
        return _make_id(content, source)

    def test_returns_hex_string(self):
        result = self._make_id("content", "source")
        assert isinstance(result, str)
        assert len(result) == 32  # MD5 hex digest

    def test_same_inputs_same_id(self):
        a = self._make_id("hello", "file.txt")
        b = self._make_id("hello", "file.txt")
        assert a == b

    def test_different_source_different_id(self):
        a = self._make_id("hello", "file1.txt")
        b = self._make_id("hello", "file2.txt")
        assert a != b

    def test_different_content_different_id(self):
        a = self._make_id("hello", "file.txt")
        b = self._make_id("world", "file.txt")
        assert a != b

    def test_uses_first_100_chars_of_content(self):
        # Two contents that differ only after position 100 should get same ID
        content_base = "a" * 100
        a = self._make_id(content_base + "extra1", "src")
        b = self._make_id(content_base + "extra2", "src")
        assert a == b


class TestProductToText:
    def _to_text(self, product):
        from agents.rag.indexer import DocumentIndexer
        return DocumentIndexer._product_to_text(product)

    def test_all_fields(self):
        product = {
            "name": "Widget",
            "description": "A great widget",
            "price": "99.99",
            "category": "Electronics",
            "stock": "10",
        }
        text = self._to_text(product)
        assert "Widget" in text
        assert "great widget" in text
        assert "99.99" in text
        assert "Electronics" in text
        assert "10" in text

    def test_minimal_fields(self):
        product = {"name": "Basic"}
        text = self._to_text(product)
        assert "Basic" in text

    def test_empty_product(self):
        text = self._to_text({})
        assert text == ""


class TestOfferToText:
    def _to_text(self, offer):
        from agents.rag.indexer import DocumentIndexer
        return DocumentIndexer._offer_to_text(offer)

    def test_full_offer(self):
        offer = {
            "name": "SuperWidget",
            "sellingMode": {"price": {"amount": "49.99", "currency": "PLN"}},
            "stock": {"available": 5},
        }
        text = self._to_text(offer)
        assert "SuperWidget" in text
        assert "49.99" in text
        assert "5" in text

    def test_minimal_offer(self):
        text = self._to_text({"name": "Gadget"})
        assert "Gadget" in text

    def test_parameters_included(self):
        offer = {
            "name": "Item",
            "parameters": [
                {"name": "Color", "values": [{"value": "Red"}]}
            ]
        }
        text = self._to_text(offer)
        assert "Color" in text
        assert "Red" in text
