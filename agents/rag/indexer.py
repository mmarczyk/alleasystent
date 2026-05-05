from __future__ import annotations

"""
Document indexer — loads and indexes content into the vector store.

Supported sources:
  - Plain text / Markdown files
  - JSON product catalog dumps
  - Allegro offer data fetched live
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from agents.rag.retriever import Document, build_retriever

logger = logging.getLogger(__name__)


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def _make_id(content: str, source: str) -> str:
    return hashlib.md5(f"{source}::{content[:100]}".encode()).hexdigest()


class DocumentIndexer:
    def __init__(self):
        self._retriever = build_retriever()

    async def index_text_file(self, path: str | Path, metadata: dict | None = None) -> int:
        """Index a plain text / Markdown file."""
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        base_meta = {"source": str(path), "type": "text", **(metadata or {})}
        docs = [
            Document(
                doc_id=_make_id(chunk, str(path)),
                content=chunk,
                metadata={**base_meta, "chunk": i},
            )
            for i, chunk in enumerate(_chunk_text(text))
        ]
        await self._retriever.add_documents(docs)
        logger.info("Indexed %d chunks from %s", len(docs), path)
        return len(docs)

    async def index_json_catalog(self, path: str | Path) -> int:
        """
        Index a JSON product catalog.

        Expected format: list of objects with at least {"name": ..., "description": ...}.
        """
        path = Path(path)
        items: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
        docs = []
        for item in items:
            content = self._product_to_text(item)
            docs.append(Document(
                doc_id=_make_id(content, str(path)),
                content=content,
                metadata={"source": str(path), "type": "product", "item_id": str(item.get("id", ""))},
            ))
        await self._retriever.add_documents(docs)
        logger.info("Indexed %d products from %s", len(docs), path)
        return len(docs)

    async def index_allegro_offers(self, offers: list[dict[str, Any]]) -> int:
        """Index live Allegro offers fetched from the API."""
        docs = []
        for offer in offers:
            content = self._offer_to_text(offer)
            docs.append(Document(
                doc_id=f"allegro_offer_{offer.get('id', '')}",
                content=content,
                metadata={
                    "source": "allegro_api",
                    "type": "allegro_offer",
                    "offer_id": offer.get("id", ""),
                },
            ))
        await self._retriever.add_documents(docs)
        logger.info("Indexed %d Allegro offers", len(docs))
        return len(docs)

    async def index_faq(self, faq_items: list[dict[str, str]]) -> int:
        """
        Index FAQ items.

        Expected format: [{"question": ..., "answer": ...}, ...]
        """
        docs = []
        for item in faq_items:
            content = f"Q: {item['question']}\nA: {item['answer']}"
            docs.append(Document(
                doc_id=_make_id(content, "faq"),
                content=content,
                metadata={"source": "faq", "type": "faq"},
            ))
        await self._retriever.add_documents(docs)
        logger.info("Indexed %d FAQ items", len(docs))
        return len(docs)

    @staticmethod
    def _product_to_text(product: dict[str, Any]) -> str:
        parts = []
        if name := product.get("name"):
            parts.append(f"Product: {name}")
        if desc := product.get("description"):
            parts.append(f"Description: {desc}")
        if price := product.get("price"):
            parts.append(f"Price: {price}")
        if category := product.get("category"):
            parts.append(f"Category: {category}")
        if stock := product.get("stock"):
            parts.append(f"Stock: {stock}")
        return "\n".join(parts)

    @staticmethod
    def _offer_to_text(offer: dict[str, Any]) -> str:
        parts = [f"Offer: {offer.get('name', '')}"]
        price = offer.get("sellingMode", {}).get("price", {})
        if price:
            parts.append(f"Price: {price.get('amount')} {price.get('currency', 'PLN')}")
        stock = offer.get("stock", {})
        if stock:
            parts.append(f"Stock available: {stock.get('available', 0)}")
        if params := offer.get("parameters", []):
            for p in params[:5]:
                parts.append(f"{p.get('name')}: {', '.join(v.get('value', '') for v in p.get('values', []))}")
        return "\n".join(parts)
