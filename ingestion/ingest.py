"""
ingestion/ingest.py
-------------------
Document ingestion script: process enterprise documents and store them in the Qdrant vector database.
Supports PDF, TXT, and DOCX (requires langchain-community).

Usage:
    python -m ingestion.ingest \
        --use_case kb_qa \
        --tenant_id acme_corp \
        --source_dir ./docs/acme_hr_policies
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    PointStruct, Filter, FieldCondition, MatchValue
)

from ingestion.chunker import AdaptiveChunker
from ingestion.embedder import DomainEmbedder
from config.use_cases import get_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DocumentIngester:
    """
    Document ingester: read files → chunk → embed → write to Qdrant
    Supports hierarchical chunking (child chunks stored as vectors, parent chunks in payloads)
    """

    def __init__(
        self,
        qdrant_url: str,
        openai_api_key: str,
        embedding_model: str = "text-embedding-3-large",
    ):
        self.qdrant = QdrantClient(url=qdrant_url)
        self.embedder = DomainEmbedder(
            api_key=openai_api_key,
            model=embedding_model,
        )

    def _ensure_collection(self, collection_name: str, vector_size: int = 1536):
        """Ensure the Qdrant collection exists, creating it if necessary."""
        existing = {c.name for c in self.qdrant.get_collections().collections}
        if collection_name not in existing:
            self.qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created collection: {collection_name}")

    def _read_file(self, filepath: Path) -> str:
        """Read file contents (supports txt/md; can be extended to other formats)."""
        suffix = filepath.suffix.lower()
        if suffix in [".txt", ".md"]:
            return filepath.read_text(encoding="utf-8")
        elif suffix == ".pdf":
            # Requires: pip install pypdf
            from pypdf import PdfReader
            reader = PdfReader(str(filepath))
            return "\n\n".join(
                page.extract_text() for page in reader.pages
                if page.extract_text()
            )
        elif suffix in [".docx"]:
            # Requires: pip install python-docx
            from docx import Document as DocxDocument
            doc = DocxDocument(str(filepath))
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        else:
            logger.warning(f"Unsupported file type: {suffix}, skipping {filepath}")
            return ""

    async def ingest_directory(
        self,
        source_dir: str,
        use_case: str,
        tenant_id: str,
        batch_size: int = 50,
    ):
        """
        Ingest all documents in a directory in batches.
        use_case determines the chunking strategy and collection name.
        """
        config = get_config(use_case)
        collection_name = f"enterprise_{use_case}"
        self._ensure_collection(collection_name)

        chunker = AdaptiveChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        strategy = "hierarchical" if config.use_hierarchical_chunking else "sentence"

        source_path = Path(source_dir)
        files = list(source_path.rglob("*"))
        files = [f for f in files if f.is_file() and f.suffix in [".txt", ".md", ".pdf", ".docx"]]

        logger.info(f"Found {len(files)} files in {source_dir}")
        all_chunks = []

        for filepath in files:
            text = self._read_file(filepath)
            if not text.strip():
                continue
            metadata = {
                "source": str(filepath.relative_to(source_path)),
                "filename": filepath.name,
                "tenant_id": tenant_id,
                "use_case": use_case,
            }
            chunks = chunker.chunk(text, metadata, strategy=strategy)
            all_chunks.extend(chunks)
            logger.info(f"  {filepath.name}: {len(chunks)} chunks")

        logger.info(f"Total chunks to ingest: {len(all_chunks)}")

        # Batch embedding
        texts = [c.page_content for c in all_chunks]
        logger.info("Embedding chunks (batch mode)...")
        embeddings = await self.embedder.embed_documents_batch(texts, batch_size=100)

        # Write to Qdrant in batches
        points = []
        for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
            point = PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "text": chunk.page_content,
                    "parent_text": chunk.parent_text,      # Parent chunk (core of hierarchical chunking)
                    "parent_id": chunk.parent_id,
                    "source": chunk.metadata.get("source", ""),
                    "filename": chunk.metadata.get("filename", ""),
                    "tenant_id": tenant_id,
                    "chunk_id": chunk.chunk_id,
                }
            )
            points.append(point)

            # Upload in batches
            if len(points) >= batch_size:
                self.qdrant.upsert(collection_name=collection_name, points=points)
                logger.info(f"  Uploaded {i+1}/{len(all_chunks)} chunks")
                points = []

        if points:
            self.qdrant.upsert(collection_name=collection_name, points=points)

        logger.info(f"✅ Ingestion complete: {len(all_chunks)} chunks "
                    f"→ collection '{collection_name}'")


# ─────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(description="Enterprise RAG Document Ingester")
    parser.add_argument("--use_case", required=True,
                        choices=["kb_qa", "helpdesk", "compliance"])
    parser.add_argument("--tenant_id", required=True)
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--qdrant_url", default="http://localhost:6333")
    args = parser.parse_args()

    ingester = DocumentIngester(
        qdrant_url=args.qdrant_url,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    await ingester.ingest_directory(
        source_dir=args.source_dir,
        use_case=args.use_case,
        tenant_id=args.tenant_id,
    )


if __name__ == "__main__":
    asyncio.run(main())
