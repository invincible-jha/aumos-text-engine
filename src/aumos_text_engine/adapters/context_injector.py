"""Multi-document context assembly adapter.

Assembles relevant context from multiple source documents for retrieval-
augmented generation (RAG). Handles chunking, relevance ranking via
embeddings, token budget management, deduplication, and source attribution.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

import structlog
from aumos_common.logging import get_logger

logger: structlog.BoundLogger = get_logger(__name__)

# Default chunking parameters
_DEFAULT_CHUNK_SIZE = 512       # tokens (approximate chars / 4)
_DEFAULT_CHUNK_OVERLAP = 64     # tokens overlap between chunks
_DEFAULT_TOP_K_CHUNKS = 5       # number of top chunks to inject
_DEFAULT_TOKEN_BUDGET = 2048    # maximum context tokens

# Characters per token approximation
_CHARS_PER_TOKEN = 4


@dataclass
class DocumentChunk:
    """A text chunk from a source document.

    Attributes:
        chunk_id: Content hash of the chunk for deduplication.
        source_id: Identifier of the source document.
        source_name: Human-readable name of the source document.
        text: Chunk text content.
        chunk_index: Position in the source document.
        estimated_tokens: Estimated token count.
        relevance_score: Similarity score to the query (set after ranking).
    """

    chunk_id: str
    source_id: str
    source_name: str
    text: str
    chunk_index: int
    estimated_tokens: int
    relevance_score: float = 0.0

    @classmethod
    def from_text(
        cls,
        text: str,
        source_id: str,
        source_name: str,
        chunk_index: int,
    ) -> "DocumentChunk":
        """Create a DocumentChunk from text with auto-computed metadata.

        Args:
            text: Chunk text content.
            source_id: Source document identifier.
            source_name: Source document name.
            chunk_index: Position in the source document.

        Returns:
            DocumentChunk instance.
        """
        chunk_id = hashlib.sha256(f"{source_id}:{chunk_index}:{text[:64]}".encode()).hexdigest()[:16]
        estimated_tokens = max(1, len(text) // _CHARS_PER_TOKEN)

        return cls(
            chunk_id=chunk_id,
            source_id=source_id,
            source_name=source_name,
            text=text,
            chunk_index=chunk_index,
            estimated_tokens=estimated_tokens,
        )


@dataclass
class AssembledContext:
    """Result of context assembly for a generation request.

    Attributes:
        context_text: Final assembled context string for injection into prompt.
        source_attributions: List of source documents referenced.
        total_tokens: Estimated total token count.
        chunks_used: Number of chunks included.
        chunks_available: Total chunks considered before budget filtering.
    """

    context_text: str
    source_attributions: list[dict[str, Any]]
    total_tokens: int
    chunks_used: int
    chunks_available: int


class ContextInjector:
    """Assembles relevant multi-document context for LLM generation.

    Pipeline:
    1. Chunk source documents with configurable size and overlap
    2. Embed chunks and query using sentence-transformers
    3. Rank chunks by cosine similarity to the query
    4. Deduplicate chunks with near-identical content
    5. Select top-k chunks within token budget
    6. Assemble into a formatted context block with source attribution

    Attributes:
        _chunk_size_tokens: Target chunk size in tokens.
        _chunk_overlap_tokens: Token overlap between consecutive chunks.
        _top_k_chunks: Maximum chunks to include in context.
        _token_budget: Maximum total tokens for assembled context.
        _embedding_model: sentence-transformers model for ranking.
        _model_initialized: Whether the embedding model is loaded.
        _log: Structured logger.
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        chunk_size_tokens: int = _DEFAULT_CHUNK_SIZE,
        chunk_overlap_tokens: int = _DEFAULT_CHUNK_OVERLAP,
        top_k_chunks: int = _DEFAULT_TOP_K_CHUNKS,
        token_budget: int = _DEFAULT_TOKEN_BUDGET,
    ) -> None:
        """Initialize the ContextInjector.

        Args:
            embedding_model_name: sentence-transformers model for relevance ranking.
            chunk_size_tokens: Target chunk size in approximate tokens.
            chunk_overlap_tokens: Overlap between consecutive chunks in tokens.
            top_k_chunks: Maximum number of chunks to include in assembled context.
            token_budget: Hard limit on total context tokens.
        """
        self._embedding_model_name = embedding_model_name
        self._chunk_size_chars = chunk_size_tokens * _CHARS_PER_TOKEN
        self._chunk_overlap_chars = chunk_overlap_tokens * _CHARS_PER_TOKEN
        self._top_k_chunks = top_k_chunks
        self._token_budget = token_budget
        self._embedding_model: Any = None
        self._model_initialized = False
        self._log: structlog.BoundLogger = get_logger(__name__)

    async def initialize(self) -> None:
        """Load the sentence-transformers embedding model.

        Args:
            None

        Returns:
            None
        """
        if self._model_initialized:
            return

        self._log.info("loading embedding model for context ranking", model=self._embedding_model_name)
        try:
            from sentence_transformers import SentenceTransformer

            loop = asyncio.get_running_loop()
            self._embedding_model = await loop.run_in_executor(
                None,
                SentenceTransformer,
                self._embedding_model_name,
            )
            self._model_initialized = True
            self._log.info("context ranking model loaded")
        except ImportError:
            self._log.warning(
                "sentence-transformers not installed — falling back to keyword ranking",
            )

    async def chunk_document(
        self,
        text: str,
        source_id: str,
        source_name: str,
    ) -> list[DocumentChunk]:
        """Split a document into overlapping chunks.

        Uses sentence-boundary-aware splitting to avoid cutting mid-sentence
        where possible, then falls back to character-level splitting.

        Args:
            text: Full document text.
            source_id: Document identifier.
            source_name: Human-readable document name.

        Returns:
            List of DocumentChunk objects.
        """
        if not text.strip():
            return []

        loop = asyncio.get_running_loop()
        chunks = await loop.run_in_executor(
            None,
            self._split_text,
            text,
            source_id,
            source_name,
        )

        self._log.debug(
            "document chunked",
            source_id=source_id,
            chunk_count=len(chunks),
        )
        return chunks

    def _split_text(
        self,
        text: str,
        source_id: str,
        source_name: str,
    ) -> list[DocumentChunk]:
        """Synchronous text splitting with sentence-boundary awareness.

        Args:
            text: Document text to split.
            source_id: Source document identifier.
            source_name: Source document name.

        Returns:
            List of DocumentChunk instances.
        """
        # Split into sentences first
        sentence_pattern = re.compile(r"(?<=[.!?])\s+")
        sentences = sentence_pattern.split(text.strip())

        chunks: list[DocumentChunk] = []
        current_chunk: list[str] = []
        current_len = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            if current_len + sentence_len > self._chunk_size_chars and current_chunk:
                # Emit current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(DocumentChunk.from_text(chunk_text, source_id, source_name, chunk_index))
                chunk_index += 1

                # Overlap: keep last few sentences
                overlap_chars = 0
                overlap_sentences: list[str] = []
                for s in reversed(current_chunk):
                    if overlap_chars + len(s) > self._chunk_overlap_chars:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_chars += len(s)

                current_chunk = overlap_sentences
                current_len = overlap_chars

            current_chunk.append(sentence)
            current_len += sentence_len

        # Emit final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(DocumentChunk.from_text(chunk_text, source_id, source_name, chunk_index))

        return chunks

    async def rank_chunks(
        self,
        query: str,
        chunks: list[DocumentChunk],
    ) -> list[DocumentChunk]:
        """Rank chunks by relevance to the query.

        Uses embedding cosine similarity when the model is available,
        otherwise falls back to keyword overlap scoring.

        Args:
            query: The generation query or topic to rank against.
            chunks: Chunks to rank.

        Returns:
            Chunks sorted by relevance_score descending.
        """
        if not chunks:
            return []

        if self._model_initialized and self._embedding_model is not None:
            ranked = await self._rank_with_embeddings(query, chunks)
        else:
            ranked = self._rank_with_keywords(query, chunks)

        return sorted(ranked, key=lambda c: c.relevance_score, reverse=True)

    async def _rank_with_embeddings(
        self,
        query: str,
        chunks: list[DocumentChunk],
    ) -> list[DocumentChunk]:
        """Rank chunks using sentence-transformer cosine similarity.

        Args:
            query: Query text.
            chunks: Chunks to rank.

        Returns:
            Chunks with relevance_score set.
        """
        loop = asyncio.get_running_loop()

        def _compute() -> list[DocumentChunk]:
            import numpy as np

            texts = [query] + [c.text for c in chunks]
            embeddings = self._embedding_model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            query_embedding = embeddings[0]
            chunk_embeddings = embeddings[1:]

            for i, chunk in enumerate(chunks):
                similarity = float(np.dot(query_embedding, chunk_embeddings[i]))
                chunk.relevance_score = max(0.0, similarity)

            return chunks

        return await loop.run_in_executor(None, _compute)

    def _rank_with_keywords(
        self,
        query: str,
        chunks: list[DocumentChunk],
    ) -> list[DocumentChunk]:
        """Rank chunks by keyword overlap with the query.

        Args:
            query: Query text.
            chunks: Chunks to rank.

        Returns:
            Chunks with relevance_score set based on Jaccard overlap.
        """
        query_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", query.lower()))

        for chunk in chunks:
            chunk_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", chunk.text.lower()))
            if not query_words or not chunk_words:
                chunk.relevance_score = 0.0
                continue
            intersection = len(query_words & chunk_words)
            union = len(query_words | chunk_words)
            chunk.relevance_score = intersection / union if union > 0 else 0.0

        return chunks

    def deduplicate_chunks(
        self,
        chunks: list[DocumentChunk],
        similarity_threshold: float = 0.85,
    ) -> list[DocumentChunk]:
        """Remove near-duplicate chunks based on chunk_id and text similarity.

        Args:
            chunks: Ranked list of chunks to deduplicate.
            similarity_threshold: Jaccard similarity above which chunks are considered duplicates.

        Returns:
            Deduplicated list of chunks, preserving ranking order.
        """
        seen_ids: set[str] = set()
        seen_texts: list[str] = []
        unique_chunks: list[DocumentChunk] = []

        for chunk in chunks:
            if chunk.chunk_id in seen_ids:
                continue

            # Check similarity against already-accepted chunks
            is_duplicate = False
            chunk_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", chunk.text.lower()))

            for seen_text in seen_texts:
                seen_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", seen_text.lower()))
                if not chunk_words or not seen_words:
                    continue
                jaccard = len(chunk_words & seen_words) / len(chunk_words | seen_words)
                if jaccard >= similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                seen_ids.add(chunk.chunk_id)
                seen_texts.append(chunk.text)
                unique_chunks.append(chunk)

        return unique_chunks

    async def assemble_context(
        self,
        query: str,
        documents: list[dict[str, str]],
        token_budget: int | None = None,
    ) -> AssembledContext:
        """Assemble relevant context from multiple documents for a query.

        Full pipeline: chunk -> rank -> deduplicate -> budget-filter -> format.

        Args:
            query: Generation query or topic hint.
            documents: List of dicts with keys: id, name, text.
            token_budget: Override default token budget.

        Returns:
            AssembledContext with formatted context string and attributions.
        """
        effective_budget = token_budget or self._token_budget
        log = self._log.bind(query_preview=query[:80], doc_count=len(documents))

        # Step 1: Chunk all documents
        all_chunks: list[DocumentChunk] = []
        for doc in documents:
            doc_chunks = await self.chunk_document(
                text=doc.get("text", ""),
                source_id=doc.get("id", "unknown"),
                source_name=doc.get("name", "Unknown Document"),
            )
            all_chunks.extend(doc_chunks)

        if not all_chunks:
            return AssembledContext(
                context_text="",
                source_attributions=[],
                total_tokens=0,
                chunks_used=0,
                chunks_available=0,
            )

        log.debug("chunks created", chunk_count=len(all_chunks))

        # Step 2: Rank by relevance
        ranked_chunks = await self.rank_chunks(query, all_chunks)

        # Step 3: Deduplicate
        unique_chunks = self.deduplicate_chunks(ranked_chunks)

        # Step 4: Apply token budget — greedy selection
        selected_chunks: list[DocumentChunk] = []
        total_tokens = 0
        source_ids_used: set[str] = set()

        for chunk in unique_chunks[: self._top_k_chunks * 2]:  # Consider 2x top_k before budget cut
            if len(selected_chunks) >= self._top_k_chunks:
                break
            if total_tokens + chunk.estimated_tokens > effective_budget:
                continue
            selected_chunks.append(chunk)
            total_tokens += chunk.estimated_tokens
            source_ids_used.add(chunk.source_id)

        # Step 5: Format context with source attribution
        context_parts: list[str] = []
        for i, chunk in enumerate(selected_chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk.source_name}]\n{chunk.text}"
            )

        context_text = "\n\n---\n\n".join(context_parts)

        # Build source attribution list
        source_attributions: list[dict[str, Any]] = []
        seen_source_ids: set[str] = set()
        for chunk in selected_chunks:
            if chunk.source_id not in seen_source_ids:
                seen_source_ids.add(chunk.source_id)
                source_attributions.append({
                    "source_id": chunk.source_id,
                    "source_name": chunk.source_name,
                    "relevance_score": round(chunk.relevance_score, 4),
                })

        log.info(
            "context assembled",
            chunks_used=len(selected_chunks),
            total_tokens=total_tokens,
            sources=len(source_attributions),
        )

        return AssembledContext(
            context_text=context_text,
            source_attributions=source_attributions,
            total_tokens=total_tokens,
            chunks_used=len(selected_chunks),
            chunks_available=len(all_chunks),
        )
