"""Document chunker adapter for streaming large document processing.

Splits documents into semantic chunks (paragraphs, sentences, or fixed size)
with configurable overlap to maintain entity continuity across chunk boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import structlog
from aumos_common.logging import get_logger


@dataclass
class DocumentChunk:
    """A single chunk of a larger document.

    Attributes:
        chunk_index: Zero-based position of this chunk in the document.
        text: The chunk text content (may include overlap region).
        start_char: Character offset of chunk start in original document.
        end_char: Character offset of chunk end in original document.
        overlap_start: Characters overlapping from previous chunk (0 if first).
        overlap_end: Characters overlapping into next chunk (0 if last).
    """

    chunk_index: int
    text: str
    start_char: int
    end_char: int
    overlap_start: int
    overlap_end: int


class DocumentChunker:
    """Splits large documents into semantic chunks for streaming processing.

    Supports three chunking strategies:
    - paragraph: Split on blank lines (best for prose documents).
    - sentence: Split on sentence boundaries (best for structured text).
    - fixed: Split on exact character count (guaranteed memory bound).

    Overlap between chunks ensures that entities spanning chunk boundaries
    are detected in at least one chunk.

    Args:
        chunk_size: Maximum characters per chunk (default: 10_000 ~ 5 pages).
        overlap: Overlap characters between adjacent chunks (default: 500).
        strategy: Chunking strategy ("paragraph" | "sentence" | "fixed").
    """

    def __init__(
        self,
        chunk_size: int = 10_000,
        overlap: int = 500,
        strategy: Literal["paragraph", "sentence", "fixed"] = "paragraph",
    ) -> None:
        """Initialize the document chunker.

        Args:
            chunk_size: Maximum characters per chunk.
            overlap: Characters to overlap between adjacent chunks.
            strategy: Chunking strategy to use.
        """
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._strategy = strategy
        self._log: structlog.BoundLogger = get_logger(__name__)

    def chunk(self, text: str) -> list[DocumentChunk]:
        """Split text into chunks with positional metadata.

        Args:
            text: Full document text to split.

        Returns:
            List of DocumentChunk objects in document order.
        """
        if not text:
            return []

        if len(text) <= self._chunk_size:
            return [
                DocumentChunk(
                    chunk_index=0,
                    text=text,
                    start_char=0,
                    end_char=len(text),
                    overlap_start=0,
                    overlap_end=0,
                )
            ]

        if self._strategy == "paragraph":
            segments = self._split_paragraphs(text)
        elif self._strategy == "sentence":
            segments = self._split_sentences(text)
        else:
            segments = self._split_fixed(text)

        chunks = self._merge_segments_into_chunks(text, segments)
        self._log.debug(
            "document chunked",
            strategy=self._strategy,
            total_chars=len(text),
            num_chunks=len(chunks),
        )
        return chunks

    def _split_paragraphs(self, text: str) -> list[tuple[int, int]]:
        """Split text on paragraph boundaries (blank lines).

        Args:
            text: Input text to split.

        Returns:
            List of (start, end) character offset tuples for each paragraph.
        """
        segments: list[tuple[int, int]] = []
        current_start = 0
        for match in re.finditer(r"\n\s*\n", text):
            end = match.end()
            if end > current_start:
                segments.append((current_start, end))
            current_start = end
        if current_start < len(text):
            segments.append((current_start, len(text)))
        return segments

    def _split_sentences(self, text: str) -> list[tuple[int, int]]:
        """Split text on sentence-ending punctuation.

        Args:
            text: Input text to split.

        Returns:
            List of (start, end) character offset tuples for each sentence.
        """
        segments: list[tuple[int, int]] = []
        current_start = 0
        for match in re.finditer(r"[.!?]+\s+", text):
            end = match.end()
            if end > current_start:
                segments.append((current_start, end))
            current_start = end
        if current_start < len(text):
            segments.append((current_start, len(text)))
        return segments

    def _split_fixed(self, text: str) -> list[tuple[int, int]]:
        """Split text into fixed-size segments.

        Args:
            text: Input text to split.

        Returns:
            List of (start, end) character offset tuples.
        """
        segments: list[tuple[int, int]] = []
        step = max(1, self._chunk_size - self._overlap)
        pos = 0
        while pos < len(text):
            end = min(pos + self._chunk_size, len(text))
            segments.append((pos, end))
            pos += step
        return segments

    def _merge_segments_into_chunks(
        self,
        text: str,
        segments: list[tuple[int, int]],
    ) -> list[DocumentChunk]:
        """Merge small segments into chunks that respect chunk_size limit.

        Args:
            text: Original full document text.
            segments: List of (start, end) offset tuples from splitting.

        Returns:
            List of DocumentChunk with overlap applied.
        """
        chunks: list[DocumentChunk] = []
        chunk_index = 0
        seg_idx = 0
        n = len(segments)

        while seg_idx < n:
            chunk_start = segments[seg_idx][0]
            chunk_end = chunk_start

            # Accumulate segments until we hit chunk_size
            while seg_idx < n:
                seg_start, seg_end = segments[seg_idx]
                if chunk_end > chunk_start and (seg_end - chunk_start) > self._chunk_size:
                    break
                chunk_end = seg_end
                seg_idx += 1

            # Apply overlap from previous chunk
            actual_start = max(0, chunk_start - self._overlap) if chunk_index > 0 else chunk_start
            # Apply overlap into next chunk
            actual_end = min(len(text), chunk_end + self._overlap) if seg_idx < n else chunk_end

            overlap_start = chunk_start - actual_start
            overlap_end = actual_end - chunk_end

            chunks.append(
                DocumentChunk(
                    chunk_index=chunk_index,
                    text=text[actual_start:actual_end],
                    start_char=actual_start,
                    end_char=actual_end,
                    overlap_start=overlap_start,
                    overlap_end=overlap_end,
                )
            )
            chunk_index += 1

        return chunks
