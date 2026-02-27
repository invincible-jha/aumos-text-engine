"""Text quality evaluation adapter.

Computes multi-dimensional quality metrics for generated synthetic text:
semantic similarity, perplexity estimation, coherence, fluency, and
domain-specific accuracy. Aggregates scores into a unified quality report.
"""

from __future__ import annotations

import asyncio
import math
import re
from typing import Any

import structlog
from aumos_common.logging import get_logger

from aumos_text_engine.api.schemas import QualityReport

logger: structlog.BoundLogger = get_logger(__name__)

# Quality thresholds
_SEMANTIC_SIMILARITY_THRESHOLD = 0.65
_COHERENCE_THRESHOLD = 0.60
_FLUENCY_THRESHOLD = 0.55
_DOMAIN_QUALITY_THRESHOLD = 0.60

# Sentence splitter pattern
_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")

# Common function words for coherence analysis
_FUNCTION_WORDS = frozenset({
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
    "but", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "can", "it", "its", "this", "that", "these",
    "those", "with", "by", "from", "as", "if", "then", "than", "so",
})

# Domain quality indicator terms
_DOMAIN_INDICATORS: dict[str, list[str]] = {
    "legal": [
        "whereas", "hereinafter", "notwithstanding", "pursuant", "thereto",
        "indemnify", "covenant", "jurisdiction", "arbitration", "plaintiff",
        "defendant", "liability", "damages", "breach", "remedy", "statute",
    ],
    "medical": [
        "diagnosis", "prognosis", "etiology", "pathophysiology", "therapeutic",
        "contraindicated", "dosage", "symptom", "bilateral", "acute", "chronic",
        "clinical", "patient", "treatment", "medication", "procedure",
    ],
    "financial": [
        "revenue", "ebitda", "amortization", "depreciation", "liquidity",
        "solvency", "leverage", "portfolio", "yield", "basis", "derivative",
        "hedge", "variance", "forecast", "fiscal", "compliance", "audit",
    ],
    "general": [],
}


class TextQualityEvaluator:
    """Multi-dimensional text quality evaluator for synthetic documents.

    Computes:
    - Semantic similarity via sentence-transformers cosine similarity
    - Perplexity estimation via bigram language model
    - Coherence scoring via entity consistency and topic drift analysis
    - Fluency scoring via readability metrics
    - Domain-specific quality via indicator term density
    - Aggregate weighted quality score

    Attributes:
        _embedding_model: sentence-transformers model for semantic similarity.
        _min_similarity_threshold: Minimum passing similarity score.
        _model_initialized: Whether the embedding model is loaded.
        _log: Structured logger.
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        min_similarity_threshold: float = _SEMANTIC_SIMILARITY_THRESHOLD,
    ) -> None:
        """Initialize the TextQualityEvaluator.

        Args:
            embedding_model_name: sentence-transformers model to use.
            min_similarity_threshold: Minimum semantic similarity for passing reports.
        """
        self._embedding_model_name = embedding_model_name
        self._min_similarity_threshold = min_similarity_threshold
        self._embedding_model: Any = None
        self._model_initialized = False
        self._log: structlog.BoundLogger = get_logger(__name__)

    async def initialize(self) -> None:
        """Load the sentence-transformers embedding model.

        Must be called once at startup. Model loading is CPU-bound and
        executed in the thread pool to avoid blocking the event loop.

        Args:
            None

        Returns:
            None
        """
        if self._model_initialized:
            return

        self._log.info("loading embedding model", model=self._embedding_model_name)

        try:
            from sentence_transformers import SentenceTransformer

            loop = asyncio.get_running_loop()
            self._embedding_model = await loop.run_in_executor(
                None,
                SentenceTransformer,
                self._embedding_model_name,
            )
            self._model_initialized = True
            self._log.info("embedding model loaded", model=self._embedding_model_name)
        except ImportError:
            self._log.warning(
                "sentence-transformers not installed — semantic similarity disabled",
                model=self._embedding_model_name,
            )

    async def validate(
        self,
        original_text: str,
        synthetic_text: str,
    ) -> QualityReport:
        """Compute quality metrics between original and synthetic text.

        Implements the QualityValidatorProtocol interface expected by
        SynthesisService. Delegates to evaluate() for full metric computation.

        Args:
            original_text: Source document (after PII replacement).
            synthetic_text: Generated synthetic document.

        Returns:
            QualityReport with semantic similarity score and pass/fail status.
        """
        return await self.evaluate(
            original_text=original_text,
            synthetic_text=synthetic_text,
            domain="general",
        )

    async def evaluate(
        self,
        original_text: str,
        synthetic_text: str,
        domain: str = "general",
    ) -> QualityReport:
        """Compute all quality metrics and return a comprehensive report.

        Args:
            original_text: Reference document text.
            synthetic_text: Generated text to evaluate.
            domain: Domain for domain-specific quality scoring.

        Returns:
            QualityReport with all quality dimensions and aggregate score.
        """
        log = self._log.bind(domain=domain, synthetic_length=len(synthetic_text))
        log.debug("evaluating text quality")

        if not synthetic_text.strip():
            return QualityReport(
                semantic_similarity=0.0,
                passes_threshold=False,
                details={"error": "empty synthetic text"},
            )

        # Run evaluations concurrently
        semantic_task = asyncio.create_task(
            self._compute_semantic_similarity(original_text, synthetic_text)
        )
        coherence_task = asyncio.create_task(
            self._compute_coherence(synthetic_text)
        )
        fluency_task = asyncio.create_task(
            self._compute_fluency(synthetic_text)
        )
        domain_task = asyncio.create_task(
            self._compute_domain_quality(synthetic_text, domain)
        )
        perplexity_task = asyncio.create_task(
            self._estimate_perplexity(synthetic_text)
        )

        semantic_score = await semantic_task
        coherence_score = await coherence_task
        fluency_score = await fluency_task
        domain_score = await domain_task
        perplexity = await perplexity_task

        # Weighted aggregate: semantic similarity is the primary signal
        aggregate_score = (
            semantic_score * 0.40
            + coherence_score * 0.25
            + fluency_score * 0.20
            + domain_score * 0.15
        )

        passes = (
            semantic_score >= self._min_similarity_threshold
            and coherence_score >= _COHERENCE_THRESHOLD
            and fluency_score >= _FLUENCY_THRESHOLD
        )

        details: dict[str, Any] = {
            "semantic_similarity": round(semantic_score, 4),
            "coherence_score": round(coherence_score, 4),
            "fluency_score": round(fluency_score, 4),
            "domain_quality_score": round(domain_score, 4),
            "perplexity_estimate": round(perplexity, 2),
            "aggregate_score": round(aggregate_score, 4),
            "domain": domain,
            "word_count": len(synthetic_text.split()),
            "sentence_count": len(_SENTENCE_PATTERN.split(synthetic_text.strip())),
        }

        log.info(
            "quality evaluation complete",
            semantic=semantic_score,
            coherence=coherence_score,
            fluency=fluency_score,
            aggregate=aggregate_score,
            passes=passes,
        )

        return QualityReport(
            semantic_similarity=round(semantic_score, 4),
            passes_threshold=passes,
            details=details,
        )

    async def _compute_semantic_similarity(
        self,
        original_text: str,
        synthetic_text: str,
    ) -> float:
        """Compute cosine similarity between document embeddings.

        Args:
            original_text: Reference document.
            synthetic_text: Generated document.

        Returns:
            Cosine similarity in [0.0, 1.0]. Returns 0.5 if model unavailable.
        """
        if not self._model_initialized or self._embedding_model is None:
            self._log.debug("embedding model not available, returning default similarity")
            return 0.5

        loop = asyncio.get_running_loop()

        def _compute() -> float:
            """Compute embeddings and cosine similarity synchronously."""
            import numpy as np

            embeddings = self._embedding_model.encode(
                [original_text, synthetic_text],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            # Cosine similarity of normalized vectors = dot product
            similarity = float(np.dot(embeddings[0], embeddings[1]))
            # Clamp to [0, 1] — normalized vectors can have small negative values
            return max(0.0, min(1.0, similarity))

        return await loop.run_in_executor(None, _compute)

    async def _compute_coherence(self, text: str) -> float:
        """Score entity and topical coherence of the text.

        Measures:
        - Entity repetition ratio (consistent entities across the document)
        - Sentence-level vocabulary overlap (topic stability)

        Args:
            text: Text to evaluate for coherence.

        Returns:
            Coherence score in [0.0, 1.0].
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._coherence_sync, text)

    def _coherence_sync(self, text: str) -> float:
        """Synchronous coherence computation.

        Args:
            text: Input text.

        Returns:
            Coherence score in [0.0, 1.0].
        """
        sentences = [s.strip() for s in _SENTENCE_PATTERN.split(text.strip()) if s.strip()]
        if len(sentences) < 2:
            return 1.0

        # Compute content word sets per sentence
        sentence_words: list[set[str]] = []
        for sentence in sentences:
            words = set(re.findall(r"\b[a-zA-Z]{3,}\b", sentence.lower()))
            content_words = words - _FUNCTION_WORDS
            sentence_words.append(content_words)

        # Measure pairwise overlap between consecutive sentences
        overlap_scores: list[float] = []
        for i in range(len(sentence_words) - 1):
            current = sentence_words[i]
            next_sent = sentence_words[i + 1]
            if not current or not next_sent:
                continue
            overlap = len(current & next_sent)
            union = len(current | next_sent)
            jaccard = overlap / union if union > 0 else 0.0
            overlap_scores.append(jaccard)

        if not overlap_scores:
            return 0.7  # Default for very short texts

        avg_overlap = sum(overlap_scores) / len(overlap_scores)
        # Normalize: 0.0 overlap -> 0.3 score; 0.3+ overlap -> near 1.0
        normalized = min(1.0, 0.3 + avg_overlap * 2.3)
        return normalized

    async def _compute_fluency(self, text: str) -> float:
        """Score fluency using readability and sentence structure metrics.

        Combines:
        - Flesch Reading Ease (adapted)
        - Average sentence length
        - Punctuation density

        Args:
            text: Text to evaluate.

        Returns:
            Fluency score in [0.0, 1.0].
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._fluency_sync, text)

    def _fluency_sync(self, text: str) -> float:
        """Synchronous fluency scoring.

        Args:
            text: Input text.

        Returns:
            Fluency score in [0.0, 1.0].
        """
        words = re.findall(r"\b\w+\b", text)
        sentences = [s for s in _SENTENCE_PATTERN.split(text.strip()) if s.strip()]

        if not words or not sentences:
            return 0.0

        word_count = len(words)
        sentence_count = len(sentences)

        # Average sentence length in words
        avg_sentence_length = word_count / sentence_count

        # Count syllables (approximation)
        syllable_count = sum(_count_syllables(w) for w in words)
        avg_syllables_per_word = syllable_count / word_count if word_count > 0 else 2.0

        # Flesch Reading Ease (206.835 - 1.015 * ASL - 84.6 * ASW)
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        # Normalize to [0, 1]: 0-30 = difficult, 60-100 = easy
        normalized_flesch = max(0.0, min(1.0, flesch_score / 100.0))

        # Penalize very long or very short sentences
        sentence_length_penalty = 0.0
        if avg_sentence_length > 50:
            sentence_length_penalty = min(0.3, (avg_sentence_length - 50) / 100.0)
        elif avg_sentence_length < 5:
            sentence_length_penalty = 0.2

        fluency = max(0.0, normalized_flesch - sentence_length_penalty)
        # Ensure at least 0.3 for any coherent text
        return max(0.3, fluency)

    async def _compute_domain_quality(self, text: str, domain: str) -> float:
        """Score domain-specific terminology density.

        Args:
            text: Generated text to evaluate.
            domain: Domain for indicator term lookup.

        Returns:
            Domain quality score in [0.0, 1.0].
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._domain_quality_sync, text, domain)

    def _domain_quality_sync(self, text: str, domain: str) -> float:
        """Synchronous domain quality scoring.

        Args:
            text: Input text.
            domain: Domain category.

        Returns:
            Domain quality score in [0.0, 1.0].
        """
        indicators = _DOMAIN_INDICATORS.get(domain, _DOMAIN_INDICATORS["general"])
        if not indicators:
            return 0.75  # No domain-specific indicators; neutral score

        text_lower = text.lower()
        words = re.findall(r"\b\w+\b", text_lower)
        word_count = len(words)

        if word_count == 0:
            return 0.0

        matched_count = sum(1 for word in words if word in indicators)
        density = matched_count / word_count

        # Normalize: 0.01+ density is excellent for domain-specific text
        score = min(1.0, density / 0.01)
        # Apply floor for any attempt at domain text
        return max(0.3, score)

    async def _estimate_perplexity(self, text: str) -> float:
        """Estimate perplexity using a simple character bigram model.

        A lower perplexity indicates more natural text. This is a lightweight
        approximation — replace with a proper LM for production accuracy.

        Args:
            text: Text to evaluate.

        Returns:
            Estimated perplexity value. Lower is better.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._perplexity_sync, text)

    def _perplexity_sync(self, text: str) -> float:
        """Synchronous bigram perplexity estimation.

        Args:
            text: Input text.

        Returns:
            Perplexity estimate.
        """
        words = text.lower().split()
        if len(words) < 3:
            return 50.0

        # Build bigram counts
        bigram_counts: dict[tuple[str, str], int] = {}
        unigram_counts: dict[str, int] = {}

        for i in range(len(words) - 1):
            bigram = (words[i], words[i + 1])
            bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
            unigram_counts[words[i]] = unigram_counts.get(words[i], 0) + 1

        # Compute average log probability with Laplace smoothing
        vocab_size = len(unigram_counts)
        total_log_prob = 0.0
        count = 0

        for i in range(len(words) - 1):
            bigram = (words[i], words[i + 1])
            count_bigram = bigram_counts.get(bigram, 0)
            count_unigram = unigram_counts.get(words[i], 0)
            # Laplace smoothed probability
            prob = (count_bigram + 1) / (count_unigram + vocab_size)
            total_log_prob += math.log(prob)
            count += 1

        if count == 0:
            return 50.0

        avg_log_prob = total_log_prob / count
        perplexity = math.exp(-avg_log_prob)
        return min(perplexity, 1000.0)


def _count_syllables(word: str) -> int:
    """Estimate syllable count in a word using vowel group heuristic.

    Args:
        word: Word to count syllables in.

    Returns:
        Estimated syllable count (minimum 1).
    """
    word = word.lower().strip(".,!?;:")
    if not word:
        return 1

    vowel_groups = re.findall(r"[aeiouy]+", word)
    count = len(vowel_groups)

    # Common adjustments
    if word.endswith("e") and count > 1:
        count -= 1
    if word.endswith("le") and len(word) > 2 and word[-3] not in "aeiouy":
        count += 1

    return max(1, count)
