"""PII detection accuracy benchmark — GAP-70.

Evaluates Presidio + spaCy NER on labeled test corpora.
Reports per-entity-type precision, recall, and F1 score.

Usage:
    python benchmarks/pii_detection/benchmark_runner.py --corpus internal --output results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class EntityMetrics:
    """Per-entity-type precision, recall, and F1 metrics.

    Attributes:
        entity_type: Presidio entity type label (e.g., "PERSON", "EMAIL_ADDRESS").
        true_positives: Correctly detected entities.
        false_positives: Incorrectly flagged spans.
        false_negatives: Missed entities.
    """

    entity_type: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)."""
        denominator = self.true_positives + self.false_positives
        if denominator == 0:
            return 0.0
        return self.true_positives / denominator

    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)."""
        denominator = self.true_positives + self.false_negatives
        if denominator == 0:
            return 0.0
        return self.true_positives / denominator

    @property
    def f1(self) -> float:
        """F1 = 2 * precision * recall / (precision + recall)."""
        denom = self.precision + self.recall
        if denom == 0:
            return 0.0
        return 2 * self.precision * self.recall / denom


class PIIBenchmarkRunner:
    """Evaluates PII detection on a labeled test corpus.

    Args:
        language: Language code for Presidio analysis (default: "en").
        score_threshold: Minimum confidence for a detection to count (default: 0.7).
    """

    def __init__(
        self,
        language: str = "en",
        score_threshold: float = 0.7,
    ) -> None:
        self._language = language
        self._score_threshold = score_threshold
        self._analyzer: Any = None

    def _build_analyzer(self) -> Any:
        """Build and return a configured Presidio AnalyzerEngine.

        Returns:
            Configured AnalyzerEngine instance.

        Raises:
            ImportError: If presidio-analyzer is not installed.
        """
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": self._language, "model_name": "en_core_web_lg"}],
        })
        return AnalyzerEngine(
            nlp_engine=provider.create_engine(),
            supported_languages=[self._language],
        )

    def _get_analyzer(self) -> Any:
        if self._analyzer is None:
            self._analyzer = self._build_analyzer()
        return self._analyzer

    def evaluate_document(
        self,
        text: str,
        ground_truth: list[dict[str, Any]],
    ) -> dict[str, EntityMetrics]:
        """Evaluate a single document and return per-entity metrics.

        Args:
            text: Document text.
            ground_truth: List of dicts with keys: entity_type, start, end.

        Returns:
            Dictionary mapping entity_type to EntityMetrics.
        """
        analyzer = self._get_analyzer()
        results = analyzer.analyze(
            text=text,
            language=self._language,
            score_threshold=self._score_threshold,
        )

        metrics: dict[str, EntityMetrics] = {}

        # Build ground truth index by entity type
        gt_by_type: dict[str, list[tuple[int, int]]] = {}
        for gt in ground_truth:
            entity_type = gt["entity_type"]
            gt_by_type.setdefault(entity_type, []).append((gt["start"], gt["end"]))

        # Build detected spans by entity type
        detected_by_type: dict[str, list[tuple[int, int]]] = {}
        for result in results:
            detected_by_type.setdefault(result.entity_type, []).append(
                (result.start, result.end)
            )

        # Compute metrics per entity type
        all_types = set(gt_by_type.keys()) | set(detected_by_type.keys())
        for entity_type in all_types:
            gt_spans = set(gt_by_type.get(entity_type, []))
            det_spans = set(detected_by_type.get(entity_type, []))

            tp = len(gt_spans & det_spans)
            fp = len(det_spans - gt_spans)
            fn = len(gt_spans - det_spans)

            metrics[entity_type] = EntityMetrics(
                entity_type=entity_type,
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
            )

        return metrics

    def run_corpus(
        self,
        corpus_path: Path,
    ) -> dict[str, EntityMetrics]:
        """Run benchmark on an entire labeled corpus directory.

        Args:
            corpus_path: Directory containing .jsonl corpus files.

        Returns:
            Aggregate EntityMetrics per entity type across all documents.
        """
        aggregate: dict[str, EntityMetrics] = {}

        for jsonl_file in corpus_path.glob("*.jsonl"):
            with jsonl_file.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    doc = json.loads(line)
                    doc_metrics = self.evaluate_document(
                        text=doc["text"],
                        ground_truth=doc.get("entities", []),
                    )
                    for entity_type, m in doc_metrics.items():
                        if entity_type not in aggregate:
                            aggregate[entity_type] = EntityMetrics(entity_type=entity_type)
                        aggregate[entity_type].true_positives += m.true_positives
                        aggregate[entity_type].false_positives += m.false_positives
                        aggregate[entity_type].false_negatives += m.false_negatives

        return aggregate


def main() -> None:
    """Run PII benchmark and output results JSON."""
    parser = argparse.ArgumentParser(description="AumOS PII Detection Benchmark")
    parser.add_argument(
        "--corpus",
        default="internal",
        choices=["internal"],
        help="Corpus to benchmark against",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/pii_detection/results/benchmark-results.json",
        help="Output results JSON path",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.7,
        help="Minimum Presidio confidence score",
    )
    args = parser.parse_args()

    corpus_path = Path(__file__).parent / "corpus"
    runner = PIIBenchmarkRunner(score_threshold=args.score_threshold)

    print(f"Running PII benchmark on corpus: {corpus_path}")
    start_ts = time.monotonic()
    metrics = runner.run_corpus(corpus_path)
    duration = time.monotonic() - start_ts

    results: dict[str, Any] = {
        "duration_s": duration,
        "score_threshold": args.score_threshold,
        "entity_metrics": {
            entity_type: {
                "precision": round(m.precision, 4),
                "recall": round(m.recall, 4),
                "f1": round(m.f1, 4),
                "true_positives": m.true_positives,
                "false_positives": m.false_positives,
                "false_negatives": m.false_negatives,
            }
            for entity_type, m in sorted(metrics.items())
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"Results written to {output_path}")
    print("\nTop Entity F1 Scores:")
    for entity_type, m in sorted(metrics.items(), key=lambda kv: kv[1].f1, reverse=True)[:10]:
        print(f"  {entity_type}: F1={m.f1:.3f} P={m.precision:.3f} R={m.recall:.3f}")


if __name__ == "__main__":
    main()
