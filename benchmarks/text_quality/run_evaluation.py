"""AumOS Text Engine quality evaluation benchmark.

Computes PII detection precision/recall/F1 and BERTScore for synthesis quality
on a labeled 500-document test corpus (50 per domain, 10 domains).

Usage:
    python benchmarks/text_quality/run_evaluation.py --output results/evaluation.json
    python benchmarks/text_quality/run_evaluation.py --domains legal medical --mode pii_only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED_DOMAINS = [
    "legal_nda",
    "legal_employment",
    "legal_privacy",
    "medical_notes",
    "medical_discharge",
    "financial_earnings",
    "financial_loans",
    "financial_compliance",
    "hr_performance",
    "hr_policy",
]


@dataclass
class PIIEvalResult:
    """PII detection evaluation results for one domain.

    Attributes:
        domain: Domain name evaluated.
        entity_metrics: Per-entity-type precision/recall/F1.
        overall_precision: Macro-averaged precision across all entity types.
        overall_recall: Macro-averaged recall across all entity types.
        overall_f1: Macro-averaged F1 across all entity types.
        num_documents: Number of documents evaluated.
    """

    domain: str
    entity_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    overall_precision: float = 0.0
    overall_recall: float = 0.0
    overall_f1: float = 0.0
    num_documents: int = 0


@dataclass
class SynthesisEvalResult:
    """Synthesis quality evaluation results for one domain.

    Attributes:
        domain: Domain name evaluated.
        bert_score_precision: BERTScore precision (token-level semantic overlap).
        bert_score_recall: BERTScore recall.
        bert_score_f1: BERTScore F1.
        num_documents: Number of document pairs evaluated.
    """

    domain: str
    bert_score_precision: float = 0.0
    bert_score_recall: float = 0.0
    bert_score_f1: float = 0.0
    num_documents: int = 0


class PIIEvaluator:
    """Evaluates PII detection precision, recall, and F1 on labeled corpus.

    Args:
        score_threshold: Minimum Presidio confidence to count a detection.
    """

    def __init__(self, score_threshold: float = 0.7) -> None:
        """Create a PIIEvaluator.

        Args:
            score_threshold: Minimum confidence for a detection to count.
        """
        self._score_threshold = score_threshold
        self._analyzer: Any = None

    def _get_analyzer(self) -> Any:
        """Build and return a Presidio AnalyzerEngine (English).

        Returns:
            Configured AnalyzerEngine instance.
        """
        if self._analyzer is None:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider

            provider = NlpEngineProvider(nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
            })
            self._analyzer = AnalyzerEngine(
                nlp_engine=provider.create_engine(),
                supported_languages=["en"],
            )
        return self._analyzer

    def evaluate_document(
        self,
        text: str,
        ground_truth: list[dict[str, Any]],
    ) -> dict[str, dict[str, int]]:
        """Evaluate PII detection on a single labeled document.

        Args:
            text: Document text to analyze.
            ground_truth: List of {entity_type, start, end} ground truth annotations.

        Returns:
            Dict mapping entity_type to {tp, fp, fn} counts.
        """
        analyzer = self._get_analyzer()
        try:
            results = analyzer.analyze(
                text=text,
                language="en",
                score_threshold=self._score_threshold,
            )
        except Exception:
            results = []

        gt_by_type: dict[str, set[tuple[int, int]]] = {}
        for gt in ground_truth:
            gt_by_type.setdefault(gt["entity_type"], set()).add((gt["start"], gt["end"]))

        det_by_type: dict[str, set[tuple[int, int]]] = {}
        for res in results:
            det_by_type.setdefault(res.entity_type, set()).add((res.start, res.end))

        all_types = set(gt_by_type) | set(det_by_type)
        counts: dict[str, dict[str, int]] = {}
        for entity_type in all_types:
            gt_spans = gt_by_type.get(entity_type, set())
            det_spans = det_by_type.get(entity_type, set())
            counts[entity_type] = {
                "tp": len(gt_spans & det_spans),
                "fp": len(det_spans - gt_spans),
                "fn": len(gt_spans - det_spans),
            }
        return counts

    def evaluate_corpus(self, corpus_path: Path, domain: str) -> PIIEvalResult:
        """Run PII evaluation on all documents in a corpus directory.

        Args:
            corpus_path: Path to directory containing {domain}.jsonl files.
            domain: Domain name to evaluate.

        Returns:
            PIIEvalResult with per-entity and overall metrics.
        """
        corpus_file = corpus_path / f"{domain}.jsonl"
        if not corpus_file.exists():
            return PIIEvalResult(domain=domain)

        aggregate: dict[str, dict[str, int]] = {}
        num_docs = 0

        with corpus_file.open() as f:
            for line in f:
                if not line.strip():
                    continue
                doc = json.loads(line)
                doc_counts = self.evaluate_document(
                    text=doc["text"],
                    ground_truth=doc.get("entities", []),
                )
                for entity_type, counts in doc_counts.items():
                    if entity_type not in aggregate:
                        aggregate[entity_type] = {"tp": 0, "fp": 0, "fn": 0}
                    for key in ("tp", "fp", "fn"):
                        aggregate[entity_type][key] += counts[key]
                num_docs += 1

        entity_metrics: dict[str, dict[str, float]] = {}
        all_p, all_r, all_f1 = [], [], []

        for entity_type, counts in aggregate.items():
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            entity_metrics[entity_type] = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
            }
            all_p.append(p)
            all_r.append(r)
            all_f1.append(f1)

        return PIIEvalResult(
            domain=domain,
            entity_metrics=entity_metrics,
            overall_precision=round(sum(all_p) / len(all_p), 4) if all_p else 0.0,
            overall_recall=round(sum(all_r) / len(all_r), 4) if all_r else 0.0,
            overall_f1=round(sum(all_f1) / len(all_f1), 4) if all_f1 else 0.0,
            num_documents=num_docs,
        )


class SynthesisEvaluator:
    """Evaluates synthesis quality using BERTScore.

    Compares original (post-PII-replacement) documents against synthesized
    documents using BERTScore for semantic similarity measurement.

    Args:
        model_type: BERTScore model (default: "distilbert-base-uncased").
    """

    def __init__(self, model_type: str = "distilbert-base-uncased") -> None:
        """Create a SynthesisEvaluator.

        Args:
            model_type: HuggingFace model identifier for BERTScore computation.
        """
        self._model_type = model_type

    def evaluate_corpus(
        self,
        corpus_path: Path,
        domain: str,
    ) -> SynthesisEvalResult:
        """Compute BERTScore between original and synthesized document pairs.

        Args:
            corpus_path: Path to corpus directory containing {domain}_pairs.jsonl.
            domain: Domain name to evaluate.

        Returns:
            SynthesisEvalResult with BERTScore metrics.
        """
        pairs_file = corpus_path / f"{domain}_pairs.jsonl"
        if not pairs_file.exists():
            return SynthesisEvalResult(domain=domain)

        originals: list[str] = []
        synthesized: list[str] = []

        with pairs_file.open() as f:
            for line in f:
                if not line.strip():
                    continue
                pair = json.loads(line)
                originals.append(pair.get("original", ""))
                synthesized.append(pair.get("synthesized", ""))

        if not originals:
            return SynthesisEvalResult(domain=domain)

        try:
            from bert_score import score as bert_score

            p_scores, r_scores, f1_scores = bert_score(
                cands=synthesized,
                refs=originals,
                model_type=self._model_type,
                lang="en",
                verbose=False,
            )
            return SynthesisEvalResult(
                domain=domain,
                bert_score_precision=round(float(p_scores.mean()), 4),
                bert_score_recall=round(float(r_scores.mean()), 4),
                bert_score_f1=round(float(f1_scores.mean()), 4),
                num_documents=len(originals),
            )
        except ImportError:
            print("bert-score not installed. pip install bert-score")
            return SynthesisEvalResult(domain=domain, num_documents=len(originals))


async def main() -> None:
    """Run full text quality evaluation benchmark."""
    parser = argparse.ArgumentParser(description="AumOS Text Engine Quality Evaluation")
    parser.add_argument(
        "--corpus",
        default="benchmarks/text_quality/corpus",
        help="Path to evaluation corpus directory",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/text_quality/results/evaluation.json",
        help="Output JSON results path",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=SUPPORTED_DOMAINS,
        help="Domains to evaluate",
    )
    parser.add_argument(
        "--mode",
        choices=["pii_only", "synthesis_only", "full"],
        default="full",
        help="Evaluation mode",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.7,
        help="PII detection confidence threshold",
    )
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    start_ts = time.monotonic()

    pii_evaluator = PIIEvaluator(score_threshold=args.score_threshold)
    synth_evaluator = SynthesisEvaluator()

    results: dict[str, Any] = {
        "run_date": __import__("datetime").datetime.utcnow().isoformat(),
        "score_threshold": args.score_threshold,
        "mode": args.mode,
        "domains_evaluated": args.domains,
        "pii_results": {},
        "synthesis_results": {},
    }

    for domain in args.domains:
        print(f"Evaluating domain: {domain}")

        if args.mode in ("pii_only", "full"):
            pii_result = await asyncio.to_thread(
                pii_evaluator.evaluate_corpus, corpus_path, domain
            )
            results["pii_results"][domain] = asdict(pii_result)
            print(
                f"  PII — P={pii_result.overall_precision:.3f} "
                f"R={pii_result.overall_recall:.3f} "
                f"F1={pii_result.overall_f1:.3f} "
                f"({pii_result.num_documents} docs)"
            )

        if args.mode in ("synthesis_only", "full"):
            synth_result = await asyncio.to_thread(
                synth_evaluator.evaluate_corpus, corpus_path, domain
            )
            results["synthesis_results"][domain] = asdict(synth_result)
            print(
                f"  BERTScore — F1={synth_result.bert_score_f1:.3f} "
                f"({synth_result.num_documents} pairs)"
            )

    duration = time.monotonic() - start_ts
    results["duration_s"] = round(duration, 2)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to {output_path}")
    print(f"Total evaluation time: {duration:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
