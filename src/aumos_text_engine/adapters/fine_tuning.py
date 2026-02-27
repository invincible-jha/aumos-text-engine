"""LoRA fine-tuning preparation and management adapter.

Handles all preparation steps for domain-specific LoRA fine-tuning:
dataset formatting, validation splits, LoRA config generation, checkpoint
management, and training job monitoring via async polling.
"""

from __future__ import annotations

import asyncio
import json
import random
import uuid
from dataclasses import dataclass, field
from typing import Any

import structlog
from aumos_common.logging import get_logger

logger: structlog.BoundLogger = get_logger(__name__)

# LoRA target modules for common model architectures
_LORA_TARGET_MODULES = {
    "llama": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "mistral": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "falcon": ["query_key_value", "dense"],
    "gpt-neox": ["query_key_value", "dense"],
    "bloom": ["query_key_value", "dense"],
    "default": ["q_proj", "v_proj"],
}

# Supported JSONL conversation formats
FORMAT_ALPACA = "alpaca"
FORMAT_SHAREGPT = "sharegpt"
FORMAT_INSTRUCT = "instruct"

# Minimum recommended training samples per LoRA rank
_MIN_SAMPLES_PER_RANK = 50


@dataclass
class LoRAConfig:
    """Complete LoRA fine-tuning configuration.

    Attributes:
        rank: LoRA rank (r). Higher = more parameters, more capacity.
        lora_alpha: LoRA scaling factor. Typically 2x rank.
        target_modules: Transformer layer names to apply LoRA to.
        lora_dropout: Dropout applied to LoRA layers.
        bias: Whether to train bias parameters (none|all|lora_only).
        task_type: Model task type for PEFT compatibility.
    """

    rank: int = 16
    lora_alpha: int = 32
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def to_peft_config(self) -> dict[str, Any]:
        """Serialize to PEFT LoraConfig constructor arguments.

        Args:
            None

        Returns:
            Dict suitable for PEFT LoraConfig(**config).
        """
        return {
            "r": self.rank,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "lora_dropout": self.lora_dropout,
            "bias": self.bias,
            "task_type": self.task_type,
        }


@dataclass
class TrainingDataset:
    """Prepared fine-tuning dataset with train/validation split.

    Attributes:
        training_samples: JSONL-formatted training examples.
        validation_samples: JSONL-formatted validation examples.
        total_samples: Total sample count before splitting.
        format_type: Dataset conversation format used.
        source_uri: Original corpus URI.
    """

    training_samples: list[dict[str, Any]]
    validation_samples: list[dict[str, Any]]
    total_samples: int
    format_type: str
    source_uri: str

    @property
    def train_count(self) -> int:
        """Number of training samples."""
        return len(self.training_samples)

    @property
    def validation_count(self) -> int:
        """Number of validation samples."""
        return len(self.validation_samples)


@dataclass
class CheckpointInfo:
    """Information about a training checkpoint.

    Attributes:
        step: Training step at which this checkpoint was saved.
        checkpoint_uri: MinIO URI of the checkpoint directory.
        training_loss: Loss at this checkpoint.
        eval_loss: Evaluation loss at this checkpoint (if computed).
        is_best: Whether this is the best checkpoint so far.
    """

    step: int
    checkpoint_uri: str
    training_loss: float
    eval_loss: float | None = None
    is_best: bool = False


class FineTuningAdapter:
    """LoRA fine-tuning preparation and monitoring adapter.

    Manages the full pipeline from raw text corpus to training-ready
    datasets, LoRA configuration, and checkpoint tracking. Does not
    run training itself — training is delegated to the ML infrastructure
    (PEFT + Transformers on GPU nodes).

    Attributes:
        _default_lora_rank: Default LoRA rank if not specified.
        _checkpoint_poll_interval: Seconds between checkpoint status polls.
        _log: Structured logger.
    """

    def __init__(
        self,
        default_lora_rank: int = 16,
        checkpoint_poll_interval: float = 30.0,
    ) -> None:
        """Initialize the FineTuningAdapter.

        Args:
            default_lora_rank: Default LoRA rank to use when not specified.
            checkpoint_poll_interval: Seconds to wait between training status polls.
        """
        self._default_lora_rank = default_lora_rank
        self._checkpoint_poll_interval = checkpoint_poll_interval
        self._checkpoints: dict[str, list[CheckpointInfo]] = {}
        self._log: structlog.BoundLogger = get_logger(__name__)

    async def prepare_dataset(
        self,
        raw_samples: list[dict[str, Any]],
        format_type: str = FORMAT_INSTRUCT,
        validation_split: float = 0.1,
        source_uri: str = "",
        shuffle: bool = True,
        seed: int = 42,
    ) -> TrainingDataset:
        """Convert raw text samples into JSONL training format with validation split.

        Supports multiple conversation formats:
        - instruct: {"instruction": ..., "input": ..., "output": ...}
        - alpaca: Same as instruct (Alpaca-style)
        - sharegpt: {"conversations": [{"from": "human", "value": ...}, ...]}

        Args:
            raw_samples: List of raw sample dicts to convert.
            format_type: Output conversation format (instruct|alpaca|sharegpt).
            validation_split: Fraction of samples to reserve for validation.
            source_uri: Original corpus URI for tracking.
            shuffle: Whether to shuffle samples before splitting.
            seed: Random seed for reproducible shuffling.

        Returns:
            TrainingDataset with formatted training and validation splits.

        Raises:
            ValueError: If raw_samples is empty or format_type is unknown.
        """
        if not raw_samples:
            raise ValueError("Cannot prepare dataset from empty samples list")

        valid_formats = {FORMAT_INSTRUCT, FORMAT_ALPACA, FORMAT_SHAREGPT}
        if format_type not in valid_formats:
            raise ValueError(f"Unknown format_type '{format_type}'. Valid: {valid_formats}")

        self._log.info(
            "preparing fine-tuning dataset",
            sample_count=len(raw_samples),
            format=format_type,
            validation_split=validation_split,
        )

        # Format samples in executor (CPU-bound)
        loop = asyncio.get_running_loop()
        formatted = await loop.run_in_executor(
            None,
            self._format_samples,
            raw_samples,
            format_type,
        )

        # Shuffle and split
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(formatted)

        split_index = max(1, int(len(formatted) * (1.0 - validation_split)))
        training_samples = formatted[:split_index]
        validation_samples = formatted[split_index:]

        dataset = TrainingDataset(
            training_samples=training_samples,
            validation_samples=validation_samples,
            total_samples=len(raw_samples),
            format_type=format_type,
            source_uri=source_uri,
        )

        self._log.info(
            "dataset prepared",
            train_count=dataset.train_count,
            validation_count=dataset.validation_count,
        )
        return dataset

    def _format_samples(
        self,
        raw_samples: list[dict[str, Any]],
        format_type: str,
    ) -> list[dict[str, Any]]:
        """Format raw samples into the target JSONL conversation format.

        Args:
            raw_samples: Raw sample dicts with text/instruction/output keys.
            format_type: Target format identifier.

        Returns:
            List of formatted sample dicts.
        """
        formatted: list[dict[str, Any]] = []

        for sample in raw_samples:
            instruction = sample.get("instruction", "")
            input_text = sample.get("input", "")
            output_text = sample.get("output", "") or sample.get("text", "")

            if not output_text:
                continue

            if format_type in (FORMAT_INSTRUCT, FORMAT_ALPACA):
                formatted.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": output_text,
                })
            elif format_type == FORMAT_SHAREGPT:
                conversations: list[dict[str, str]] = []
                if instruction:
                    user_message = f"{instruction}\n\n{input_text}".strip()
                else:
                    user_message = input_text or output_text
                conversations.append({"from": "human", "value": user_message})
                conversations.append({"from": "gpt", "value": output_text})
                formatted.append({"conversations": conversations})

        return formatted

    async def serialize_to_jsonl(self, dataset: TrainingDataset) -> tuple[str, str]:
        """Serialize training and validation splits to JSONL strings.

        Args:
            dataset: Prepared training dataset.

        Returns:
            Tuple of (training_jsonl_string, validation_jsonl_string).
        """
        loop = asyncio.get_running_loop()
        train_jsonl, val_jsonl = await loop.run_in_executor(
            None,
            self._serialize_splits,
            dataset.training_samples,
            dataset.validation_samples,
        )
        return train_jsonl, val_jsonl

    def _serialize_splits(
        self,
        training_samples: list[dict[str, Any]],
        validation_samples: list[dict[str, Any]],
    ) -> tuple[str, str]:
        """Synchronous JSONL serialization of both splits.

        Args:
            training_samples: Training split samples.
            validation_samples: Validation split samples.

        Returns:
            Tuple of (training_jsonl, validation_jsonl) strings.
        """
        train_jsonl = "\n".join(json.dumps(s, ensure_ascii=False) for s in training_samples)
        val_jsonl = "\n".join(json.dumps(s, ensure_ascii=False) for s in validation_samples)
        return train_jsonl, val_jsonl

    def generate_lora_config(
        self,
        base_model: str,
        rank: int | None = None,
        lora_alpha: int | None = None,
        target_modules: list[str] | None = None,
        lora_dropout: float = 0.05,
        bias: str = "none",
    ) -> LoRAConfig:
        """Generate a LoRA configuration for a given base model.

        Automatically selects appropriate target modules based on the
        model architecture prefix. Falls back to default modules if
        the architecture is not recognized.

        Args:
            base_model: Base model name (e.g. llama3-8b-instruct).
            rank: LoRA rank. Uses default if not provided.
            lora_alpha: Scaling factor. Defaults to 2x rank.
            target_modules: Override module names. Auto-detected if None.
            lora_dropout: Dropout rate for LoRA layers.
            bias: Bias training mode (none|all|lora_only).

        Returns:
            LoRAConfig instance with all parameters set.
        """
        effective_rank = rank or self._default_lora_rank
        effective_alpha = lora_alpha or (effective_rank * 2)

        # Auto-detect target modules from model name
        if target_modules is None:
            detected_modules = _LORA_TARGET_MODULES["default"]
            model_lower = base_model.lower()
            for arch, modules in _LORA_TARGET_MODULES.items():
                if arch != "default" and arch in model_lower:
                    detected_modules = modules
                    break
            effective_target_modules = detected_modules
        else:
            effective_target_modules = target_modules

        lora_config = LoRAConfig(
            rank=effective_rank,
            lora_alpha=effective_alpha,
            target_modules=effective_target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
        )

        self._log.debug(
            "LoRA config generated",
            base_model=base_model,
            rank=effective_rank,
            target_modules=effective_target_modules,
        )
        return lora_config

    def validate_dataset_for_rank(
        self,
        dataset: TrainingDataset,
        lora_config: LoRAConfig,
    ) -> list[str]:
        """Validate that the dataset is sufficient for the given LoRA rank.

        Higher ranks require more training samples to converge effectively.
        Returns a list of warnings (empty if dataset is adequate).

        Args:
            dataset: Prepared training dataset.
            lora_config: LoRA configuration to validate against.

        Returns:
            List of warning strings (empty if no issues).
        """
        warnings: list[str] = []
        min_recommended = lora_config.rank * _MIN_SAMPLES_PER_RANK

        if dataset.train_count < min_recommended:
            warnings.append(
                f"Training set has {dataset.train_count} samples, but LoRA rank "
                f"{lora_config.rank} recommends at least {min_recommended} samples. "
                "Consider reducing rank or increasing corpus size."
            )

        if dataset.validation_count < 10:
            warnings.append(
                f"Validation set has only {dataset.validation_count} samples. "
                "Evaluation metrics may not be reliable."
            )

        return warnings

    def record_checkpoint(
        self,
        job_id: str,
        step: int,
        checkpoint_uri: str,
        training_loss: float,
        eval_loss: float | None = None,
    ) -> CheckpointInfo:
        """Record a training checkpoint for a job.

        Args:
            job_id: Fine-tuning job identifier.
            step: Training step number.
            checkpoint_uri: MinIO URI for the checkpoint.
            training_loss: Loss at this checkpoint.
            eval_loss: Optional evaluation loss.

        Returns:
            CheckpointInfo for the recorded checkpoint.
        """
        if job_id not in self._checkpoints:
            self._checkpoints[job_id] = []

        existing = self._checkpoints[job_id]
        best_eval = min(
            (c.eval_loss for c in existing if c.eval_loss is not None),
            default=float("inf"),
        )

        is_best = eval_loss is not None and eval_loss < best_eval

        checkpoint = CheckpointInfo(
            step=step,
            checkpoint_uri=checkpoint_uri,
            training_loss=training_loss,
            eval_loss=eval_loss,
            is_best=is_best,
        )

        if is_best:
            # Clear previous best flags
            for c in existing:
                c.is_best = False

        self._checkpoints[job_id].append(checkpoint)

        self._log.info(
            "checkpoint recorded",
            job_id=job_id,
            step=step,
            training_loss=training_loss,
            eval_loss=eval_loss,
            is_best=is_best,
        )
        return checkpoint

    def get_best_checkpoint(self, job_id: str) -> CheckpointInfo | None:
        """Return the best checkpoint for a job based on eval loss.

        Args:
            job_id: Fine-tuning job identifier.

        Returns:
            CheckpointInfo for the best checkpoint, or None if no checkpoints.
        """
        checkpoints = self._checkpoints.get(job_id, [])
        if not checkpoints:
            return None

        evaluated = [c for c in checkpoints if c.eval_loss is not None]
        if evaluated:
            return min(evaluated, key=lambda c: c.eval_loss or float("inf"))

        # Fall back to latest checkpoint
        return checkpoints[-1]

    def list_checkpoints(self, job_id: str) -> list[CheckpointInfo]:
        """List all checkpoints for a job in chronological order.

        Args:
            job_id: Fine-tuning job identifier.

        Returns:
            List of CheckpointInfo sorted by step.
        """
        return sorted(self._checkpoints.get(job_id, []), key=lambda c: c.step)

    def generate_merge_config(
        self,
        job_id: str,
        base_model: str,
        lora_config: LoRAConfig,
        output_uri: str,
    ) -> dict[str, Any]:
        """Generate the configuration for merging LoRA weights into the base model.

        Produces a config dict that can be used with peft.merge_adapter()
        or the merge_lora.py utility to create a standalone model.

        Args:
            job_id: Fine-tuning job identifier.
            base_model: Base model name to merge into.
            lora_config: LoRA configuration used during training.
            output_uri: MinIO URI where the merged model should be saved.

        Returns:
            Merge configuration dict.
        """
        best_checkpoint = self.get_best_checkpoint(job_id)
        checkpoint_uri = best_checkpoint.checkpoint_uri if best_checkpoint else ""

        return {
            "job_id": job_id,
            "base_model": base_model,
            "lora_checkpoint_uri": checkpoint_uri,
            "lora_config": lora_config.to_peft_config(),
            "output_uri": output_uri,
            "merge_method": "linear",
            "safe_serialization": True,
        }
