"""Structured output validation and parsing adapter.

Validates LLM outputs against JSON schemas, parses YAML, extracts fields
from unstructured text, sanitizes markdown artifacts, and coerces values
to expected types. Supports retry-on-failure with reformatted prompts.
"""

from __future__ import annotations

import json
import re
from typing import Any

import structlog
from aumos_common.logging import get_logger

logger: structlog.BoundLogger = get_logger(__name__)

# Markdown fence pattern
_FENCE_PATTERN = re.compile(r"```(?:json|yaml|python|xml|csv)?\s*\n?(.*?)\n?```", re.DOTALL)

# JSON extraction pattern (find JSON object or array in text)
_JSON_OBJECT_PATTERN = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)
_JSON_ARRAY_PATTERN = re.compile(r"\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]", re.DOTALL)

# YAML block indicators
_YAML_INDICATORS = {":", "- ", "---"}


class OutputParser:
    """Structured LLM output parser with schema validation.

    Parses and validates LLM-generated structured outputs (JSON, YAML),
    extracts fields from unstructured text using patterns, sanitizes
    markdown formatting artifacts, and coerces values to expected schemas.

    Attributes:
        _strict_mode: Whether to raise on schema mismatches vs. returning partial data.
        _log: Structured logger.
    """

    def __init__(self, strict_mode: bool = False) -> None:
        """Initialize the OutputParser.

        Args:
            strict_mode: If True, raise ValueError on schema mismatches.
                         If False, return partial data with warnings logged.
        """
        self._strict_mode = strict_mode
        self._log: structlog.BoundLogger = get_logger(__name__)

    def sanitize(self, raw_text: str) -> str:
        """Remove markdown artifacts and normalize whitespace.

        Strips:
        - Code fences (```json, ```, etc.)
        - Leading/trailing whitespace
        - Common LLM preamble patterns ("Sure, here is..." etc.)

        Args:
            raw_text: Raw LLM output string.

        Returns:
            Sanitized text string.
        """
        text = raw_text.strip()

        # Extract content from code fences if present
        fence_match = _FENCE_PATTERN.search(text)
        if fence_match:
            return fence_match.group(1).strip()

        # Remove common LLM preamble phrases
        preamble_patterns = [
            r"^Sure,?\s+here(?:'s| is).*?:\s*",
            r"^Here(?:'s| is).*?:\s*",
            r"^I'll\s+\w+.*?:\s*",
            r"^Certainly[!.]?\s*",
            r"^Of course[!.]?\s*",
        ]
        for pattern in preamble_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

        return text.strip()

    def parse_json(self, raw_text: str) -> dict[str, Any]:
        """Parse JSON from LLM output, attempting multiple extraction strategies.

        Strategies (in order):
        1. Direct JSON parse after sanitization
        2. Extract first JSON object using regex
        3. Extract first JSON array using regex

        Args:
            raw_text: Raw LLM output that should contain JSON.

        Returns:
            Parsed JSON as a dict or wrapped list.

        Raises:
            ValueError: If no valid JSON can be extracted.
        """
        # Strategy 1: Direct parse after sanitization
        sanitized = self.sanitize(raw_text)
        try:
            parsed = json.loads(sanitized)
            return parsed if isinstance(parsed, dict) else {"items": parsed}
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract JSON object via regex
        object_match = _JSON_OBJECT_PATTERN.search(raw_text)
        if object_match:
            try:
                parsed = json.loads(object_match.group(0))
                return dict(parsed) if isinstance(parsed, dict) else {"items": parsed}
            except json.JSONDecodeError:
                pass

        # Strategy 3: Extract JSON array via regex
        array_match = _JSON_ARRAY_PATTERN.search(raw_text)
        if array_match:
            try:
                parsed = json.loads(array_match.group(0))
                return {"items": parsed}
            except json.JSONDecodeError:
                pass

        raise ValueError(
            f"Could not extract valid JSON from output. "
            f"First 200 chars: {raw_text[:200]}"
        )

    def parse_yaml(self, raw_text: str) -> dict[str, Any]:
        """Parse YAML from LLM output.

        Args:
            raw_text: Raw LLM output containing YAML.

        Returns:
            Parsed YAML as a dict.

        Raises:
            ValueError: If YAML parsing fails or yaml package unavailable.
        """
        try:
            import yaml
        except ImportError as exc:
            raise ValueError("PyYAML not installed — cannot parse YAML output") from exc

        sanitized = self.sanitize(raw_text)

        try:
            parsed = yaml.safe_load(sanitized)
            if isinstance(parsed, dict):
                return parsed
            elif isinstance(parsed, list):
                return {"items": parsed}
            else:
                return {"value": parsed}
        except yaml.YAMLError as exc:
            raise ValueError(f"YAML parse failed: {exc}") from exc

    def validate_schema(
        self,
        data: dict[str, Any],
        schema: dict[str, Any],
    ) -> list[str]:
        """Validate a parsed dict against a JSON schema.

        Performs lightweight validation:
        - Checks required fields presence
        - Checks basic type constraints
        - Returns list of validation errors (empty if valid)

        For full JSON Schema validation, use jsonschema library.

        Args:
            data: Parsed data dict to validate.
            schema: JSON schema dict with "required" and "properties".

        Returns:
            List of validation error messages. Empty if valid.
        """
        errors: list[str] = []

        # Check required fields
        required_fields = schema.get("required", [])
        for field_name in required_fields:
            if field_name not in data:
                errors.append(f"Missing required field: '{field_name}'")

        # Check property types
        properties = schema.get("properties", {})
        for field_name, field_schema in properties.items():
            if field_name not in data:
                continue

            value = data[field_name]
            expected_type = field_schema.get("type")

            if expected_type and not _check_json_type(value, expected_type):
                errors.append(
                    f"Field '{field_name}': expected type '{expected_type}', "
                    f"got '{type(value).__name__}'"
                )

            # Check enum values
            enum_values = field_schema.get("enum")
            if enum_values and value not in enum_values:
                errors.append(
                    f"Field '{field_name}': value '{value}' not in allowed values {enum_values}"
                )

            # Check string constraints
            if expected_type == "string" and isinstance(value, str):
                min_length = field_schema.get("minLength")
                max_length = field_schema.get("maxLength")
                if min_length and len(value) < min_length:
                    errors.append(f"Field '{field_name}': string too short (min {min_length})")
                if max_length and len(value) > max_length:
                    errors.append(f"Field '{field_name}': string too long (max {max_length})")

        return errors

    def parse_and_validate(
        self,
        raw_text: str,
        schema: dict[str, Any],
        format_type: str = "json",
    ) -> tuple[dict[str, Any], list[str]]:
        """Parse output and validate against schema in one step.

        Args:
            raw_text: Raw LLM output.
            schema: JSON schema to validate against.
            format_type: Expected format ("json" or "yaml").

        Returns:
            Tuple of (parsed_data, validation_errors). If validation errors
            exist and strict_mode is enabled, raises ValueError instead.

        Raises:
            ValueError: If parsing fails or (strict_mode=True and validation fails).
        """
        if format_type == "yaml":
            data = self.parse_yaml(raw_text)
        else:
            data = self.parse_json(raw_text)

        errors = self.validate_schema(data, schema)

        if errors and self._strict_mode:
            raise ValueError(f"Schema validation failed: {errors}")

        if errors:
            self._log.warning("schema validation warnings", errors=errors)

        return data, errors

    def extract_fields(
        self,
        text: str,
        field_patterns: dict[str, str],
    ) -> dict[str, str]:
        """Extract named fields from unstructured text using regex patterns.

        Args:
            text: Unstructured text to extract fields from.
            field_patterns: Dict mapping field names to regex patterns.
                           Each pattern should have one capture group.

        Returns:
            Dict of extracted field name -> matched value.
            Missing matches are omitted from the result.
        """
        extracted: dict[str, str] = {}

        for field_name, pattern in field_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    extracted[field_name] = match.group(1).strip()
                except IndexError:
                    extracted[field_name] = match.group(0).strip()

        self._log.debug(
            "field extraction complete",
            fields_requested=len(field_patterns),
            fields_extracted=len(extracted),
        )
        return extracted

    def coerce_types(
        self,
        data: dict[str, Any],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Coerce string values in data to types specified by the schema.

        Handles common LLM output issues where numbers are returned as strings.
        Supported coercions: string -> int, string -> float, string -> bool.

        Args:
            data: Parsed data dict with potentially incorrect types.
            schema: JSON schema specifying expected types.

        Returns:
            Data dict with coerced value types.
        """
        coerced = dict(data)
        properties = schema.get("properties", {})

        for field_name, field_schema in properties.items():
            if field_name not in coerced:
                continue

            value = coerced[field_name]
            expected_type = field_schema.get("type")

            if expected_type == "integer" and isinstance(value, str):
                try:
                    coerced[field_name] = int(value.replace(",", ""))
                except ValueError:
                    self._log.warning("coercion failed", field=field_name, value=str(value)[:50])

            elif expected_type == "number" and isinstance(value, str):
                try:
                    coerced[field_name] = float(value.replace(",", ""))
                except ValueError:
                    self._log.warning("coercion failed", field=field_name, value=str(value)[:50])

            elif expected_type == "boolean" and isinstance(value, str):
                lower_val = value.lower().strip()
                if lower_val in ("true", "yes", "1", "on"):
                    coerced[field_name] = True
                elif lower_val in ("false", "no", "0", "off"):
                    coerced[field_name] = False

            elif expected_type == "array" and isinstance(value, str):
                # Try to parse as JSON array
                try:
                    coerced[field_name] = json.loads(value)
                except json.JSONDecodeError:
                    # Split by comma as fallback
                    coerced[field_name] = [v.strip() for v in value.split(",") if v.strip()]

        return coerced

    def build_retry_prompt(
        self,
        original_prompt: str,
        failed_output: str,
        validation_errors: list[str],
        schema: dict[str, Any],
    ) -> str:
        """Build a reformatted prompt for retry after validation failure.

        Args:
            original_prompt: The original generation prompt.
            failed_output: The invalid output from the previous attempt.
            validation_errors: List of validation error messages.
            schema: JSON schema the output must conform to.

        Returns:
            Augmented prompt instructing the model to fix the issues.
        """
        error_list = "\n".join(f"- {err}" for err in validation_errors)
        schema_str = json.dumps(schema, indent=2)

        return (
            f"{original_prompt}\n\n"
            f"Your previous response had the following issues:\n{error_list}\n\n"
            f"The response failed to match this JSON schema:\n{schema_str}\n\n"
            "Please provide a corrected response that strictly follows the schema. "
            "Respond ONLY with the JSON object, no additional text."
        )


def _check_json_type(value: Any, json_type: str) -> bool:
    """Check if a Python value matches a JSON schema type.

    Args:
        value: Python value to check.
        json_type: JSON schema type string.

    Returns:
        True if the value matches the expected JSON type.
    """
    type_map: dict[str, type | tuple[type, ...]] = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }
    expected_python_type = type_map.get(json_type)
    if expected_python_type is None:
        return True  # Unknown type — pass

    # bool is a subclass of int in Python; distinguish them
    if json_type == "integer" and isinstance(value, bool):
        return False
    if json_type == "boolean" and not isinstance(value, bool):
        return False

    return isinstance(value, expected_python_type)
