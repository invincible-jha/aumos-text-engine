"""Context-aware PII entity replacement adapter.

Replaces detected PII entities with plausible fake values that preserve
format, context, and consistency. Names → realistic fake names, dates →
format-preserving shifted dates, emails → valid fake emails, etc.
"""

from __future__ import annotations

import hashlib
import random
import re
import string
from datetime import datetime, timedelta
from typing import Any

import structlog
from aumos_common.logging import get_logger

from aumos_text_engine.api.schemas import PIIEntity, PIIReplaceResult

# Fake name pools (culturally diverse)
_FIRST_NAMES = [
    "Alexandra", "Benjamin", "Catherine", "Daniel", "Elena",
    "Franklin", "Grace", "Henry", "Isabella", "James",
    "Katherine", "Liam", "Maya", "Nathan", "Olivia",
    "Patrick", "Quinn", "Rachel", "Samuel", "Taylor",
    "Uma", "Victor", "Wendy", "Xavier", "Yasmin",
    "Zachary", "Aria", "Blake", "Carmen", "Devon",
]

_LAST_NAMES = [
    "Anderson", "Brown", "Chen", "Davis", "Evans",
    "Franklin", "Garcia", "Harris", "Ingram", "Johnson",
    "Kim", "Lewis", "Miller", "Nelson", "O'Brien",
    "Patel", "Quinn", "Roberts", "Smith", "Thompson",
    "Underwood", "Vasquez", "Williams", "Xu", "Young",
    "Zhang", "Alvarez", "Brooks", "Cooper", "Dixon",
]

_DOMAINS = [
    "example.com", "testmail.org", "synthetic.net", "fakeemail.io",
    "placeholder.com", "demo.org", "sample.net", "noreply.io",
]

_COMPANIES = [
    "Acme Corporation", "Global Industries", "TechVentures LLC",
    "Premier Solutions", "Horizon Group", "Summit Enterprises",
    "Atlas Partners", "Nexus Consulting", "Pinnacle Services",
    "Meridian Holdings", "Vertex Systems", "Apex Dynamics",
]

_CITIES = [
    "Springfield", "Riverdale", "Lakewood", "Hillcrest", "Maplewood",
    "Fairview", "Clearwater", "Brookfield", "Oakdale", "Westfield",
]

_STATES = [
    "Alabama", "Alaska", "Arizona", "California", "Colorado",
    "Florida", "Georgia", "Illinois", "New York", "Texas",
]


class ContextAwareEntityReplacer:
    """Replaces PII entities with context-preserving fake values.

    Maintains a replacement mapping for consistency — the same original value
    always maps to the same fake value within a single document processing.
    This ensures entity coherence (e.g., "John Smith" always becomes the same
    fake name throughout the document).

    Attributes:
        _replacement_mapping: Dict from original PII value to fake replacement.
        _rng: Seeded random number generator for deterministic fakes.
        _log: Structured logger.
    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the entity replacer.

        Args:
            seed: Optional random seed for reproducible replacements.
        """
        self._replacement_mapping: dict[str, str] = {}
        self._rng = random.Random(seed)
        self._log: structlog.BoundLogger = get_logger(__name__)

    async def replace(
        self,
        text: str,
        entities: list[PIIEntity],
        strategy: str = "entity_aware",
    ) -> PIIReplaceResult:
        """Replace all PII entities in text with fake values.

        Processes entities from right to left (by position) to preserve
        character offsets when substituting values of different lengths.

        Args:
            text: Original text containing PII.
            entities: Detected PII entities sorted by start position.
            strategy: Replacement strategy:
                - entity_aware: Context-preserving fake values (default)
                - mask: Replace with [ENTITY_TYPE] placeholders
                - random: Random alphanumeric replacement

        Returns:
            PIIReplaceResult with anonymized text and replacement details.
        """
        if not entities:
            return PIIReplaceResult(
                anonymized_text=text,
                entities=[],
                replacement_mapping={},
            )

        # Sort entities by position (reversed for right-to-left substitution)
        sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)

        result_text = text
        processed_entities: list[PIIEntity] = []

        for entity in sorted_entities:
            original_value = text[entity.start : entity.end]

            # Check consistency cache
            if original_value in self._replacement_mapping:
                fake_value = self._replacement_mapping[original_value]
            else:
                fake_value = self._generate_replacement(
                    entity_type=entity.entity_type,
                    original_value=original_value,
                    strategy=strategy,
                )
                self._replacement_mapping[original_value] = fake_value

            # Substitute in text
            result_text = result_text[: entity.start] + fake_value + result_text[entity.end :]

            processed_entities.append(
                PIIEntity(
                    entity_type=entity.entity_type,
                    value=original_value,
                    replacement=fake_value,
                    start=entity.start,
                    end=entity.end,
                    confidence=entity.confidence,
                )
            )

        # Restore original order
        processed_entities.sort(key=lambda e: e.start)

        self._log.debug(
            "entity replacement complete",
            entities_replaced=len(processed_entities),
        )

        return PIIReplaceResult(
            anonymized_text=result_text,
            entities=processed_entities,
            replacement_mapping=dict(self._replacement_mapping),
        )

    def _generate_replacement(
        self,
        entity_type: str,
        original_value: str,
        strategy: str,
    ) -> str:
        """Generate a fake replacement for a single entity.

        Routes to type-specific generators for entity_aware strategy,
        or falls back to masking/random for other strategies.

        Args:
            entity_type: PII entity type string.
            original_value: Original PII text to replace.
            strategy: Replacement strategy name.

        Returns:
            Fake replacement string.
        """
        if strategy == "mask":
            return f"[{entity_type}]"

        if strategy == "random":
            return "".join(self._rng.choices(string.ascii_letters, k=len(original_value)))

        # entity_aware strategy — type-specific generators
        generators: dict[str, Any] = {
            "PERSON": self._fake_person_name,
            "EMAIL_ADDRESS": self._fake_email,
            "PHONE_NUMBER": self._fake_phone,
            "CREDIT_CARD": self._fake_credit_card,
            "US_SSN": self._fake_ssn,
            "IP_ADDRESS": self._fake_ip,
            "DATE_TIME": self._fake_date,
            "LOCATION": self._fake_location,
            "ORGANIZATION": self._fake_organization,
            "URL": self._fake_url,
            "US_BANK_NUMBER": self._fake_bank_account,
            "IBAN_CODE": self._fake_iban,
        }

        generator = generators.get(entity_type)
        if generator:
            return generator(original_value)

        # Default: mask with entity type for unknown types
        return f"[{entity_type}]"

    def _fake_person_name(self, original: str) -> str:
        """Generate a realistic fake full name.

        Args:
            original: Original name value (unused but kept for API consistency).

        Returns:
            Fake full name string.
        """
        first = self._rng.choice(_FIRST_NAMES)
        last = self._rng.choice(_LAST_NAMES)
        # Preserve format: if original has comma (Last, First), mirror it
        if "," in original:
            return f"{last}, {first}"
        return f"{first} {last}"

    def _fake_email(self, original: str) -> str:
        """Generate a fake email address preserving local-part length.

        Args:
            original: Original email address.

        Returns:
            Fake email at a synthetic domain.
        """
        local_part = "".join(self._rng.choices(string.ascii_lowercase, k=8))
        domain = self._rng.choice(_DOMAINS)
        return f"{local_part}@{domain}"

    def _fake_phone(self, original: str) -> str:
        """Generate a fake phone number preserving format.

        Detects common patterns (US, international) and generates
        a fake number with the same format.

        Args:
            original: Original phone number.

        Returns:
            Fake phone number string.
        """
        # Extract format by replacing digits with random digits
        digits = re.sub(r"\D", "", original)
        fake_digits = "".join(str(self._rng.randint(0, 9)) for _ in digits)

        result = original
        digit_idx = 0
        for i, char in enumerate(result):
            if char.isdigit() and digit_idx < len(fake_digits):
                result = result[:i] + fake_digits[digit_idx] + result[i + 1 :]
                digit_idx += 1
        return result

    def _fake_credit_card(self, original: str) -> str:
        """Generate a Luhn-valid fake credit card number.

        Args:
            original: Original credit card number.

        Returns:
            Fake Luhn-valid card number with the same formatting.
        """
        # Generate 15 random digits + compute Luhn check digit
        prefix = [self._rng.randint(0, 9) for _ in range(15)]
        check_digit = self._luhn_check_digit(prefix)
        digits = prefix + [check_digit]
        fake_number = "".join(map(str, digits))

        # Preserve original formatting (spaces/dashes)
        result = ""
        digit_idx = 0
        for char in original:
            if char.isdigit() and digit_idx < len(fake_number):
                result += fake_number[digit_idx]
                digit_idx += 1
            else:
                result += char
        return result

    def _luhn_check_digit(self, digits: list[int]) -> int:
        """Compute Luhn check digit for a list of digits.

        Args:
            digits: List of integer digits (without check digit).

        Returns:
            Luhn check digit (0-9).
        """
        total = 0
        for i, digit in enumerate(reversed(digits)):
            if i % 2 == 0:
                doubled = digit * 2
                total += doubled - 9 if doubled > 9 else doubled
            else:
                total += digit
        return (10 - (total % 10)) % 10

    def _fake_ssn(self, original: str) -> str:
        """Generate a fake US SSN with the same format.

        Args:
            original: Original SSN string.

        Returns:
            Fake SSN string preserving separators.
        """
        parts = re.findall(r"\d+", original)
        fake_parts = [str(self._rng.randint(100, 999)), str(self._rng.randint(10, 99)), str(self._rng.randint(1000, 9999))]
        separator = "-" if "-" in original else (" " if " " in original else "")
        return separator.join(fake_parts[:len(parts)])

    def _fake_ip(self, original: str) -> str:
        """Generate a fake IP address (private range).

        Args:
            original: Original IP address string.

        Returns:
            Fake private IP address (192.168.x.x).
        """
        return f"192.168.{self._rng.randint(1, 254)}.{self._rng.randint(1, 254)}"

    def _fake_date(self, original: str) -> str:
        """Generate a date shifted by a random offset preserving format.

        Shifts the original date by a random number of days to preserve
        temporal relationships while anonymizing specific dates.

        Args:
            original: Original date string in any common format.

        Returns:
            Fake date string (format-preserving best-effort).
        """
        # Try to parse common date formats
        formats = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y"]
        shift_days = self._rng.randint(-365 * 5, 365 * 5)

        for fmt in formats:
            try:
                parsed = datetime.strptime(original.strip(), fmt)
                fake_date = parsed + timedelta(days=shift_days)
                return fake_date.strftime(fmt)
            except ValueError:
                continue

        # Fallback: return a generic fake date
        return f"{self._rng.randint(2000, 2020)}-{self._rng.randint(1, 12):02d}-{self._rng.randint(1, 28):02d}"

    def _fake_location(self, original: str) -> str:
        """Generate a fake city/location name.

        Args:
            original: Original location value.

        Returns:
            Fake city name from the predefined list.
        """
        return self._rng.choice(_CITIES)

    def _fake_organization(self, original: str) -> str:
        """Generate a fake company/organization name.

        Args:
            original: Original organization name.

        Returns:
            Fake company name.
        """
        return self._rng.choice(_COMPANIES)

    def _fake_url(self, original: str) -> str:
        """Generate a fake URL preserving protocol and structure.

        Args:
            original: Original URL string.

        Returns:
            Fake URL at a synthetic domain.
        """
        path_len = self._rng.randint(4, 12)
        path = "".join(self._rng.choices(string.ascii_lowercase, k=path_len))
        protocol = "https" if "https" in original else "http"
        return f"{protocol}://synthetic-{self._rng.randint(1000, 9999)}.example.com/{path}"

    def _fake_bank_account(self, original: str) -> str:
        """Generate a fake bank account number preserving length.

        Args:
            original: Original bank account number.

        Returns:
            Fake account number with same digit count.
        """
        digit_count = sum(1 for c in original if c.isdigit())
        return "".join(str(self._rng.randint(0, 9)) for _ in range(digit_count))

    def _fake_iban(self, original: str) -> str:
        """Generate a fake IBAN preserving country code and length.

        Args:
            original: Original IBAN string.

        Returns:
            Fake IBAN with same country code and length.
        """
        # Preserve country code (first 2 chars) + fake remaining digits
        if len(original) >= 2:
            country = original[:2].upper()
            rest_len = len(original) - 2
            rest = "".join(
                str(self._rng.randint(0, 9)) if c.isdigit() else self._rng.choice(string.ascii_uppercase)
                for c in original[2:]
            )
            return f"{country}{rest}"
        return original

    def get_replacement_mapping(self) -> dict[str, str]:
        """Return the original→replacement mapping accumulated so far.

        Returns:
            Dict mapping original PII values to their fake replacements.
        """
        return dict(self._replacement_mapping)
