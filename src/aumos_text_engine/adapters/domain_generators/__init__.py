"""Domain-specific text generators for aumos-text-engine.

Provides domain-aware LLM generation for legal, medical, and financial
document synthesis. All generators use fictional content only.
"""

from aumos_text_engine.adapters.domain_generators.financial import FinancialTextGenerator
from aumos_text_engine.adapters.domain_generators.legal import LegalTextGenerator
from aumos_text_engine.adapters.domain_generators.medical import MedicalTextGenerator

__all__ = [
    "LegalTextGenerator",
    "MedicalTextGenerator",
    "FinancialTextGenerator",
]
