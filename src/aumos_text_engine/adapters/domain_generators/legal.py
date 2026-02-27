"""Legal domain text generator.

Generates synthetic legal documents including contract clauses, legal briefs,
regulatory compliance text, case summaries, and legal correspondence.
All outputs use fictional party names and do not contain real legal advice.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from aumos_common.logging import get_logger

from aumos_text_engine.api.schemas import GenerationConfig

logger: structlog.BoundLogger = get_logger(__name__)

# Legal terminology consistency glossary
_LEGAL_TERMS: dict[str, str] = {
    "agreement": "Agreement",
    "party": "Party",
    "parties": "Parties",
    "hereinafter": "hereinafter",
    "whereof": "whereof",
    "whereas": "WHEREAS",
    "notwithstanding": "notwithstanding",
    "indemnify": "indemnify, defend, and hold harmless",
    "governing law": "Governing Law and Jurisdiction",
    "force majeure": "Force Majeure",
}

# Common jurisdictions for synthetic documents
_JURISDICTIONS = [
    "the State of Delaware, United States",
    "the State of New York, United States",
    "England and Wales",
    "the Province of Ontario, Canada",
    "the Commonwealth of Victoria, Australia",
]

# Contract types
_CONTRACT_TYPES = [
    "Software License Agreement",
    "Non-Disclosure Agreement",
    "Master Services Agreement",
    "Statement of Work",
    "Data Processing Agreement",
    "Employment Agreement",
    "Asset Purchase Agreement",
    "Commercial Lease Agreement",
]

# ICD-inspired case types for legal documents
_CASE_TYPES = [
    "breach of contract",
    "intellectual property infringement",
    "employment discrimination",
    "product liability",
    "securities fraud",
    "antitrust violation",
    "data privacy violation",
]


class LegalTextGenerator:
    """Generates synthetic legal documents using LLM inference.

    Produces contract clauses, legal briefs, regulatory filings, and
    case summaries with domain-appropriate terminology and structure.
    Maintains legal terminology consistency within a single generation session.

    Attributes:
        _llm_client: Configured LLM client for generation.
        _template_manager: Prompt template manager.
        _terminology_map: Local terminology overrides for consistency.
        _log: Structured logger.
    """

    def __init__(
        self,
        llm_client: Any,
        template_manager: Any,
    ) -> None:
        """Initialize the LegalTextGenerator.

        Args:
            llm_client: UnifiedLLMClient instance for inference.
            template_manager: PromptTemplateManager for template access.
        """
        self._llm_client = llm_client
        self._template_manager = template_manager
        self._terminology_map: dict[str, str] = dict(_LEGAL_TERMS)
        self._log: structlog.BoundLogger = get_logger(__name__)

    async def generate_contract_clause(
        self,
        clause_type: str,
        contract_type: str = "Master Services Agreement",
        jurisdiction: str = "the State of Delaware, United States",
        key_terms: str = "",
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate a synthetic contract clause.

        Args:
            clause_type: Type of clause (e.g. "Limitation of Liability", "IP Ownership").
            contract_type: Type of agreement this clause belongs to.
            jurisdiction: Governing law jurisdiction.
            key_terms: Comma-separated key terms to address in the clause.
            config: LLM generation parameters.

        Returns:
            Formatted contract clause text.
        """
        effective_config = config or GenerationConfig(temperature=0.3, max_tokens=1024)
        log = self._log.bind(clause_type=clause_type, contract_type=contract_type)

        variables = {
            "clause_type": clause_type,
            "contract_type": contract_type,
            "jurisdiction": jurisdiction,
            "key_terms": key_terms or "standard commercial terms",
        }

        try:
            prompt = self._template_manager.render("legal.contract_clause.v1", variables)
        except (KeyError, ValueError):
            # Fallback prompt if template not found
            prompt = (
                f"Draft a {clause_type} clause for a {contract_type} agreement. "
                f"Jurisdiction: {jurisdiction}. "
                f"Use standard legal terminology and formatting. "
                f"Key terms to address: {key_terms or 'standard commercial terms'}. "
                "Do not use real party names or identifying information."
            )

        log.info("generating contract clause")
        generated = await self._llm_client.generate(prompt, effective_config)
        result = self._apply_terminology_consistency(generated)

        log.info("contract clause generated", length=len(result))
        return result

    async def generate_legal_brief(
        self,
        brief_type: str,
        case_type: str,
        legal_issue: str,
        desired_outcome: str,
        court_name: str = "United States District Court for the District of Delaware",
        page_count: int = 5,
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate a synthetic legal brief.

        Args:
            brief_type: Type of brief (e.g. "Motion to Dismiss", "Summary Judgment").
            case_type: Category of the legal case.
            legal_issue: Primary legal question at issue.
            desired_outcome: Relief sought by the filing party.
            court_name: Synthetic court name for the brief header.
            page_count: Approximate length in pages.
            config: LLM generation parameters.

        Returns:
            Formatted legal brief text.
        """
        effective_config = config or GenerationConfig(temperature=0.25, max_tokens=2048)
        log = self._log.bind(brief_type=brief_type, case_type=case_type)

        variables = {
            "brief_type": brief_type,
            "case_type": case_type,
            "court_name": court_name,
            "legal_issue": legal_issue,
            "desired_outcome": desired_outcome,
            "page_count": str(page_count),
        }

        try:
            prompt = self._template_manager.render("legal.brief.v1", variables)
        except (KeyError, ValueError):
            prompt = (
                f"Generate a synthetic {brief_type} brief for a {case_type} case.\n"
                f"Court: {court_name}\n"
                f"Legal issue: {legal_issue}\n"
                f"Desired outcome: {desired_outcome}\n"
                f"Length: approximately {page_count} pages.\n"
                "Use fictional parties (Plaintiff Corp. and Defendant Inc.). "
                "Follow standard legal brief formatting with sections: "
                "Introduction, Statement of Facts, Argument, Conclusion."
            )

        log.info("generating legal brief")
        generated = await self._llm_client.generate(prompt, effective_config)
        result = self._apply_terminology_consistency(generated)

        log.info("legal brief generated", length=len(result))
        return result

    async def generate_regulatory_compliance_text(
        self,
        document_type: str,
        regulation: str,
        sector: str,
        scope: str,
        date_context: str = "effective January 1, 2024",
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate regulatory compliance documentation.

        Args:
            document_type: Type of compliance document (e.g. "Policy", "Procedure").
            regulation: Applicable regulation (e.g. "GDPR", "SOC 2 Type II").
            sector: Industry sector for context.
            scope: Scope of the compliance requirement.
            date_context: Effective date language.
            config: LLM generation parameters.

        Returns:
            Compliance documentation text.
        """
        effective_config = config or GenerationConfig(temperature=0.2, max_tokens=2048)
        log = self._log.bind(regulation=regulation, document_type=document_type)

        variables = {
            "document_type": document_type,
            "regulation": regulation,
            "sector": sector,
            "scope": scope,
            "date_context": date_context,
        }

        try:
            prompt = self._template_manager.render("legal.compliance.v1", variables)
        except (KeyError, ValueError):
            prompt = (
                f"Generate a synthetic {document_type} for compliance with {regulation}.\n"
                f"Industry sector: {sector}\n"
                f"Scope: {scope}\n"
                f"Effective date context: {date_context}\n"
                "Use fictional organization names. Include specific regulatory citations, "
                "definitions, obligations, controls, and audit requirements."
            )

        log.info("generating compliance document")
        generated = await self._llm_client.generate(prompt, effective_config)
        result = self._apply_terminology_consistency(generated)

        log.info("compliance document generated", length=len(result))
        return result

    async def generate_case_summary(
        self,
        case_type: str,
        outcome: str,
        key_findings: list[str],
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate a synthetic case summary for legal training data.

        Args:
            case_type: Type of legal case.
            outcome: How the case was resolved.
            key_findings: List of key legal findings or holdings.
            config: LLM generation parameters.

        Returns:
            Case summary text with findings and analysis.
        """
        effective_config = config or GenerationConfig(temperature=0.3, max_tokens=1024)
        findings_text = "\n".join(f"- {finding}" for finding in key_findings)

        prompt = (
            f"Generate a synthetic case summary for a {case_type} matter.\n"
            f"Outcome: {outcome}\n"
            f"Key findings:\n{findings_text}\n\n"
            "Format: Case Background, Issues Presented, Holdings, Analysis, Conclusion. "
            "Use fictional case names (Smith v. Jones Corp.) and fictional court citations. "
            "Maintain precise legal language throughout."
        )

        log = self._log.bind(case_type=case_type, outcome=outcome)
        log.info("generating case summary")
        generated = await self._llm_client.generate(prompt, effective_config)
        result = self._apply_terminology_consistency(generated)

        log.info("case summary generated", length=len(result))
        return result

    async def generate_batch(
        self,
        document_specs: list[dict[str, Any]],
        config: GenerationConfig | None = None,
    ) -> list[str]:
        """Generate multiple legal documents concurrently.

        Args:
            document_specs: List of dicts specifying document type and parameters.
                Each dict must have "type" key and type-specific parameters.
            config: Shared generation config for all documents.

        Returns:
            List of generated document strings in the same order as specs.
        """
        tasks = []
        for spec in document_specs:
            doc_type = spec.get("type", "contract_clause")
            if doc_type == "contract_clause":
                tasks.append(self.generate_contract_clause(
                    clause_type=spec.get("clause_type", "General Terms"),
                    contract_type=spec.get("contract_type", "Master Services Agreement"),
                    jurisdiction=spec.get("jurisdiction", "the State of Delaware, United States"),
                    key_terms=spec.get("key_terms", ""),
                    config=config,
                ))
            elif doc_type == "legal_brief":
                tasks.append(self.generate_legal_brief(
                    brief_type=spec.get("brief_type", "Memorandum of Law"),
                    case_type=spec.get("case_type", "breach of contract"),
                    legal_issue=spec.get("legal_issue", "contractual obligations"),
                    desired_outcome=spec.get("desired_outcome", "dismissal"),
                    config=config,
                ))
            elif doc_type == "compliance":
                tasks.append(self.generate_regulatory_compliance_text(
                    document_type=spec.get("document_type", "Policy"),
                    regulation=spec.get("regulation", "SOC 2"),
                    sector=spec.get("sector", "Technology"),
                    scope=spec.get("scope", "enterprise-wide"),
                    config=config,
                ))
            else:
                tasks.append(self.generate_case_summary(
                    case_type=spec.get("case_type", "civil litigation"),
                    outcome=spec.get("outcome", "settlement"),
                    key_findings=spec.get("key_findings", ["parties reached agreement"]),
                    config=config,
                ))

        return list(await asyncio.gather(*tasks))

    def _apply_terminology_consistency(self, text: str) -> str:
        """Normalize legal terminology for consistency within the document.

        Args:
            text: Generated legal text.

        Returns:
            Text with consistent legal terminology applied.
        """
        # Ensure defined terms are consistently capitalized after first use
        result = text
        # Capitalize "Agreement" when used as a defined term reference
        result = result.replace(" the agreement ", " the Agreement ")
        result = result.replace(" this agreement ", " this Agreement ")
        return result
