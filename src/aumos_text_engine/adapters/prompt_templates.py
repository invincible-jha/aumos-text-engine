"""Domain-specific prompt template manager.

Maintains a versioned library of prompt templates for legal, medical,
financial, and general text generation. Supports variable substitution,
few-shot example injection, and system/user/assistant message formatting.
"""

from __future__ import annotations

import re
from typing import Any

import structlog
from aumos_common.logging import get_logger

logger: structlog.BoundLogger = get_logger(__name__)

# Domain category constants
DOMAIN_LEGAL = "legal"
DOMAIN_MEDICAL = "medical"
DOMAIN_FINANCIAL = "financial"
DOMAIN_GENERAL = "general"


class PromptTemplate:
    """A versioned prompt template with variable substitution.

    Attributes:
        template_id: Unique identifier for this template.
        version: Semantic version string.
        domain: Domain category.
        description: Human-readable description.
        system_message: System role message template.
        user_message_template: User message with {{variable}} placeholders.
        required_variables: Variables that must be supplied for rendering.
        few_shot_examples: List of (user, assistant) example pairs.
    """

    def __init__(
        self,
        template_id: str,
        version: str,
        domain: str,
        description: str,
        system_message: str,
        user_message_template: str,
        required_variables: list[str] | None = None,
        few_shot_examples: list[dict[str, str]] | None = None,
    ) -> None:
        """Initialize a PromptTemplate.

        Args:
            template_id: Unique identifier for this template.
            version: Semantic version string (e.g. "1.0.0").
            domain: Domain category (legal|medical|financial|general).
            description: Human-readable description of what this generates.
            system_message: Fixed system role message.
            user_message_template: User message with {{var}} placeholders.
            required_variables: Variables required for rendering.
            few_shot_examples: List of {"user": ..., "assistant": ...} dicts.
        """
        self.template_id = template_id
        self.version = version
        self.domain = domain
        self.description = description
        self.system_message = system_message
        self.user_message_template = user_message_template
        self.required_variables: list[str] = required_variables or []
        self.few_shot_examples: list[dict[str, str]] = few_shot_examples or []

    def validate_variables(self, variables: dict[str, Any]) -> list[str]:
        """Check that all required variables are provided.

        Args:
            variables: Variable dict to validate.

        Returns:
            List of missing variable names (empty if all present).
        """
        return [var for var in self.required_variables if var not in variables]

    def render_user_message(self, variables: dict[str, Any]) -> str:
        """Render the user message by substituting {{variable}} placeholders.

        Args:
            variables: Dict mapping variable names to their values.

        Returns:
            Rendered user message string.

        Raises:
            ValueError: If required variables are missing.
        """
        missing = self.validate_variables(variables)
        if missing:
            raise ValueError(f"Missing required template variables: {missing}")

        rendered = self.user_message_template
        for key, value in variables.items():
            rendered = rendered.replace(f"{{{{{key}}}}}", str(value))

        # Check for any unreplaced placeholders
        remaining = re.findall(r"\{\{(\w+)\}\}", rendered)
        if remaining:
            raise ValueError(f"Unresolved template placeholders: {remaining}")

        return rendered

    def to_messages(
        self,
        variables: dict[str, Any],
        include_few_shot: bool = True,
    ) -> list[dict[str, str]]:
        """Build a complete messages list for the LLM.

        Assembles system message, few-shot examples, and the rendered
        user message into the standard messages format.

        Args:
            variables: Template variable values.
            include_few_shot: Whether to inject few-shot examples.

        Returns:
            List of {"role": ..., "content": ...} message dicts.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_message},
        ]

        if include_few_shot and self.few_shot_examples:
            for example in self.few_shot_examples:
                messages.append({"role": "user", "content": example["user"]})
                messages.append({"role": "assistant", "content": example["assistant"]})

        messages.append({"role": "user", "content": self.render_user_message(variables)})
        return messages

    def format_as_single_prompt(
        self,
        variables: dict[str, Any],
        include_few_shot: bool = True,
    ) -> str:
        """Render the template as a single concatenated prompt string.

        Args:
            variables: Template variable values.
            include_few_shot: Whether to inject few-shot examples.

        Returns:
            Single prompt string combining all message parts.
        """
        messages = self.to_messages(variables, include_few_shot)
        parts: list[str] = []
        for message in messages:
            role = message["role"].upper()
            parts.append(f"[{role}]\n{message['content']}")
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Built-in template registry
# ---------------------------------------------------------------------------

_BUILTIN_TEMPLATES: list[PromptTemplate] = [
    # ----- General -----
    PromptTemplate(
        template_id="general.document.v1",
        version="1.0.0",
        domain=DOMAIN_GENERAL,
        description="Generate a generic synthetic document",
        system_message=(
            "You are an expert document generator. Generate realistic, professional "
            "synthetic documents that contain no real personally identifiable information. "
            "Preserve format and structure but use entirely fictional content."
        ),
        user_message_template=(
            "Generate a synthetic {{document_type}} document.\n"
            "Topic: {{topic}}\n"
            "Length: approximately {{word_count}} words.\n"
            "Tone: {{tone}}"
        ),
        required_variables=["document_type", "topic", "word_count", "tone"],
        few_shot_examples=[
            {
                "user": "Generate a synthetic memo document.\nTopic: office supply reorder\nLength: approximately 150 words.\nTone: professional",
                "assistant": (
                    "MEMO\nTo: All Staff\nFrom: Office Manager\nDate: March 15, 2024\n"
                    "Re: Office Supply Reorder\n\nPlease be advised that we will be placing our "
                    "quarterly office supply order next week. If you require any specific items, "
                    "kindly submit your requests to the administrative office by end of business Friday."
                ),
            }
        ],
    ),
    # ----- Legal -----
    PromptTemplate(
        template_id="legal.contract_clause.v1",
        version="1.0.0",
        domain=DOMAIN_LEGAL,
        description="Generate a contract clause for a specified subject",
        system_message=(
            "You are a legal document drafting assistant specializing in commercial contracts. "
            "Generate professionally worded, legally precise contract clauses. "
            "Use standard legal terminology. Do not include real party names or identifying information."
        ),
        user_message_template=(
            "Draft a {{clause_type}} clause for a {{contract_type}} agreement.\n"
            "Jurisdiction: {{jurisdiction}}\n"
            "Key terms: {{key_terms}}\n"
            "Include appropriate cross-references and definitions."
        ),
        required_variables=["clause_type", "contract_type", "jurisdiction", "key_terms"],
    ),
    PromptTemplate(
        template_id="legal.brief.v1",
        version="1.0.0",
        domain=DOMAIN_LEGAL,
        description="Generate a legal brief summary",
        system_message=(
            "You are a senior litigation attorney assistant. Generate concise, well-structured "
            "legal briefs following standard court filing conventions. "
            "Use fictional case details only."
        ),
        user_message_template=(
            "Generate a {{brief_type}} brief for a {{case_type}} case.\n"
            "Court: {{court_name}}\n"
            "Legal issue: {{legal_issue}}\n"
            "Desired outcome: {{desired_outcome}}\n"
            "Approximate length: {{page_count}} pages."
        ),
        required_variables=["brief_type", "case_type", "court_name", "legal_issue", "desired_outcome", "page_count"],
    ),
    PromptTemplate(
        template_id="legal.compliance.v1",
        version="1.0.0",
        domain=DOMAIN_LEGAL,
        description="Generate regulatory compliance documentation",
        system_message=(
            "You are a regulatory compliance expert. Generate accurate, comprehensive compliance "
            "documentation referencing relevant regulations. Use only fictional organization details."
        ),
        user_message_template=(
            "Generate a {{document_type}} for compliance with {{regulation}}.\n"
            "Industry sector: {{sector}}\n"
            "Scope: {{scope}}\n"
            "Effective date context: {{date_context}}"
        ),
        required_variables=["document_type", "regulation", "sector", "scope", "date_context"],
    ),
    # ----- Medical -----
    PromptTemplate(
        template_id="medical.clinical_note.v1",
        version="1.0.0",
        domain=DOMAIN_MEDICAL,
        description="Generate a synthetic clinical note (SOAP format)",
        system_message=(
            "You are a clinical documentation specialist. Generate realistic, medically accurate "
            "synthetic clinical notes in SOAP format. Use only fictional patient details. "
            "Apply appropriate ICD-10 diagnostic codes. Maintain HIPAA-compliant language patterns."
        ),
        user_message_template=(
            "Generate a {{note_type}} clinical note.\n"
            "Specialty: {{specialty}}\n"
            "Chief complaint: {{chief_complaint}}\n"
            "Relevant diagnoses (ICD-10): {{diagnoses}}\n"
            "Medications: {{medications}}"
        ),
        required_variables=["note_type", "specialty", "chief_complaint", "diagnoses", "medications"],
    ),
    PromptTemplate(
        template_id="medical.discharge_summary.v1",
        version="1.0.0",
        domain=DOMAIN_MEDICAL,
        description="Generate a hospital discharge summary",
        system_message=(
            "You are a hospital documentation specialist. Generate complete, accurate synthetic "
            "discharge summaries following Joint Commission standards. "
            "Use only fictional patient and provider details."
        ),
        user_message_template=(
            "Generate a discharge summary for a patient with:\n"
            "Primary diagnosis: {{primary_diagnosis}}\n"
            "Secondary diagnoses: {{secondary_diagnoses}}\n"
            "Procedures performed: {{procedures}}\n"
            "Length of stay: {{length_of_stay}} days\n"
            "Discharge disposition: {{disposition}}\n"
            "Follow-up instructions: {{followup}}"
        ),
        required_variables=[
            "primary_diagnosis", "secondary_diagnoses", "procedures",
            "length_of_stay", "disposition", "followup",
        ],
    ),
    # ----- Financial -----
    PromptTemplate(
        template_id="financial.report.v1",
        version="1.0.0",
        domain=DOMAIN_FINANCIAL,
        description="Generate a financial report section",
        system_message=(
            "You are a financial analyst and reporting specialist. Generate professional, "
            "analytically rigorous synthetic financial report sections. "
            "Use realistic but entirely fictional financial figures and company details."
        ),
        user_message_template=(
            "Generate a {{report_section}} section for a {{report_type}}.\n"
            "Company type: {{company_type}}\n"
            "Fiscal period: {{fiscal_period}}\n"
            "Key metrics to address: {{key_metrics}}\n"
            "Market conditions: {{market_conditions}}"
        ),
        required_variables=[
            "report_section", "report_type", "company_type",
            "fiscal_period", "key_metrics", "market_conditions",
        ],
    ),
    PromptTemplate(
        template_id="financial.risk_narrative.v1",
        version="1.0.0",
        domain=DOMAIN_FINANCIAL,
        description="Generate a risk assessment narrative",
        system_message=(
            "You are a risk management specialist. Generate comprehensive, professional risk "
            "assessment narratives following Basel III and enterprise risk management frameworks. "
            "Use fictional entity details."
        ),
        user_message_template=(
            "Generate a {{risk_type}} risk assessment narrative.\n"
            "Business unit: {{business_unit}}\n"
            "Risk factors: {{risk_factors}}\n"
            "Mitigation strategies: {{mitigation_strategies}}\n"
            "Risk rating: {{risk_rating}}"
        ),
        required_variables=[
            "risk_type", "business_unit", "risk_factors",
            "mitigation_strategies", "risk_rating",
        ],
    ),
    PromptTemplate(
        template_id="financial.regulatory_filing.v1",
        version="1.0.0",
        domain=DOMAIN_FINANCIAL,
        description="Generate regulatory filing text",
        system_message=(
            "You are a regulatory filing specialist with expertise in SEC, FINRA, and banking "
            "regulations. Generate accurate synthetic regulatory filing language. "
            "Use only fictional firm details."
        ),
        user_message_template=(
            "Generate {{filing_type}} language for a {{filing_form}} filing.\n"
            "Regulatory body: {{regulatory_body}}\n"
            "Subject matter: {{subject_matter}}\n"
            "Disclosure period: {{disclosure_period}}"
        ),
        required_variables=[
            "filing_type", "filing_form", "regulatory_body",
            "subject_matter", "disclosure_period",
        ],
    ),
]


class PromptTemplateManager:
    """Registry and renderer for domain-specific prompt templates.

    Maintains a versioned template library with support for:
    - Template lookup by ID or domain
    - Variable substitution with validation
    - Few-shot example injection
    - System/user/assistant message formatting

    Attributes:
        _registry: Dict mapping template_id -> PromptTemplate.
        _log: Structured logger.
    """

    def __init__(self) -> None:
        """Initialize the PromptTemplateManager with built-in templates.

        Args:
            None
        """
        self._registry: dict[str, PromptTemplate] = {}
        self._log: structlog.BoundLogger = get_logger(__name__)

        # Register all built-in templates
        for template in _BUILTIN_TEMPLATES:
            self._registry[template.template_id] = template

        self._log.info("prompt template manager initialized", template_count=len(self._registry))

    def register_template(self, template: PromptTemplate) -> None:
        """Register a custom template in the manager.

        If a template with the same ID already exists, it will be overwritten.

        Args:
            template: PromptTemplate instance to register.

        Returns:
            None
        """
        self._registry[template.template_id] = template
        self._log.debug("template registered", template_id=template.template_id, version=template.version)

    def get_template(self, template_id: str) -> PromptTemplate:
        """Retrieve a template by ID.

        Args:
            template_id: The template identifier to look up.

        Returns:
            The matching PromptTemplate.

        Raises:
            KeyError: If no template with the given ID exists.
        """
        if template_id not in self._registry:
            raise KeyError(f"Prompt template '{template_id}' not found in registry")
        return self._registry[template_id]

    def list_templates(self, domain: str | None = None) -> list[dict[str, str]]:
        """List all registered templates, optionally filtered by domain.

        Args:
            domain: Optional domain filter (legal|medical|financial|general).

        Returns:
            List of dicts with template_id, version, domain, description.
        """
        templates = list(self._registry.values())
        if domain:
            templates = [t for t in templates if t.domain == domain]

        return [
            {
                "template_id": t.template_id,
                "version": t.version,
                "domain": t.domain,
                "description": t.description,
                "required_variables": ", ".join(t.required_variables),
            }
            for t in templates
        ]

    def render(
        self,
        template_id: str,
        variables: dict[str, Any],
        include_few_shot: bool = True,
    ) -> str:
        """Render a template to a single prompt string.

        Args:
            template_id: Template to render.
            variables: Variable values for substitution.
            include_few_shot: Whether to include few-shot examples.

        Returns:
            Rendered prompt string.

        Raises:
            KeyError: If template not found.
            ValueError: If required variables are missing.
        """
        template = self.get_template(template_id)
        rendered = template.format_as_single_prompt(variables, include_few_shot)
        self._log.debug(
            "template rendered",
            template_id=template_id,
            prompt_length=len(rendered),
        )
        return rendered

    def render_messages(
        self,
        template_id: str,
        variables: dict[str, Any],
        include_few_shot: bool = True,
    ) -> list[dict[str, str]]:
        """Render a template to a structured messages list.

        Args:
            template_id: Template to render.
            variables: Variable values for substitution.
            include_few_shot: Whether to include few-shot examples.

        Returns:
            List of {"role": ..., "content": ...} message dicts.

        Raises:
            KeyError: If template not found.
            ValueError: If required variables are missing.
        """
        template = self.get_template(template_id)
        return template.to_messages(variables, include_few_shot)

    def get_default_template_for_domain(self, domain: str) -> PromptTemplate:
        """Return the first registered template for a given domain.

        Args:
            domain: Domain category to look up.

        Returns:
            First matching PromptTemplate.

        Raises:
            KeyError: If no templates exist for the domain.
        """
        for template in self._registry.values():
            if template.domain == domain:
                return template
        raise KeyError(f"No templates registered for domain '{domain}'")

    def validate_template_variables(
        self,
        template_id: str,
        variables: dict[str, Any],
    ) -> list[str]:
        """Check which required variables are missing for a template.

        Args:
            template_id: Template to validate against.
            variables: Proposed variable dict.

        Returns:
            List of missing variable names (empty if all provided).
        """
        template = self.get_template(template_id)
        return template.validate_variables(variables)
