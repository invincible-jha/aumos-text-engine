"""Financial domain text generator.

Generates synthetic financial documents including reports, risk assessments,
regulatory filings, market analysis, and compliance documentation.
All outputs use fictional financial figures and entity names.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

import structlog
from aumos_common.logging import get_logger

from aumos_text_engine.api.schemas import GenerationConfig

logger: structlog.BoundLogger = get_logger(__name__)

# Financial report types
_REPORT_TYPES = [
    "Annual Report (10-K)",
    "Quarterly Report (10-Q)",
    "Earnings Release",
    "Investor Presentation",
    "Management Discussion & Analysis",
    "Auditor's Report",
    "Board Risk Committee Report",
]

# Risk categories (Basel III aligned)
_RISK_CATEGORIES = [
    "Credit Risk",
    "Market Risk",
    "Operational Risk",
    "Liquidity Risk",
    "Interest Rate Risk",
    "Counterparty Credit Risk",
    "Model Risk",
    "Cyber Risk",
    "Reputational Risk",
    "Regulatory Compliance Risk",
]

# Regulatory bodies and frameworks
_REGULATORY_FRAMEWORKS: dict[str, list[str]] = {
    "SEC": ["10-K", "10-Q", "8-K", "S-1", "DEF 14A"],
    "FINRA": ["Rule 4210", "Rule 4370", "Rule 3110"],
    "Basel": ["Basel III Pillar 3", "ICAAP", "ILAAP"],
    "FDIC": ["Call Report", "FFIEC 009", "Resolution Plan"],
    "FSB": ["SIFI Disclosure", "Climate Risk Report"],
}

# Financial metric names for narrative generation
_FINANCIAL_METRICS = [
    "Revenue", "EBITDA", "Net Income", "Operating Cash Flow",
    "Return on Equity (ROE)", "Return on Assets (ROA)",
    "Debt-to-Equity Ratio", "Current Ratio", "Quick Ratio",
    "Net Interest Margin", "Tier 1 Capital Ratio", "Loan Loss Provisions",
    "Assets Under Management (AUM)", "Net Asset Value (NAV)",
    "Price-to-Earnings Ratio", "Price-to-Book Ratio",
]


class FinancialTextGenerator:
    """Generates synthetic financial documents using LLM inference.

    Produces analytically rigorous synthetic financial reports, risk narratives,
    regulatory filings, and market analysis with domain-appropriate metrics
    and regulatory language. All financial figures are fictional.

    Attributes:
        _llm_client: Configured LLM client for generation.
        _template_manager: Prompt template manager.
        _rng: Random number generator for synthetic figures.
        _log: Structured logger.
    """

    def __init__(
        self,
        llm_client: Any,
        template_manager: Any,
        seed: int | None = None,
    ) -> None:
        """Initialize the FinancialTextGenerator.

        Args:
            llm_client: UnifiedLLMClient instance for inference.
            template_manager: PromptTemplateManager for template access.
            seed: Optional random seed for reproducible generation.
        """
        self._llm_client = llm_client
        self._template_manager = template_manager
        self._rng = random.Random(seed)
        self._log: structlog.BoundLogger = get_logger(__name__)

    async def generate_financial_report_section(
        self,
        report_section: str,
        report_type: str = "Annual Report (10-K)",
        company_type: str = "diversified financial services holding company",
        fiscal_period: str = "Fiscal Year 2024",
        key_metrics: str = "",
        market_conditions: str = "moderately favorable economic environment",
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate a section of a synthetic financial report.

        Args:
            report_section: Section name (e.g. "Management Discussion & Analysis").
            report_type: Type of financial report.
            company_type: Description of the fictional company type.
            fiscal_period: Reporting period.
            key_metrics: Comma-separated metrics to address.
            market_conditions: Description of market environment for context.
            config: LLM generation parameters.

        Returns:
            Formatted financial report section text.
        """
        effective_config = config or GenerationConfig(temperature=0.3, max_tokens=2048)
        effective_metrics = key_metrics or ", ".join(self._rng.sample(_FINANCIAL_METRICS, 5))

        variables = {
            "report_section": report_section,
            "report_type": report_type,
            "company_type": company_type,
            "fiscal_period": fiscal_period,
            "key_metrics": effective_metrics,
            "market_conditions": market_conditions,
        }

        try:
            prompt = self._template_manager.render("financial.report.v1", variables)
        except (KeyError, ValueError):
            prompt = (
                f"Generate a synthetic {report_section} section for a {report_type}.\n"
                f"Company type: {company_type}\n"
                f"Fiscal period: {fiscal_period}\n"
                f"Key metrics to address: {effective_metrics}\n"
                f"Market conditions: {market_conditions}\n\n"
                "Use entirely fictional company name (e.g., Meridian Financial Holdings Corp.). "
                "Include realistic but synthetic financial figures with year-over-year comparisons. "
                "Use formal financial reporting language (MD&A style). "
                "Include forward-looking statements disclaimer."
            )

        log = self._log.bind(report_section=report_section, report_type=report_type)
        log.info("generating financial report section")
        generated = await self._llm_client.generate(prompt, effective_config)

        log.info("financial report section generated", length=len(generated))
        return generated

    async def generate_risk_assessment_narrative(
        self,
        risk_type: str,
        business_unit: str = "Corporate Banking Division",
        risk_factors: list[str] | None = None,
        mitigation_strategies: list[str] | None = None,
        risk_rating: str = "Medium",
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate a synthetic risk assessment narrative.

        Args:
            risk_type: Category of risk being assessed.
            business_unit: Business unit subject to the risk.
            risk_factors: Key risk drivers and contributing factors.
            mitigation_strategies: Controls and mitigants in place.
            risk_rating: Overall risk rating (Low|Medium|High|Critical).
            config: LLM generation parameters.

        Returns:
            Risk assessment narrative text following ERM framework.
        """
        effective_config = config or GenerationConfig(temperature=0.25, max_tokens=1500)

        default_factors = [
            f"Concentration in {business_unit} portfolio",
            "Macroeconomic sensitivity",
            "Regulatory capital requirements",
        ]
        default_mitigants = [
            "Portfolio diversification limits",
            "Stress testing and scenario analysis",
            "Regular board risk committee review",
        ]

        effective_factors = risk_factors or default_factors
        effective_mitigants = mitigation_strategies or default_mitigants

        factors_str = "\n".join(f"- {f}" for f in effective_factors)
        mitigants_str = "\n".join(f"- {m}" for m in effective_mitigants)

        variables = {
            "risk_type": risk_type,
            "business_unit": business_unit,
            "risk_factors": factors_str,
            "mitigation_strategies": mitigants_str,
            "risk_rating": risk_rating,
        }

        try:
            prompt = self._template_manager.render("financial.risk_narrative.v1", variables)
        except (KeyError, ValueError):
            prompt = (
                f"Generate a synthetic {risk_type} risk assessment narrative.\n"
                f"Business unit: {business_unit}\n"
                f"Risk factors:\n{factors_str}\n"
                f"Mitigation strategies:\n{mitigants_str}\n"
                f"Overall risk rating: {risk_rating}\n\n"
                "Format with sections: Executive Summary, Risk Identification, "
                "Risk Quantification, Current Controls, Residual Risk Assessment, "
                "Management Action Plan. "
                "Use Basel III and COSO ERM framework language. "
                "Include fictional but realistic quantitative metrics."
            )

        log = self._log.bind(risk_type=risk_type, risk_rating=risk_rating)
        log.info("generating risk assessment narrative")
        generated = await self._llm_client.generate(prompt, effective_config)

        log.info("risk assessment narrative generated", length=len(generated))
        return generated

    async def generate_regulatory_filing_text(
        self,
        filing_type: str,
        filing_form: str = "10-K",
        regulatory_body: str = "SEC",
        subject_matter: str = "",
        disclosure_period: str = "Fiscal Year Ended December 31, 2024",
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate synthetic regulatory filing text.

        Args:
            filing_type: Type of disclosure (e.g. "Risk Factors", "Legal Proceedings").
            filing_form: Form number (e.g. "10-K", "10-Q", "8-K").
            regulatory_body: Regulatory authority (SEC, FINRA, FDIC, etc.).
            subject_matter: Specific subject matter for the disclosure.
            disclosure_period: Reporting period.
            config: LLM generation parameters.

        Returns:
            Regulatory filing text following disclosure requirements.
        """
        effective_config = config or GenerationConfig(temperature=0.2, max_tokens=2048)
        effective_subject = subject_matter or f"Material business developments for {disclosure_period}"

        variables = {
            "filing_type": filing_type,
            "filing_form": filing_form,
            "regulatory_body": regulatory_body,
            "subject_matter": effective_subject,
            "disclosure_period": disclosure_period,
        }

        try:
            prompt = self._template_manager.render("financial.regulatory_filing.v1", variables)
        except (KeyError, ValueError):
            prompt = (
                f"Generate synthetic {filing_type} language for a {filing_form} filing.\n"
                f"Regulatory body: {regulatory_body}\n"
                f"Subject matter: {effective_subject}\n"
                f"Disclosure period: {disclosure_period}\n\n"
                "Use formal SEC/regulatory disclosure language. "
                "Include appropriate risk factors, material information disclosures, and "
                "standard cautionary language for forward-looking statements. "
                "Use a fictional registrant (e.g., Apex Financial Holdings, Inc., CIK: XXXXXXXX). "
                "Follow Regulation S-K and S-X requirements where applicable."
            )

        log = self._log.bind(filing_type=filing_type, filing_form=filing_form)
        log.info("generating regulatory filing text")
        generated = await self._llm_client.generate(prompt, effective_config)

        log.info("regulatory filing text generated", length=len(generated))
        return generated

    async def generate_market_analysis(
        self,
        asset_class: str,
        sector: str,
        analysis_horizon: str = "12-month",
        key_themes: list[str] | None = None,
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate a synthetic market analysis report.

        Args:
            asset_class: Asset class being analyzed (e.g. "Equities", "Fixed Income").
            sector: Industry sector focus.
            analysis_horizon: Time horizon for the analysis.
            key_themes: List of key market themes to address.
            config: LLM generation parameters.

        Returns:
            Market analysis text with synthetic data and projections.
        """
        effective_config = config or GenerationConfig(temperature=0.4, max_tokens=1500)
        default_themes = [
            "Interest rate sensitivity",
            "Geopolitical risk premium",
            "Earnings revision cycle",
            "Regulatory headwinds",
        ]
        effective_themes = key_themes or default_themes
        themes_str = ", ".join(effective_themes)

        prompt = (
            f"Generate a synthetic {analysis_horizon} market analysis for {asset_class} — {sector}.\n"
            f"Key themes: {themes_str}\n\n"
            "Sections: Executive Summary, Market Overview, Key Drivers, "
            "Risks and Opportunities, Valuation Analysis, Outlook and Recommendations. "
            "Include realistic-looking but entirely fictional quantitative data (indices, yields, spreads). "
            "Use institutional research report style. "
            "Include appropriate investment risk disclaimers."
        )

        log = self._log.bind(asset_class=asset_class, sector=sector)
        log.info("generating market analysis")
        generated = await self._llm_client.generate(prompt, effective_config)

        log.info("market analysis generated", length=len(generated))
        return generated

    async def generate_compliance_documentation(
        self,
        compliance_area: str,
        regulatory_framework: str,
        document_type: str = "Policy",
        effective_date: str = "January 1, 2024",
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate synthetic financial compliance documentation.

        Args:
            compliance_area: Compliance subject (e.g. "AML", "GDPR", "MiFID II").
            regulatory_framework: Governing regulatory framework.
            document_type: Type of compliance document (Policy|Procedure|Control).
            effective_date: Policy effective date.
            config: LLM generation parameters.

        Returns:
            Compliance documentation text.
        """
        effective_config = config or GenerationConfig(temperature=0.2, max_tokens=2048)

        prompt = (
            f"Generate a synthetic {compliance_area} {document_type}.\n"
            f"Regulatory framework: {regulatory_framework}\n"
            f"Document type: {document_type}\n"
            f"Effective date: {effective_date}\n\n"
            "Include: Purpose, Scope, Policy Statement, Responsibilities, "
            "Procedures, Controls, Monitoring and Testing, Exceptions, "
            "Related Documents, Revision History. "
            "Use a fictional financial institution (e.g., Pinnacle Bank & Trust N.A.). "
            "Cite actual regulatory requirements but apply them to the fictional entity. "
            "Follow financial services compliance documentation standards."
        )

        log = self._log.bind(compliance_area=compliance_area, document_type=document_type)
        log.info("generating compliance documentation")
        generated = await self._llm_client.generate(prompt, effective_config)

        log.info("compliance documentation generated", length=len(generated))
        return generated

    async def generate_batch(
        self,
        document_specs: list[dict[str, Any]],
        config: GenerationConfig | None = None,
    ) -> list[str]:
        """Generate multiple financial documents concurrently.

        Args:
            document_specs: List of dicts with "type" and type-specific parameters.
            config: Shared generation config.

        Returns:
            List of generated financial document strings.
        """
        tasks = []
        for spec in document_specs:
            doc_type = spec.get("type", "report_section")
            if doc_type == "report_section":
                tasks.append(self.generate_financial_report_section(
                    report_section=spec.get("report_section", "Management Discussion & Analysis"),
                    report_type=spec.get("report_type", "Annual Report (10-K)"),
                    company_type=spec.get("company_type", "diversified financial services holding company"),
                    fiscal_period=spec.get("fiscal_period", "Fiscal Year 2024"),
                    key_metrics=spec.get("key_metrics", ""),
                    market_conditions=spec.get("market_conditions", "moderately favorable economic environment"),
                    config=config,
                ))
            elif doc_type == "risk_assessment":
                tasks.append(self.generate_risk_assessment_narrative(
                    risk_type=spec.get("risk_type", "Credit Risk"),
                    business_unit=spec.get("business_unit", "Corporate Banking Division"),
                    risk_factors=spec.get("risk_factors"),
                    mitigation_strategies=spec.get("mitigation_strategies"),
                    risk_rating=spec.get("risk_rating", "Medium"),
                    config=config,
                ))
            elif doc_type == "regulatory_filing":
                tasks.append(self.generate_regulatory_filing_text(
                    filing_type=spec.get("filing_type", "Risk Factors"),
                    filing_form=spec.get("filing_form", "10-K"),
                    regulatory_body=spec.get("regulatory_body", "SEC"),
                    subject_matter=spec.get("subject_matter", ""),
                    disclosure_period=spec.get("disclosure_period", "Fiscal Year Ended December 31, 2024"),
                    config=config,
                ))
            elif doc_type == "market_analysis":
                tasks.append(self.generate_market_analysis(
                    asset_class=spec.get("asset_class", "Equities"),
                    sector=spec.get("sector", "Financial Services"),
                    analysis_horizon=spec.get("analysis_horizon", "12-month"),
                    key_themes=spec.get("key_themes"),
                    config=config,
                ))
            else:
                tasks.append(self.generate_compliance_documentation(
                    compliance_area=spec.get("compliance_area", "AML/BSA"),
                    regulatory_framework=spec.get("regulatory_framework", "Bank Secrecy Act"),
                    document_type=spec.get("document_type", "Policy"),
                    effective_date=spec.get("effective_date", "January 1, 2024"),
                    config=config,
                ))

        return list(await asyncio.gather(*tasks))
