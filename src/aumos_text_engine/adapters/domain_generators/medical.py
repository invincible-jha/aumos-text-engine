"""Medical domain text generator.

Generates synthetic clinical notes, discharge summaries, medical reports,
and clinical correspondence following medical documentation standards.
All outputs use fictional patient identifiers and HIPAA-compliant patterns.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

import structlog
from aumos_common.logging import get_logger

from aumos_text_engine.api.schemas import GenerationConfig

logger: structlog.BoundLogger = get_logger(__name__)

# Common ICD-10 code ranges for synthetic documents (non-real patient data)
_ICD10_CATEGORIES: dict[str, list[str]] = {
    "cardiovascular": ["I10", "I25.10", "I48.91", "I50.9", "I63.9"],
    "respiratory": ["J18.9", "J44.1", "J45.50", "J96.00", "J20.9"],
    "diabetes": ["E11.9", "E11.65", "E11.40", "E13.9", "E10.9"],
    "musculoskeletal": ["M54.5", "M17.11", "M25.511", "M79.3", "M47.816"],
    "neurological": ["G43.909", "G35", "G47.00", "G89.29", "G20"],
    "gastrointestinal": ["K21.0", "K57.30", "K92.1", "K29.70", "K58.9"],
    "mental_health": ["F32.9", "F41.1", "F43.10", "F31.9", "F90.9"],
}

# Medical specialties
_SPECIALTIES = [
    "Internal Medicine", "Cardiology", "Pulmonology", "Neurology",
    "Orthopedic Surgery", "Gastroenterology", "Psychiatry", "Emergency Medicine",
    "Family Medicine", "Endocrinology", "Rheumatology", "Oncology",
]

# Common medications for synthetic notes (fictional dosages)
_COMMON_MEDICATIONS = [
    "Lisinopril 10 mg daily",
    "Metformin 500 mg twice daily",
    "Atorvastatin 40 mg at bedtime",
    "Levothyroxine 50 mcg daily",
    "Omeprazole 20 mg daily",
    "Amlodipine 5 mg daily",
    "Metoprolol succinate 25 mg daily",
    "Aspirin 81 mg daily",
]

# Vital sign ranges for synthetic notes
_VITAL_RANGES: dict[str, str] = {
    "BP": "118/76 mmHg",
    "HR": "72 bpm",
    "RR": "16 breaths/min",
    "Temp": "98.6°F (37.0°C)",
    "SpO2": "98% on room air",
    "Weight": "68 kg",
    "Height": "170 cm",
    "BMI": "23.5 kg/m²",
}


class MedicalTextGenerator:
    """Generates synthetic medical documents using LLM inference.

    Produces clinically accurate synthetic documents following healthcare
    documentation standards (SOAP notes, discharge summaries, medical reports).
    All patient identifiers are fictional. Uses ICD-10-aware generation.

    Attributes:
        _llm_client: Configured LLM client for generation.
        _template_manager: Prompt template manager.
        _rng: Random number generator for synthetic vitals and codes.
        _log: Structured logger.
    """

    def __init__(
        self,
        llm_client: Any,
        template_manager: Any,
        seed: int | None = None,
    ) -> None:
        """Initialize the MedicalTextGenerator.

        Args:
            llm_client: UnifiedLLMClient instance for inference.
            template_manager: PromptTemplateManager for template access.
            seed: Optional random seed for reproducible generation.
        """
        self._llm_client = llm_client
        self._template_manager = template_manager
        self._rng = random.Random(seed)
        self._log: structlog.BoundLogger = get_logger(__name__)

    async def generate_clinical_note(
        self,
        note_type: str = "progress",
        specialty: str = "Internal Medicine",
        chief_complaint: str = "",
        diagnoses: list[str] | None = None,
        medications: list[str] | None = None,
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate a synthetic clinical note in SOAP format.

        Args:
            note_type: Type of note (progress|admission|consultation|procedure).
            specialty: Clinical specialty context.
            chief_complaint: Primary reason for the visit.
            diagnoses: List of ICD-10 codes or diagnosis descriptions.
            medications: List of current medications.
            config: LLM generation parameters.

        Returns:
            Clinical note in SOAP format with synthetic patient data.
        """
        effective_config = config or GenerationConfig(temperature=0.3, max_tokens=1024)

        effective_complaint = chief_complaint or "routine follow-up"
        effective_diagnoses = diagnoses or [self._rng.choice(
            list(_ICD10_CATEGORIES.values())[self._rng.randint(0, len(_ICD10_CATEGORIES) - 1)]
        )]
        effective_medications = medications or self._rng.sample(
            _COMMON_MEDICATIONS, min(3, len(_COMMON_MEDICATIONS))
        )

        diagnoses_str = ", ".join(effective_diagnoses)
        medications_str = "; ".join(effective_medications)

        variables = {
            "note_type": note_type,
            "specialty": specialty,
            "chief_complaint": effective_complaint,
            "diagnoses": diagnoses_str,
            "medications": medications_str,
        }

        try:
            prompt = self._template_manager.render("medical.clinical_note.v1", variables)
        except (KeyError, ValueError):
            vitals_str = ", ".join(f"{k}: {v}" for k, v in _VITAL_RANGES.items())
            prompt = (
                f"Generate a synthetic {note_type} clinical note for a patient in {specialty}.\n"
                f"Chief complaint: {effective_complaint}\n"
                f"Active diagnoses (ICD-10): {diagnoses_str}\n"
                f"Current medications: {medications_str}\n"
                f"Vitals: {vitals_str}\n\n"
                "Format as SOAP note (Subjective, Objective, Assessment, Plan). "
                "Use a fictional patient name and ID (e.g., Patient: J. Doe, MRN: XXXXXXXX). "
                "Include relevant physical exam findings, lab values, and clinical reasoning. "
                "Do not include any real patient information."
            )

        log = self._log.bind(note_type=note_type, specialty=specialty)
        log.info("generating clinical note")
        generated = await self._llm_client.generate(prompt, effective_config)
        result = self._ensure_hipaa_patterns(generated)

        log.info("clinical note generated", length=len(result))
        return result

    async def generate_discharge_summary(
        self,
        primary_diagnosis: str,
        secondary_diagnoses: list[str] | None = None,
        procedures: list[str] | None = None,
        length_of_stay: int = 3,
        disposition: str = "Home with follow-up",
        followup: str = "Primary care physician in 1 week",
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate a synthetic hospital discharge summary.

        Args:
            primary_diagnosis: Primary admission diagnosis.
            secondary_diagnoses: Additional active diagnoses.
            procedures: Procedures performed during admission.
            length_of_stay: Number of inpatient days.
            disposition: Discharge destination.
            followup: Follow-up instructions and appointments.
            config: LLM generation parameters.

        Returns:
            Complete discharge summary following Joint Commission standards.
        """
        effective_config = config or GenerationConfig(temperature=0.25, max_tokens=2048)

        secondary_str = ", ".join(secondary_diagnoses or ["None"])
        procedures_str = ", ".join(procedures or ["None"])

        variables = {
            "primary_diagnosis": primary_diagnosis,
            "secondary_diagnoses": secondary_str,
            "procedures": procedures_str,
            "length_of_stay": str(length_of_stay),
            "disposition": disposition,
            "followup": followup,
        }

        try:
            prompt = self._template_manager.render("medical.discharge_summary.v1", variables)
        except (KeyError, ValueError):
            prompt = (
                f"Generate a synthetic hospital discharge summary.\n"
                f"Primary diagnosis: {primary_diagnosis}\n"
                f"Secondary diagnoses: {secondary_str}\n"
                f"Procedures performed: {procedures_str}\n"
                f"Length of stay: {length_of_stay} days\n"
                f"Discharge disposition: {disposition}\n"
                f"Follow-up instructions: {followup}\n\n"
                "Include: Patient demographics (fictional), Admission date, Attending physician "
                "(fictional), Hospital course narrative, Medications at discharge, "
                "Condition at discharge, and follow-up instructions. "
                "Use a fictional patient name and MRN. Apply Joint Commission documentation standards."
            )

        log = self._log.bind(primary_diagnosis=primary_diagnosis, los=length_of_stay)
        log.info("generating discharge summary")
        generated = await self._llm_client.generate(prompt, effective_config)
        result = self._ensure_hipaa_patterns(generated)

        log.info("discharge summary generated", length=len(result))
        return result

    async def generate_medical_report(
        self,
        report_type: str,
        findings: list[str],
        impression: str,
        modality: str = "",
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate a synthetic medical report (radiology, pathology, etc.).

        Args:
            report_type: Report category (e.g. "Radiology", "Pathology", "Lab").
            findings: List of clinical findings to include.
            impression: Summary impression/conclusion.
            modality: Imaging modality or test type (e.g. "CT Chest", "CBC").
            config: LLM generation parameters.

        Returns:
            Structured medical report text.
        """
        effective_config = config or GenerationConfig(temperature=0.2, max_tokens=1024)
        findings_text = "\n".join(f"- {f}" for f in findings)

        prompt = (
            f"Generate a synthetic {report_type} report.\n"
            f"Modality/Test: {modality or report_type}\n"
            f"Clinical findings:\n{findings_text}\n"
            f"Impression: {impression}\n\n"
            "Format with standard report sections: Patient Information (fictional), "
            "Clinical History, Technique/Procedure, Findings, Impression. "
            "Use medically precise language. Patient ID must be fictional (e.g., PT-XXXXXXXX)."
        )

        log = self._log.bind(report_type=report_type)
        log.info("generating medical report")
        generated = await self._llm_client.generate(prompt, effective_config)
        result = self._ensure_hipaa_patterns(generated)

        log.info("medical report generated", length=len(result))
        return result

    async def generate_referral_letter(
        self,
        referring_specialty: str,
        receiving_specialty: str,
        reason_for_referral: str,
        relevant_diagnoses: list[str],
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate a synthetic specialist referral letter.

        Args:
            referring_specialty: Specialty of the referring physician.
            receiving_specialty: Specialty of the receiving physician.
            reason_for_referral: Clinical reason for the referral.
            relevant_diagnoses: Active diagnoses relevant to the referral.
            config: LLM generation parameters.

        Returns:
            Formatted referral letter text.
        """
        effective_config = config or GenerationConfig(temperature=0.3, max_tokens=800)
        diagnoses_str = ", ".join(relevant_diagnoses)

        prompt = (
            f"Generate a synthetic specialist referral letter.\n"
            f"From: {referring_specialty} physician\n"
            f"To: {receiving_specialty} specialist\n"
            f"Reason for referral: {reason_for_referral}\n"
            f"Relevant diagnoses: {diagnoses_str}\n\n"
            "Include: Date, referring provider information (fictional), receiving provider "
            "(fictional), patient summary (fictional patient), clinical history, reason for "
            "referral, current medications, and requested evaluation. "
            "Use professional clinical language."
        )

        log = self._log.bind(referring_specialty=referring_specialty, receiving_specialty=receiving_specialty)
        log.info("generating referral letter")
        generated = await self._llm_client.generate(prompt, effective_config)
        result = self._ensure_hipaa_patterns(generated)

        log.info("referral letter generated", length=len(result))
        return result

    async def generate_batch(
        self,
        document_specs: list[dict[str, Any]],
        config: GenerationConfig | None = None,
    ) -> list[str]:
        """Generate multiple medical documents concurrently.

        Args:
            document_specs: List of dicts with "type" and type-specific parameters.
            config: Shared generation config.

        Returns:
            List of generated medical document strings.
        """
        tasks = []
        for spec in document_specs:
            doc_type = spec.get("type", "clinical_note")
            if doc_type == "clinical_note":
                tasks.append(self.generate_clinical_note(
                    note_type=spec.get("note_type", "progress"),
                    specialty=spec.get("specialty", "Internal Medicine"),
                    chief_complaint=spec.get("chief_complaint", ""),
                    diagnoses=spec.get("diagnoses"),
                    medications=spec.get("medications"),
                    config=config,
                ))
            elif doc_type == "discharge_summary":
                tasks.append(self.generate_discharge_summary(
                    primary_diagnosis=spec.get("primary_diagnosis", "Unspecified condition"),
                    secondary_diagnoses=spec.get("secondary_diagnoses"),
                    procedures=spec.get("procedures"),
                    length_of_stay=spec.get("length_of_stay", 3),
                    disposition=spec.get("disposition", "Home with follow-up"),
                    followup=spec.get("followup", "PCP in 1 week"),
                    config=config,
                ))
            elif doc_type == "referral":
                tasks.append(self.generate_referral_letter(
                    referring_specialty=spec.get("referring_specialty", "Family Medicine"),
                    receiving_specialty=spec.get("receiving_specialty", "Cardiology"),
                    reason_for_referral=spec.get("reason_for_referral", "Evaluation and management"),
                    relevant_diagnoses=spec.get("relevant_diagnoses", ["Hypertension"]),
                    config=config,
                ))
            else:
                tasks.append(self.generate_medical_report(
                    report_type=spec.get("report_type", "Radiology"),
                    findings=spec.get("findings", ["No acute findings"]),
                    impression=spec.get("impression", "Normal study"),
                    modality=spec.get("modality", ""),
                    config=config,
                ))

        return list(await asyncio.gather(*tasks))

    def _ensure_hipaa_patterns(self, text: str) -> str:
        """Apply HIPAA-compliant text patterns.

        Ensures the generated text uses fictional identifiers and
        does not contain patterns that could match real PHI formats.
        This is a best-effort sanitization — the LLM prompt instructions
        are the primary control.

        Args:
            text: Generated medical text.

        Returns:
            Text with HIPAA-compliant fictional identifiers.
        """
        import re

        # Replace any 9-digit sequences that look like SSNs
        text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN REDACTED]", text)

        # Replace 10-digit sequences that look like phone numbers if not already formatted
        text = re.sub(
            r"\b(?<!\d)(\d{10})(?!\d)\b",
            lambda m: f"({m.group(1)[:3]}) {m.group(1)[3:6]}-XXXX",
            text,
        )

        return text
