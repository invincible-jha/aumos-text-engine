# Security Policy

## Reporting a Vulnerability

The AumOS platform team takes security vulnerabilities seriously. We appreciate your
efforts to responsibly disclose your findings.

**Please do NOT report security vulnerabilities through public GitHub issues.**

### How to Report

Email your findings to: **security@aumos.io**

Include in your report:
- A description of the vulnerability and its potential impact
- Steps to reproduce the issue
- Any proof-of-concept code (if applicable)
- Your recommended fix (if you have one)

You will receive an acknowledgment within **48 hours** and a detailed response
within **5 business days**.

## Scope

The following are in scope for security reports for the Text Engine specifically:

- PII leakage in synthetic text output (PII not fully replaced)
- Tenant isolation violations (one tenant accessing another's documents)
- LLM prompt injection enabling data exfiltration
- Fine-tuning data poisoning attacks
- Model artifact integrity (malicious weights injection)
- Storage access control bypass (unauthorized MinIO access)
- Differential privacy budget bypass
- SQL injection or other injection attacks
- Remote code execution via model loading
- Sensitive data exposure (PII, document contents, fine-tuning corpus)

The following are out of scope:
- Denial of service attacks
- Social engineering of AumOS staff
- Physical security issues
- Vulnerabilities in Presidio, spaCy, or Transformers themselves
- Issues in repositories not owned by AumOS Enterprise

## Response Timeline

| Stage | Timeline |
|-------|----------|
| Acknowledgment | Within 48 hours |
| Initial assessment | Within 5 business days |
| Status update | Every 7 days during investigation |
| Fix deployment (critical) | Within 7 days of confirmation |
| Fix deployment (high) | Within 30 days of confirmation |
| Fix deployment (medium/low) | Next scheduled release |

## Disclosure Policy

- We follow a **90-day coordinated disclosure** policy
- We will notify you when the fix is deployed
- We will credit you in our release notes (unless you prefer anonymity)
- We ask that you do not publicly disclose the vulnerability until we have released a fix

## Security Best Practices for Contributors

When contributing to this repository:

1. Never commit secrets, API keys, or credentials (even test credentials)
2. Validate all inputs at system boundaries using Pydantic
3. Never log PII or sensitive document content
4. Use parameterized queries — never string concatenation in SQL
5. Verify PII replacement is complete before returning synthetic text
6. Never allow model loading from untrusted sources
7. Ensure tenant isolation on all storage operations
8. Check dependency licenses and security advisories before adding packages
