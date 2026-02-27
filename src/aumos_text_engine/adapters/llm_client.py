"""Unified LLM client adapter for vLLM, Ollama, and LiteLLM backends.

Wraps multiple local and remote LLM inference providers behind a single
async interface. Implements token counting, rate limiting, retry with
exponential backoff, and structured JSON output parsing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any

import httpx
import structlog
from aumos_common.logging import get_logger

from aumos_text_engine.api.schemas import GenerationConfig

logger: structlog.BoundLogger = get_logger(__name__)

# Default retry settings
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BACKOFF_BASE = 1.5
_DEFAULT_BACKOFF_MAX = 30.0
_DEFAULT_TIMEOUT_SECONDS = 120.0

# Provider identifiers
PROVIDER_VLLM = "vllm"
PROVIDER_OLLAMA = "ollama"
PROVIDER_LITELLM = "litellm"


class RateLimiter:
    """Token-bucket rate limiter for LLM API calls.

    Attributes:
        _capacity: Maximum tokens in the bucket.
        _tokens: Current available tokens.
        _refill_rate: Tokens added per second.
        _last_refill: Timestamp of last refill.
        _lock: Asyncio lock for thread-safe updates.
    """

    def __init__(self, requests_per_minute: int = 60) -> None:
        """Initialize the rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute.
        """
        self._capacity = float(requests_per_minute)
        self._tokens = float(requests_per_minute)
        self._refill_rate = float(requests_per_minute) / 60.0
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary.

        Args:
            None

        Returns:
            None
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(
                self._capacity,
                self._tokens + elapsed * self._refill_rate,
            )
            self._last_refill = now

            if self._tokens < 1.0:
                wait_seconds = (1.0 - self._tokens) / self._refill_rate
                await asyncio.sleep(wait_seconds)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


class UnifiedLLMClient:
    """Async LLM client supporting vLLM, Ollama, and LiteLLM backends.

    Provides a single generate() interface regardless of the underlying
    inference provider. Handles:
    - Provider routing based on model prefix or explicit provider selection
    - Async HTTP calls via httpx
    - Token budget estimation
    - Retry with exponential backoff on transient errors
    - Structured JSON output parsing
    - Rate limiting per provider

    Attributes:
        _vllm_base_url: Base URL for vLLM OpenAI-compatible API.
        _ollama_base_url: Base URL for Ollama API.
        _litellm_base_url: Base URL for LiteLLM proxy.
        _default_provider: Which provider to use when unspecified.
        _default_model: Default model name when none specified in GenerationConfig.
        _http_client: Shared async httpx client.
        _rate_limiters: Per-provider rate limiters.
        _log: Structured logger.
    """

    def __init__(
        self,
        vllm_base_url: str = "http://localhost:8000",
        ollama_base_url: str = "http://localhost:11434",
        litellm_base_url: str = "http://localhost:4000",
        default_provider: str = PROVIDER_VLLM,
        default_model: str = "llama3-8b-instruct",
        requests_per_minute: int = 60,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ) -> None:
        """Initialize the UnifiedLLMClient.

        Args:
            vllm_base_url: Base URL for vLLM OpenAI-compatible server.
            ollama_base_url: Base URL for Ollama server.
            litellm_base_url: Base URL for LiteLLM proxy server.
            default_provider: Which provider to use by default.
            default_model: Default model name.
            requests_per_minute: Rate limit per provider.
            timeout_seconds: HTTP request timeout.
            max_retries: Maximum retry attempts on transient failures.
        """
        self._vllm_base_url = vllm_base_url.rstrip("/")
        self._ollama_base_url = ollama_base_url.rstrip("/")
        self._litellm_base_url = litellm_base_url.rstrip("/")
        self._default_provider = default_provider
        self._default_model = default_model
        self._max_retries = max_retries
        self._timeout = httpx.Timeout(timeout_seconds)
        self._http_client: httpx.AsyncClient | None = None
        self._rate_limiters: dict[str, RateLimiter] = {
            PROVIDER_VLLM: RateLimiter(requests_per_minute),
            PROVIDER_OLLAMA: RateLimiter(requests_per_minute),
            PROVIDER_LITELLM: RateLimiter(requests_per_minute),
        }
        self._log: structlog.BoundLogger = get_logger(__name__)

    async def _get_client(self) -> httpx.AsyncClient:
        """Return the shared async HTTP client, creating it if needed.

        Args:
            None

        Returns:
            Async httpx client instance.
        """
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=self._timeout,
                headers={"Content-Type": "application/json"},
            )
        return self._http_client

    def _resolve_provider(self, model: str | None) -> tuple[str, str]:
        """Resolve provider and model name from the model string.

        Supports prefixed model names:
        - "vllm/llama3-8b" -> provider=vllm, model=llama3-8b
        - "ollama/mistral" -> provider=ollama, model=mistral
        - "litellm/gpt-4" -> provider=litellm, model=gpt-4
        - "llama3-8b" -> provider=default_provider, model=llama3-8b

        Args:
            model: Model identifier, optionally prefixed with provider name.

        Returns:
            Tuple of (provider, model_name).
        """
        effective_model = model or self._default_model

        for provider in (PROVIDER_VLLM, PROVIDER_OLLAMA, PROVIDER_LITELLM):
            prefix = f"{provider}/"
            if effective_model.startswith(prefix):
                return provider, effective_model[len(prefix):]

        return self._default_provider, effective_model

    def estimate_token_count(self, text: str) -> int:
        """Estimate token count for a text string using character heuristics.

        Uses the common 4-characters-per-token approximation. For production
        accuracy, replace with a proper tokenizer (tiktoken, transformers).

        Args:
            text: Input text to estimate.

        Returns:
            Estimated token count.
        """
        return max(1, len(text) // 4)

    async def generate(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> str:
        """Generate text using the configured LLM provider.

        Selects the provider based on model name prefix, applies rate limiting,
        retries on transient errors with exponential backoff.

        Args:
            prompt: Full generation prompt.
            config: Generation parameters including model, temperature, etc.

        Returns:
            Generated text string.

        Raises:
            RuntimeError: If generation fails after all retries.
        """
        provider, model_name = self._resolve_provider(config.model)
        log = self._log.bind(provider=provider, model=model_name)

        for attempt in range(self._max_retries + 1):
            try:
                await self._rate_limiters[provider].acquire()

                if provider == PROVIDER_VLLM:
                    result = await self._generate_vllm(prompt, model_name, config)
                elif provider == PROVIDER_OLLAMA:
                    result = await self._generate_ollama(prompt, model_name, config)
                else:
                    result = await self._generate_litellm(prompt, model_name, config)

                log.info(
                    "generation complete",
                    attempt=attempt + 1,
                    output_tokens=self.estimate_token_count(result),
                )
                return result

            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt == self._max_retries:
                    log.error("generation failed after retries", error=str(exc), attempts=attempt + 1)
                    raise RuntimeError(f"LLM generation failed after {self._max_retries} retries: {exc}") from exc

                backoff = min(
                    _DEFAULT_BACKOFF_BASE ** attempt,
                    _DEFAULT_BACKOFF_MAX,
                )
                log.warning(
                    "transient error, retrying",
                    attempt=attempt + 1,
                    backoff_seconds=backoff,
                    error=str(exc),
                )
                await asyncio.sleep(backoff)

            except httpx.HTTPStatusError as exc:
                # 429 rate limit — always retry; 4xx client errors — don't retry
                if exc.response.status_code == 429 and attempt < self._max_retries:
                    backoff = min(_DEFAULT_BACKOFF_BASE ** (attempt + 2), _DEFAULT_BACKOFF_MAX)
                    log.warning("rate limited by provider", backoff_seconds=backoff)
                    await asyncio.sleep(backoff)
                else:
                    log.error("HTTP error from LLM provider", status=exc.response.status_code)
                    raise RuntimeError(f"LLM provider returned HTTP {exc.response.status_code}") from exc

        raise RuntimeError("LLM generation failed: unreachable code path")

    async def _generate_vllm(
        self,
        prompt: str,
        model: str,
        config: GenerationConfig,
    ) -> str:
        """Generate text via vLLM's OpenAI-compatible API.

        Args:
            prompt: Full generation prompt.
            model: Model name to use.
            config: Generation parameters.

        Returns:
            Generated text from vLLM.
        """
        client = await self._get_client()
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }

        response = await client.post(
            f"{self._vllm_base_url}/v1/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"])

    async def _generate_ollama(
        self,
        prompt: str,
        model: str,
        config: GenerationConfig,
    ) -> str:
        """Generate text via Ollama's native API.

        Args:
            prompt: Full generation prompt.
            model: Ollama model name.
            config: Generation parameters.

        Returns:
            Generated text from Ollama.
        """
        client = await self._get_client()
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
            },
        }

        response = await client.post(
            f"{self._ollama_base_url}/api/generate",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return str(data["response"])

    async def _generate_litellm(
        self,
        prompt: str,
        model: str,
        config: GenerationConfig,
    ) -> str:
        """Generate text via LiteLLM proxy (OpenAI-compatible).

        Args:
            prompt: Full generation prompt.
            model: Model name as understood by LiteLLM.
            config: Generation parameters.

        Returns:
            Generated text from LiteLLM.
        """
        client = await self._get_client()
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }

        response = await client.post(
            f"{self._litellm_base_url}/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"])

    async def generate_structured(
        self,
        prompt: str,
        config: GenerationConfig,
        json_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate structured JSON output validated against a schema.

        Adds explicit JSON formatting instructions to the prompt and parses
        the response. Retries if the output is not valid JSON.

        Args:
            prompt: Base generation prompt.
            config: Generation parameters.
            json_schema: JSON schema dict describing expected output structure.

        Returns:
            Parsed JSON dict matching the provided schema.

        Raises:
            ValueError: If valid JSON cannot be extracted after retries.
        """
        schema_description = json.dumps(json_schema, indent=2)
        json_prompt = (
            f"{prompt}\n\n"
            f"Respond ONLY with valid JSON matching this schema:\n{schema_description}\n"
            "Do not include any text outside the JSON object."
        )

        for attempt in range(self._max_retries + 1):
            raw_output = await self.generate(json_prompt, config)
            cleaned = _strip_markdown_fences(raw_output)

            try:
                parsed = json.loads(cleaned)
                return dict(parsed)
            except json.JSONDecodeError:
                if attempt == self._max_retries:
                    raise ValueError(
                        f"Could not extract valid JSON after {self._max_retries} attempts. "
                        f"Last output: {raw_output[:200]}"
                    )
                self._log.warning(
                    "JSON parse failed, retrying",
                    attempt=attempt + 1,
                    raw_preview=raw_output[:100],
                )

        raise ValueError("Structured generation failed: unreachable code path")

    async def close(self) -> None:
        """Close the underlying HTTP client and release connections.

        Args:
            None

        Returns:
            None
        """
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._log.info("LLM client closed")


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from LLM output.

    Handles ```json ... ``` and ``` ... ``` blocks, returning only
    the inner content for JSON parsing.

    Args:
        text: Raw LLM output possibly containing markdown fences.

    Returns:
        Cleaned text with fences removed.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # Remove first line (```json or ```) and last line (```)
        inner_lines = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        return "\n".join(inner_lines).strip()
    return stripped
