"""OpenAI-compatible client wrapper with retries, validation, and logging."""

from __future__ import annotations

import logging
import os
import time
import traceback
from typing import Any

try:
    from openai import AzureOpenAI, OpenAI
except Exception:  # pragma: no cover
    AzureOpenAI = None
    OpenAI = None


LOGGER = logging.getLogger(__name__)


class ConfigurationError(RuntimeError):
    """Raised when FC-LAMMR deployment configuration is incomplete."""


def get_project_deployment_config() -> dict[str, str]:
    """Read deployment names directly from project_secrets.py."""
    try:
        import project_secrets  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ConfigurationError("project_secrets.py could not be imported for deployment validation.") from exc

    aliases = {
        "EXTRACTION_DEPLOYMENT_NAME": ["EXTRACTION_DEPLOYMENT_NAME"],
        "REASONING_DEPLOYMENT_NAME": ["REASONING_DEPLOYMENT_NAME"],
        "TOMIL_DEPLOYMENT_NAME": ["TOMIL_DEPLOYMENT_NAME"],
    }
    config: dict[str, str] = {}
    for key, candidates in aliases.items():
        value = None
        for candidate in candidates:
            value = getattr(project_secrets, candidate, None)
            if value:
                break
        if value:
            config[key] = str(value)
    return config


def validate_deployment_config() -> None:
    """
    Called once at router initialisation.
    Raises ConfigurationError with a specific, actionable message
    if any required deployment key is missing from project_secrets.
    Never silently falls back to a default deployment name.
    """
    config = get_project_deployment_config()
    required_keys = [
        "EXTRACTION_DEPLOYMENT_NAME",
        "REASONING_DEPLOYMENT_NAME",
        "TOMIL_DEPLOYMENT_NAME",
    ]
    missing = [key for key in required_keys if not config.get(key)]
    if missing:
        raise ConfigurationError(
            f"Missing deployment configuration keys: {missing}. "
            f"Add these to project_secrets.py before running evaluation. "
            f"Do not use placeholder values - deployment name mismatches "
            f"invalidate the comparison against the baseline router."
        )


class OpenAICompatibleLLMClient:
    """Thin wrapper around an OpenAI-compatible client."""

    def __init__(self, client: Any | None = None, default_model: str | None = None, provider: str = "openai"):
        self.default_model = default_model
        self.provider = provider
        if client is not None:
            self.client = client
            return
        if provider == "azure":
            validate_deployment_config()
            self.client = self._build_azure_client()
        else:
            self.client = self._build_openai_client()

    def _build_openai_client(self) -> Any:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Set it in the environment or inject a compatible llm client.")
        if OpenAI is None:
            raise RuntimeError("The 'openai' package is not available in this environment.")
        return OpenAI(api_key=api_key)

    def _build_azure_client(self) -> Any:
        endpoint = None
        api_key = None
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
        try:
            import project_secrets  # type: ignore

            endpoint = getattr(project_secrets, "AZURE_LLAMA_ENDPOINT", endpoint)
            api_key = getattr(project_secrets, "AZURE_LLAMA_KEY", api_key)
            api_version = getattr(project_secrets, "AZURE_OPENAI_API_VERSION", api_version)
        except Exception:
            endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if not endpoint or not api_key:
            raise RuntimeError("Azure client requested but Azure endpoint/key were not found in project_secrets.py or environment.")
        if AzureOpenAI is None:
            raise RuntimeError("The 'openai' package with AzureOpenAI support is not available in this environment.")
        return AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)

    def create_chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> Any:
        """Issue a chat completion with typed failure semantics and rich logging."""
        selected_model = model or self.default_model
        delay = 0.5
        for attempt in range(1, 4):
            start = time.perf_counter()
            try:
                response = self.client.chat.completions.create(
                    model=selected_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                latency = time.perf_counter() - start
                usage = getattr(response, "usage", None)
                tokens = getattr(usage, "total_tokens", None)
                LOGGER.debug(
                    "LLM call succeeded model=%s attempt=%s latency=%.3fs tokens=%s prompt=%s response=%s",
                    selected_model,
                    attempt,
                    latency,
                    tokens,
                    messages,
                    response,
                )
                return response
            except Exception as exc:  # pragma: no cover
                latency = time.perf_counter() - start
                error_text = str(exc)
                lowered = error_text.lower()
                if "content_filter" in lowered or "responsibleaipolicyviolation" in lowered or "jailbreak" in lowered:
                    LOGGER.warning(
                        "LLM call blocked by content filter model=%s attempt=%s latency=%.3fs error=%s",
                        selected_model,
                        attempt,
                        latency,
                        exc,
                    )
                    raise RuntimeError(f"CONTENT_FILTER_BLOCKED::{error_text}") from exc
                transient_markers = ["429", "503", "timeout", "temporarily unavailable", "connection error", "connecterror"]
                if any(marker in lowered for marker in transient_markers):
                    LOGGER.error(
                        "LLM transient failure model=%s attempt=%s latency=%.3fs error=%s traceback=%s",
                        selected_model,
                        attempt,
                        latency,
                        exc,
                        traceback.format_exc(),
                    )
                    if attempt == 3:
                        raise RuntimeError(f"FAILED_LLM_CALL::{error_text}") from exc
                    time.sleep(delay)
                    delay *= 2
                    continue
                LOGGER.error(
                    "LLM non-retryable failure model=%s attempt=%s latency=%.3fs error=%s traceback=%s",
                    selected_model,
                    attempt,
                    latency,
                    exc,
                    traceback.format_exc(),
                )
                raise
