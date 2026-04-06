"""LLM API client abstraction.

Supports OpenAI, Azure OpenAI, and Anthropic backends via a unified interface.
Includes retry logic, response caching, and optional logprobs extraction.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

import openai as _openai_sdk

from src.utils.logging_utils import get_logger
from src.utils.io import save_json, load_json

logger = get_logger(__name__)


def _retryable_openai(exc: BaseException) -> bool:
    """Retry only transient failures; do not retry auth / model-access / bad requests."""
    return isinstance(
        exc,
        (
            _openai_sdk.RateLimitError,
            _openai_sdk.APIConnectionError,
            _openai_sdk.APITimeoutError,
            _openai_sdk.InternalServerError,
        ),
    )


def _retryable_anthropic(exc: BaseException) -> bool:
    import anthropic as anthropic_sdk

    return isinstance(
        exc,
        (
            anthropic_sdk.RateLimitError,
            anthropic_sdk.APIConnectionError,
            anthropic_sdk.APITimeoutError,
            anthropic_sdk.InternalServerError,
        ),
    )


# ====================================================================
# Unified response container
# ====================================================================

class LLMResponse:
    """Standardised container for an LLM completion response."""

    def __init__(
        self,
        text: str,
        model: str = "",
        finish_reason: str = "",
        logprobs: Optional[List[Dict]] = None,
        raw: Optional[Dict] = None,
        usage: Optional[Dict] = None,
    ):
        self.text = text
        self.model = model
        self.finish_reason = finish_reason
        self.logprobs = logprobs  # list of token logprob dicts if available
        self.raw = raw or {}
        self.usage = usage or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "logprobs": self.logprobs,
            "usage": self.usage,
        }


# ====================================================================
# Cache helpers
# ====================================================================

def _cache_key(messages: List[Dict], model: str, temperature: float, seed: int | None) -> str:
    """Deterministic hash for a request."""
    payload = json.dumps({"messages": messages, "model": model, "temperature": temperature, "seed": seed}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _check_cache(cache_dir: str | None, key: str) -> Optional[LLMResponse]:
    if not cache_dir:
        return None
    fp = Path(cache_dir) / f"{key}.json"
    if fp.exists():
        data = load_json(fp)
        return LLMResponse(**data)
    return None


def _write_cache(cache_dir: str | None, key: str, resp: LLMResponse) -> None:
    if not cache_dir:
        return
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    save_json(resp.to_dict(), Path(cache_dir) / f"{key}.json")


# ====================================================================
# Provider-specific callers
# ====================================================================

class _OpenAIClient:
    """OpenAI / Azure OpenAI chat completion client."""

    def __init__(self, cfg: Dict):
        import openai

        self.cfg = cfg
        provider = cfg.get("provider", "openai")

        if provider == "azure":
            self.client = openai.AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            )
            self.model = os.getenv("AZURE_OPENAI_DEPLOYMENT", cfg.get("model", "gpt-4"))
        else:
            base_url = (cfg.get("base_url") or "").strip() or None
            key = os.getenv("OPENAI_API_KEY")
            # Some OpenAI-compatible gateways accept any non-empty API key when base_url is set.
            if not key:
                if base_url:
                    key = "not-used"
                else:
                    raise ValueError(
                        "OPENAI_API_KEY is not set. Add it to the repo root .env or export it in your shell."
                    )
            client_kw: Dict[str, Any] = {"api_key": key}
            if base_url:
                client_kw["base_url"] = base_url
            # Org/project headers are for OpenAI Cloud only; omit for custom base_url gateways.
            if not base_url:
                org = os.getenv("OPENAI_ORG_ID")
                if org:
                    client_kw["organization"] = org
                proj = os.getenv("OPENAI_PROJECT_ID")
                if proj:
                    client_kw["project"] = proj
            self.client = openai.OpenAI(**client_kw)
            self.model = cfg.get("model", "gpt-4-0125-preview")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(min=2, max=60),
        retry=retry_if_exception(_retryable_openai),
        reraise=True,
    )
    def complete(self, messages: List[Dict], **kwargs) -> LLMResponse:
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.cfg.get("temperature", 0.0),
            "max_tokens": self.cfg.get("max_tokens", 1024),
        }
        # OpenAI cloud supports seed; many compatible gateways omit or reject it.
        if self.cfg.get("seed") and not (self.cfg.get("base_url") or "").strip():
            params["seed"] = self.cfg["seed"]
        if self.cfg.get("request_logprobs"):
            params["logprobs"] = True
            params["top_logprobs"] = self.cfg.get("top_logprobs", 5)

        params.update(kwargs)
        resp = self.client.chat.completions.create(**params)
        choice = resp.choices[0]

        logprobs_data = None
        if choice.logprobs and choice.logprobs.content:
            logprobs_data = [
                {"token": t.token, "logprob": t.logprob, "top_logprobs": [{tl.token: tl.logprob for tl in t.top_logprobs}] if t.top_logprobs else []}
                for t in choice.logprobs.content
            ]

        return LLMResponse(
            text=choice.message.content or "",
            model=resp.model,
            finish_reason=choice.finish_reason or "",
            logprobs=logprobs_data,
            usage={"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens} if resp.usage else {},
        )


class _AnthropicClient:
    """Anthropic Claude client."""

    def __init__(self, cfg: Dict):
        import anthropic
        self.cfg = cfg
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = cfg.get("model", "claude-3-5-sonnet-20241022")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(min=2, max=60),
        retry=retry_if_exception(_retryable_anthropic),
        reraise=True,
    )
    def complete(self, messages: List[Dict], **kwargs) -> LLMResponse:
        # Anthropic uses a system param separate from messages
        system_msg = ""
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                user_messages.append(m)

        params: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.cfg.get("max_tokens", 1024),
            "temperature": self.cfg.get("temperature", 0.0),
            "messages": user_messages,
        }
        if system_msg:
            params["system"] = system_msg

        resp = self.client.messages.create(**params)
        text = resp.content[0].text if resp.content else ""

        return LLMResponse(
            text=text,
            model=resp.model,
            finish_reason=resp.stop_reason or "",
            logprobs=None,  # Anthropic does not support logprobs
            usage={"prompt_tokens": resp.usage.input_tokens, "completion_tokens": resp.usage.output_tokens} if resp.usage else {},
        )


# ====================================================================
# Unified LLM client
# ====================================================================

class LLMClient:
    """Unified LLM client with caching and rate limiting."""

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        provider = cfg.get("provider", "openai")
        self.cache_dir = cfg.get("cache_dir") if cfg.get("cache_responses") else None
        self._rpm = cfg.get("rate_limit_rpm", 30)
        self._last_call = 0.0

        if provider in ("openai", "azure"):
            self._client = _OpenAIClient(cfg)
        elif provider == "anthropic":
            self._client = _AnthropicClient(cfg)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

        self.model_name = getattr(self._client, "model", provider)
        logger.info("LLMClient initialised — provider=%s  model=%s", provider, self.model_name)

    def complete(self, messages: List[Dict], use_cache: bool = True, **kwargs) -> LLMResponse:
        """Send a chat completion request, with optional caching and rate limiting."""
        key = _cache_key(messages, self.model_name, self.cfg.get("temperature", 0), self.cfg.get("seed"))

        if use_cache:
            cached = _check_cache(self.cache_dir, key)
            if cached is not None:
                logger.debug("Cache hit: %s", key[:12])
                return cached

        # Simple rate limiting
        if self._rpm > 0:
            min_interval = 60.0 / self._rpm
            elapsed = time.time() - self._last_call
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

        resp = self._client.complete(messages, **kwargs)
        self._last_call = time.time()

        if use_cache:
            _write_cache(self.cache_dir, key, resp)

        return resp
