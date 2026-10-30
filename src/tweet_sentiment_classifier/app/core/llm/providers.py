"""LLM providers over plain HTTPS (urllib): no SDK to install and nothing to download in tests.

Groq and OpenAI share one API shape, so one class serves both. To add a provider (Gemini, ...), write a
class with `name`, `model` and `complete()` and add it to `build_provider`.
"""
import json
import logging
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger("api")

GROQ_URL = "https://api.groq.com/openai/v1"
OPENAI_URL = "https://api.openai.com/v1"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODELS = {"groq": "qwen/qwen3.8-27b", "anthropic": "claude-haiku-5-5"}  # openai: set LLM_MODEL


class LLMError(RuntimeError):
    """A provider call failed. The message never contains the API key or the response body."""


@dataclass
class LLMResponse:
    text: str
    tokens_in: int
    tokens_out: int


def _post_json(url: str, headers: dict, body: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "User-Agent": "tweet-sentiment-monitor/1.0", **headers})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        raise LLMError(f"HTTP {e.code} from {url.split('/')[2]}") from None
    except (OSError, ValueError) as e:
        raise LLMError(f"request to {url.split('/')[2]} failed: {type(e).__name__}") from None


class LLMProvider(ABC):
    name: str
    model: str

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 600) -> LLMResponse: ...


class OpenAICompatible(LLMProvider):
    def __init__(self, name: str, base_url: str, api_key: str, model: str, timeout: float = 30.0):
        self.name, self.base_url, self.api_key, self.model, self.timeout = name, base_url, api_key, model, timeout

    def complete(self, system: str, user: str, max_tokens: int = 600) -> LLMResponse:
        data = _post_json(f"{self.base_url}/chat/completions", {"Authorization": f"Bearer {self.api_key}"}, {
            "model": self.model, "max_tokens": max_tokens, "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }, self.timeout)
        try:
            usage = data.get("usage") or {}
            return LLMResponse(data["choices"][0]["message"]["content"] or "",
                               int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)))
        except (KeyError, IndexError, TypeError):
            raise LLMError("unexpected response shape") from None


class Anthropic(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str, model: str, timeout: float = 30.0):
        self.api_key, self.model, self.timeout = api_key, model, timeout

    def complete(self, system: str, user: str, max_tokens: int = 600) -> LLMResponse:
        data = _post_json(ANTHROPIC_URL, {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}, {
            "model": self.model, "max_tokens": max_tokens, "temperature": 0.2, "system": system,
            "messages": [{"role": "user", "content": user}],
        }, self.timeout)
        try:
            usage = data.get("usage") or {}
            return LLMResponse("".join(b.get("text", "") for b in data["content"]),
                               int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)))
        except (KeyError, TypeError, AttributeError):
            raise LLMError("unexpected response shape") from None


def build_provider(settings) -> LLMProvider | None:
    """The configured provider, or None (the insight features that need an LLM are then skipped)."""
    name = (settings.LLM_PROVIDER or ("groq" if settings.GROQ_API_KEY else "")).lower()
    model = settings.LLM_MODEL or DEFAULT_MODELS.get(name, "")
    timeout = settings.LLM_TIMEOUT_SECONDS
    if name == "groq" and settings.GROQ_API_KEY:
        return OpenAICompatible("groq", GROQ_URL, settings.GROQ_API_KEY, model, timeout)
    if name == "openai" and settings.OPENAI_API_KEY and model:
        return OpenAICompatible("openai", OPENAI_URL, settings.OPENAI_API_KEY, model, timeout)
    if name == "anthropic" and settings.ANTHROPIC_API_KEY:
        return Anthropic(settings.ANTHROPIC_API_KEY, model, timeout)
    if name:
        logger.warning(f"LLM provider '{name}' is not usable (unknown name, missing key or missing model)")
    return None
