"""Model resolution: a model string plus optional endpoint overrides -> pydantic-ai Model.

Supported strings:

- ``"scripted"`` - the deterministic stand-in from ``scripted.py`` (no API needed)
- ``"test"`` - pydantic-ai's ``TestModel`` (calls every tool once, then answers)
- any pydantic-ai model string, e.g. ``"anthropic:claude-sonnet-4-5"``,
  ``"openai:gpt-4.1-mini"``, ``"openai-chat:<name>"``, ``"ollama:<name>"``,
  ``"vllm:<name>"``, ``"openrouter:<vendor>/<name>"``; credentials come from the
  provider's usual environment variables (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ...)

With ``base_url`` set, the model name is served through the OpenAI Chat Completions
protocol at that address - the common denominator of vLLM, Ollama, LM Studio, llama.cpp
and most hosted gateways - which is how a local ≤20B model on the project GPU is wired in.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic_ai.models import Model, infer_model

from .scripted import MODEL_NAME as SCRIPTED
from .scripted import scripted_model

TEST = "test"


def build_model(
    name: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    scripted_options: dict[str, Any] | None = None,
) -> Model:
    if name == SCRIPTED:
        return scripted_model(**(scripted_options or {}))
    if name == TEST:
        from pydantic_ai.models.test import TestModel

        return TestModel()
    if base_url:
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        model_name = name.split(":", 1)[1] if ":" in name else name
        return OpenAIChatModel(model_name, provider=OpenAIProvider(base_url=base_url, api_key=api_key or "unused"))
    return infer_model(name)


def model_slug(name: str) -> str:
    """A file-system-safe tag for a model string, e.g. ``openai-chat:gpt-oss-20b`` -> ``openai-chat_gpt-oss-20b``."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "model"
