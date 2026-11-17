"""Resolve a model alias to a model implementaton."""

from __future__ import annotations

import httpx2
from openai import APIConnectionError, AsyncOpenAI

from ._chat_completions import (
    OpenAiLlmClient,
)
from .models import (
    AgentDefinition,
    LlmClient,
)

_OLLAMA_BASE_URL = "http://localhost:11434/v1"

_ONLINE_MODELS = {
    "gpt4o": "gpt-4o",
    "gpt-4o": "gpt-4o",
}


async def get_llm(
    agent: AgentDefinition,
) -> LlmClient | None:
    """Get the LLM client for the specified agent."""
    local_model = await find_ollama_model(agent.model, _OLLAMA_BASE_URL)

    if local_model is not None:
        return OpenAiLlmClient(
            model=local_model,
            base_url=_OLLAMA_BASE_URL,
            api_key="ollama",
            trust_env=False,
            supports_tool_choice=False,
        )

    online_model = _ONLINE_MODELS.get(agent.model)

    if online_model is None:
        raise RuntimeError(
            f"Model '{agent.model}' was not found in Ollama "
            "and has no configured online model mapping."
        )

    return OpenAiLlmClient(model=online_model)


async def find_ollama_model(
    requested_model: str,
    base_url: str,
) -> str | None:
    """Find the specified Ollama model locally."""
    try:
        async with AsyncOpenAI(
            base_url=base_url,
            api_key="ollama",
            http_client=httpx2.AsyncClient(trust_env=False),
            timeout=20.0,
            max_retries=0,
        ) as client:
            page = await client.models.list()

    except APIConnectionError as error:
        cause = error.__cause__ or error

        raise RuntimeError(
            f"Ollama discovery failed at {base_url}: "
            f"{type(error).__name__}; cause: {cause!r}"
        ) from error

    available_names = {model.id for model in page.data}

    if requested_model in available_names:
        return requested_model

    # Ollama names without an explicit tag commonly mean ":latest".
    final_compnent = requested_model.rsplit("/", 1)[-1]

    if ":" not in final_compnent:
        tagged_name = f"{requested_model}:latest"

        if tagged_name in available_names:
            return tagged_name

    return None
