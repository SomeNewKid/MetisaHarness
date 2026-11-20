"""Shared fixtures for the Metisa Harness test modules."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _harness_helpers import (
    _completion,
)

from metisa_harness.models import (
    AgentDefinition,
)


@pytest.fixture
def agent() -> AgentDefinition:
    """Provide a valid agent definition for execution tests."""
    return AgentDefinition(
        name="Assistant",
        description="Test agent",
        model="test-model",
        instructions="Answer briefly.",
    )


@pytest.fixture(autouse=True)
def http_client_factory() -> Iterator[MagicMock]:
    """Prevent real HTTP clients from being opened and expose their configuration."""
    with (
        patch("metisa_harness._model_resolver.httpx2.AsyncClient") as factory,
        patch("metisa_harness._chat_completions.httpx2.AsyncClient", new=factory),
    ):
        yield factory


@pytest.fixture
def model_endpoints() -> Iterator[tuple[MagicMock, AsyncMock, AsyncMock]]:
    """Provide separate mocked Ollama and OpenAI discovery/completion endpoints."""
    local = AsyncMock()
    local.models.list.return_value = SimpleNamespace(data=[])
    local.chat.completions.create.return_value = _completion("Local answer.")
    online = AsyncMock()
    online.chat.completions.create.return_value = _completion("Online answer.")

    def _client_context(*, base_url: str, **options: object) -> MagicMock:
        client = online if base_url == "https://api.openai.com/v1" else local
        context = MagicMock()
        context.__aenter__.return_value = client
        return context

    with (
        patch("metisa_harness._model_resolver.AsyncOpenAI") as factory,
        patch("metisa_harness._chat_completions.AsyncOpenAI", new=factory),
    ):
        factory.side_effect = _client_context
        yield factory, local, online
