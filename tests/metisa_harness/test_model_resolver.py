"""Tests for local-model precedence, online resolution, and discovery failures."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import httpx2
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from metisa_harness.harness import AgentHarness
from metisa_harness.models import (
    AgentDefinition,
)


@pytest.mark.parametrize(
    ("requested", "installed", "expected"),
    [
        ("granite4.1:3b", ["granite4.1:3b"], "granite4.1:3b"),
        ("granite4.1:8b", ["granite4.1:8b"], "granite4.1:8b"),
        ("gemma4:12b", ["gemma4:12b"], "gemma4:12b"),
        ("gpt4o", ["gpt4o"], "gpt4o"),
        ("gpt4o", ["gpt4o:latest"], "gpt4o:latest"),
        ("gpt4o", ["gpt4o:latest", "gpt4o"], "gpt4o"),
        ("library/local", ["library/local:latest"], "library/local:latest"),
    ],
    ids=[
        "granite-3b",
        "granite-8b",
        "gemma",
        "online-name-collision",
        "default-tag",
        "exact-before-default-tag",
        "namespaced-default-tag",
    ],
)
def test_run_prefers_installed_ollama_models(
    agent: AgentDefinition,
    model_endpoints: tuple[MagicMock, AsyncMock, AsyncMock],
    requested: str,
    installed: list[str],
    expected: str,
) -> None:
    """Local names take precedence, including collisions with online aliases."""
    _, local, online = model_endpoints
    local.models.list.return_value = SimpleNamespace(
        data=[SimpleNamespace(id=name) for name in installed]
    )

    result = asyncio.run(AgentHarness.run(replace(agent, model=requested), input="Hi"))

    assert result.final_text == "Local answer."
    local.models.list.assert_awaited_once_with()
    local.chat.completions.create.assert_awaited_once()
    assert local.chat.completions.create.await_args.kwargs["model"] == expected
    online.chat.completions.create.assert_not_awaited()


@pytest.mark.parametrize("requested", ["gpt4o", "gpt-4o"])
def test_run_falls_back_to_online_model_when_not_installed(
    agent: AgentDefinition,
    model_endpoints: tuple[MagicMock, AsyncMock, AsyncMock],
    http_client_factory: MagicMock,
    requested: str,
) -> None:
    """Unrelated local models do not prevent resolving either supported online name."""
    factory, local, online = model_endpoints
    local.models.list.return_value = SimpleNamespace(
        data=[SimpleNamespace(id="granite4.1:3b")]
    )

    result = asyncio.run(AgentHarness.run(replace(agent, model=requested), input="Hi"))

    assert result.final_text == "Online answer."
    local.models.list.assert_awaited_once_with()
    local.chat.completions.create.assert_not_awaited()
    online.chat.completions.create.assert_awaited_once()
    assert online.chat.completions.create.await_args.kwargs["model"] == "gpt-4o"
    assert factory.call_args.kwargs["base_url"] == "https://api.openai.com/v1"
    assert factory.call_args.kwargs["api_key"] is None
    assert http_client_factory.call_args_list == [
        call(trust_env=False),
        call(trust_env=True),
    ]


@pytest.mark.parametrize("failure", ["connection", "timeout"])
def test_run_reports_ollama_discovery_failure_without_online_fallback(
    agent: AgentDefinition,
    model_endpoints: tuple[MagicMock, AsyncMock, AsyncMock],
    failure: str,
) -> None:
    """Discovery errors retain their cause and do not trigger online inference."""
    _, local, online = model_endpoints
    request = httpx2.Request("GET", "http://localhost:11434/v1/models")
    error = (
        APITimeoutError(request=request)
        if failure == "timeout"
        else APIConnectionError(request=request)
    )
    local.models.list.side_effect = error

    with pytest.raises(RuntimeError, match="Ollama discovery failed") as caught:
        asyncio.run(AgentHarness.run(replace(agent, model="gpt4o"), input="Hi"))

    assert caught.value.__cause__ is error
    assert type(error).__name__ in str(caught.value)
    assert "http://localhost:11434/v1" in str(caught.value)
    local.models.list.assert_awaited_once_with()
    local.chat.completions.create.assert_not_awaited()
    online.chat.completions.create.assert_not_awaited()


@pytest.mark.parametrize("unavailable", [False, True])
def test_run_reports_missing_or_unreachable_model(
    agent: AgentDefinition,
    model_endpoints: tuple[MagicMock, AsyncMock, AsyncMock],
    unavailable: bool,
) -> None:
    """Missing models and discovery failures report distinct errors before inference."""
    _, local, online = model_endpoints
    if unavailable:
        local.models.list.side_effect = APIConnectionError(
            request=httpx2.Request("GET", "http://localhost:11434/v1/models")
        )

    expected_error = (
        "Ollama discovery failed"
        if unavailable
        else "has no configured online model mapping"
    )
    with pytest.raises(RuntimeError, match=expected_error):
        asyncio.run(AgentHarness.run(replace(agent, model="missing:3b"), input="Hi"))

    local.models.list.assert_awaited_once_with()
    local.chat.completions.create.assert_not_awaited()
    online.chat.completions.create.assert_not_awaited()


def test_run_does_not_replace_an_explicit_ollama_tag(
    agent: AgentDefinition,
    model_endpoints: tuple[MagicMock, AsyncMock, AsyncMock],
) -> None:
    """A different installed variant is not a match for an explicit requested tag."""
    _, local, online = model_endpoints
    local.models.list.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(id="granite4.1:8b"),
            SimpleNamespace(id="granite4.1:latest"),
        ]
    )

    with pytest.raises(RuntimeError, match="has no configured online model mapping"):
        asyncio.run(AgentHarness.run(replace(agent, model="granite4.1:3b"), input="Hi"))

    local.chat.completions.create.assert_not_awaited()
    online.chat.completions.create.assert_not_awaited()


def test_run_does_not_fall_back_after_local_completion_failure(
    agent: AgentDefinition,
    model_endpoints: tuple[MagicMock, AsyncMock, AsyncMock],
) -> None:
    """Local inference failures propagate without switching to the online provider."""
    _, local, online = model_endpoints
    local.models.list.return_value = SimpleNamespace(data=[SimpleNamespace(id="gpt4o")])
    error = APIConnectionError(
        request=httpx2.Request("POST", "http://localhost:11434/v1/chat/completions")
    )
    local.chat.completions.create.side_effect = error

    with pytest.raises(APIConnectionError) as caught:
        asyncio.run(AgentHarness.run(replace(agent, model="gpt4o"), input="Hi"))

    assert caught.value is error
    local.models.list.assert_awaited_once_with()
    local.chat.completions.create.assert_awaited_once()
    online.chat.completions.create.assert_not_awaited()


def test_run_surfaces_unexpected_ollama_discovery_errors(
    agent: AgentDefinition,
    model_endpoints: tuple[MagicMock, AsyncMock, AsyncMock],
) -> None:
    """HTTP failures are not mistaken for an absent model or unavailable service."""
    _, local, online = model_endpoints
    request = httpx2.Request("GET", "http://localhost:11434/v1/models")
    error = APIStatusError(
        "Discovery failed.",
        response=httpx2.Response(500, request=request),
        body=None,
    )
    local.models.list.side_effect = error

    with pytest.raises(APIStatusError) as caught:
        asyncio.run(AgentHarness.run(replace(agent, model="gpt4o"), input="Hi"))

    assert caught.value is error
    local.chat.completions.create.assert_not_awaited()
    online.chat.completions.create.assert_not_awaited()
