"""Tests for Chat Completions requests, responses, and tool-message conversion."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from _harness_helpers import (
    _completion,
    _get_current_date_and_time,
    _local_tool,
    _remote_tool,
    _server,
)
from mcp.types import CallToolResult
from openai import omit
from openai.types.chat import ChatCompletion, ChatCompletionMessageCustomToolCall

from metisa_common.mcp_helper import create_tool
from metisa_harness.harness import AgentHarness
from metisa_harness.models import (
    AgentDefinition,
    ToolCall,
)


@pytest.mark.parametrize("include_tools", [False, True])
def test_run_sends_openai_tool_definitions(
    agent: AgentDefinition, include_tools: bool
) -> None:
    """The adapter translates metadata and omits options when no tools exist."""
    remote = _remote_tool("remote")
    server = _server(remote)
    definition = replace(
        agent,
        model="gpt4o",
        tools=[_get_current_date_and_time] if include_tools else [],
        mcp_clients=[server] if include_tools else [],
    )
    client = AsyncMock()
    client.models.list.return_value = SimpleNamespace(data=[])
    client.chat.completions.create.return_value = _completion("Hello.")

    with (
        patch("metisa_harness._model_resolver.AsyncOpenAI") as factory,
        patch("metisa_harness._chat_completions.AsyncOpenAI", new=factory),
    ):
        factory.return_value.__aenter__.return_value = client
        result = asyncio.run(AgentHarness.run(definition, input="Hello."))

    client.chat.completions.create.assert_awaited_once()
    request = client.chat.completions.create.await_args.kwargs
    assert request["model"] == "gpt-4o"
    assert request["messages"] == [
        {"role": "system", "content": agent.instructions},
        {"role": "user", "content": "Hello."},
    ]
    assert result.final_text == "Hello."
    if include_tools:
        local = create_tool(_get_current_date_and_time)
        assert request["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": local.name,
                    "description": local.description,
                    "parameters": local.input_schema,
                    "strict": False,
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "remote",
                    "description": "",
                    "parameters": remote.input_schema,
                    "strict": False,
                },
            },
        ]
        assert request["tool_choice"] == "auto"
        server.call_tool.assert_not_awaited()
    else:
        assert request["tools"] is omit
        assert request["tool_choice"] is omit


def test_run_round_trips_openai_tool_calls_and_results(agent: AgentDefinition) -> None:
    """Real adapter conversion preserves assistant text, raw arguments, and call IDs."""
    events: list[str] = []
    function = _local_tool("local", {"value": 42}, events)
    server = _server(_remote_tool("remote"))
    server.call_tool.return_value = CallToolResult(
        content=[], structured_content={"value": 9}
    )
    calls = [
        ToolCall(id="local-id", name="local", arguments='{ "value": 7 }'),
        ToolCall(id="remote-id", name="remote", arguments='{"query":"x"}'),
    ]
    client = AsyncMock()
    client.models.list.return_value = SimpleNamespace(data=[])
    client.chat.completions.create.side_effect = [
        _completion("Checking.", calls=calls, finish_reason="tool_calls"),
        _completion("Done."),
    ]
    definition = replace(agent, model="gpt4o", tools=[function], mcp_clients=[server])

    with (
        patch("metisa_harness._model_resolver.AsyncOpenAI") as factory,
        patch("metisa_harness._chat_completions.AsyncOpenAI", new=factory),
    ):
        factory.return_value.__aenter__.return_value = client
        result = asyncio.run(AgentHarness.run(definition, input="Lookup.", max_turns=2))

    assert result.final_text == "Done."
    assert events == ["local:7"]
    server.call_tool.assert_awaited_once_with("remote", {"query": "x"})
    assert client.chat.completions.create.await_count == 2
    first, second = client.chat.completions.create.await_args_list
    assert second.kwargs["messages"] == [
        {"role": "system", "content": agent.instructions},
        {"role": "user", "content": "Lookup."},
        {
            "role": "assistant",
            "content": "Checking.",
            "tool_calls": [
                {
                    "id": item.id,
                    "type": "function",
                    "function": {
                        "name": item.name,
                        "arguments": item.arguments,
                    },
                }
                for item in calls
            ],
        },
        {"role": "tool", "tool_call_id": "local-id", "content": '{"value": 42}'},
        {"role": "tool", "tool_call_id": "remote-id", "content": '{"value": 9}'},
    ]
    assert first.kwargs["messages"] == second.kwargs["messages"][:2]
    assert first.kwargs["tools"] == second.kwargs["tools"]
    assert first.kwargs["tool_choice"] == second.kwargs["tool_choice"] == "auto"


@pytest.mark.parametrize(
    ("completion_factory", "error_message"),
    [
        (
            lambda: _completion_without_choices(),
            "OpenAI returned no completion choices",
        ),
        (
            lambda: _completion(None, finish_reason="length"),
            "OpenAI completion ended with: length",
        ),
        (
            lambda: _completion(None, finish_reason="content_filter"),
            "OpenAI completion ended with: content_filter",
        ),
        (
            lambda: _completion(None, finish_reason="tool_calls"),
            "OpenAI returned no tool calls",
        ),
        (lambda: _completion(None), "The model returned no final text"),
        (
            lambda: _completion_with_custom_tool(),
            "Unsupported tool call type",
        ),
    ],
    ids=[
        "no-choices",
        "truncated",
        "filtered",
        "missing-calls",
        "missing-text",
        "custom-tool",
    ],
)
def test_run_rejects_invalid_openai_responses(
    agent: AgentDefinition,
    completion_factory: Callable[[], ChatCompletion],
    error_message: str,
) -> None:
    """Malformed or incomplete responses must not produce a successful run."""
    client = AsyncMock()
    client.models.list.return_value = SimpleNamespace(data=[])
    client.chat.completions.create.return_value = completion_factory()

    with (
        patch("metisa_harness._model_resolver.AsyncOpenAI") as factory,
        patch("metisa_harness._chat_completions.AsyncOpenAI", new=factory),
    ):
        factory.return_value.__aenter__.return_value = client
        with pytest.raises(RuntimeError, match=error_message):
            asyncio.run(AgentHarness.run(replace(agent, model="gpt4o"), input="Hi"))


@pytest.mark.parametrize("include_tools", [False, True])
def test_run_configures_local_discovery_and_completion_requests(
    agent: AgentDefinition,
    model_endpoints: tuple[MagicMock, AsyncMock, AsyncMock],
    http_client_factory: MagicMock,
    include_tools: bool,
) -> None:
    """Local requests use dummy credentials, bypass proxies, and omit tool choice."""
    factory, local, online = model_endpoints
    local.models.list.return_value = SimpleNamespace(
        data=[SimpleNamespace(id="gemma4:12b")]
    )
    definition = replace(
        agent,
        model="gemma4:12b",
        tools=[_get_current_date_and_time] if include_tools else [],
    )

    asyncio.run(AgentHarness.run(definition, input="Hello."))

    assert factory.call_args_list == [
        call(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            http_client=http_client_factory.return_value,
            timeout=20.0,
            max_retries=0,
        ),
        call(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            http_client=http_client_factory.return_value,
        ),
    ]
    assert http_client_factory.call_args_list == [
        call(trust_env=False),
        call(trust_env=False),
    ]
    local.chat.completions.create.assert_awaited_once()
    request = local.chat.completions.create.await_args.kwargs
    assert request["model"] == "gemma4:12b"
    assert request["messages"] == [
        {"role": "system", "content": agent.instructions},
        {"role": "user", "content": "Hello."},
    ]
    assert request["tool_choice"] is omit
    if include_tools:
        metadata = create_tool(_get_current_date_and_time)
        assert request["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": metadata.name,
                    "description": metadata.description,
                    "parameters": metadata.input_schema,
                    "strict": False,
                },
            }
        ]
    else:
        assert request["tools"] is omit
    online.chat.completions.create.assert_not_awaited()


def _completion_without_choices() -> ChatCompletion:
    completion = _completion(None)
    completion.choices = []
    return completion


def _completion_with_custom_tool() -> ChatCompletion:
    completion = _completion(None, finish_reason="tool_calls")
    custom_call = ChatCompletionMessageCustomToolCall.model_validate(
        {
            "id": "custom-id",
            "type": "custom",
            "custom": {"name": "custom-tool", "input": "test input"},
        }
    )
    completion.choices[0].message.tool_calls = [custom_call]
    return completion
