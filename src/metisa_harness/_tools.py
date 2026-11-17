from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable

from jsonschema import validate
from mcp import Client
from mcp.types import Tool

from metisa_common.mcp_helper import create_tool

from .models import AgentDefinition, RegisteredTool, ToolCall


async def invoke_tool_call(
    tool_registry: dict[str, RegisteredTool],
    tool_call: ToolCall,
) -> str:
    """Invoke the specified tool call using the registered tools."""
    registered = tool_registry.get(tool_call.name)

    if registered is None:
        raise RuntimeError(f"The model requested an unknown tool: {tool_call.name}")

    arguments = json.loads(tool_call.arguments)

    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object.")

    validate(instance=arguments, schema=registered.tool.input_schema)

    return await registered.invoke(arguments)


async def get_tool_registry(
    agent: AgentDefinition,
) -> dict[str, RegisteredTool]:
    """Get the tool registry for the specified agent."""
    registered_tools: list[RegisteredTool] = []

    for function in agent.tools:
        registered = _register_local_tool(function)
        registered_tools.append(registered)

    for server in agent.mcp_clients:
        cursor: str | None = None

        while True:
            page = await server.list_tools(cursor=cursor)

            for tool in page.tools:
                registered = _register_mcp_tool(server, tool)
                registered_tools.append(registered)

            cursor = page.next_cursor
            if cursor is None:
                break

    registry: dict[str, RegisteredTool] = {}

    for registered in registered_tools:
        name = registered.tool.name

        if name in registry:
            raise ValueError(f"Duplicate tool name: {name}")

        registry[name] = registered

    return registry


def _register_local_tool(
    function: Callable[..., object],
) -> RegisteredTool:
    tool = create_tool(function)

    async def _invoke(arguments: dict[str, object]) -> str:
        # Run synchronous functions outside the event-loop thread.
        # Calling an async function here returns its corouting.
        result = await asyncio.to_thread(function, **arguments)

        if inspect.isawaitable(result):
            result = await result

        if tool.output_schema is not None:
            validate(instance=result, schema=tool.output_schema)

        return json.dumps(result, allow_nan=False)

    return RegisteredTool(tool=tool, invoke=_invoke)


def _register_mcp_tool(
    server: Client,
    tool: Tool,
) -> RegisteredTool:
    async def _invoke(arguments: dict[str, object]) -> str:
        result = await server.call_tool(tool.name, arguments)

        if result.is_error:
            raise RuntimeError(f"MCP tool failed: {tool.name}")

        payload = result.structured_content
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"MCP tool {tool.name} did not returned structured JSON."
            )

        if tool.output_schema is not None:
            validate(instance=payload, schema=tool.output_schema)

        return json.dumps(payload, allow_nan=False)

    return RegisteredTool(tool=tool, invoke=_invoke)
