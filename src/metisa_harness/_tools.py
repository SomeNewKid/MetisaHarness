from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable
from uuid import uuid4

from jsonschema import validate
from mcp import Client
from mcp.types import Tool

from metisa_common.mcp_helper import create_tool

from ._guardrails import apply_guardrails
from .guardrails import GuardrailMetadata
from .models import AgentDefinition, RegisteredTool, ToolCall


async def invoke_tool_call(
    agent: AgentDefinition,
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

    metadata = GuardrailMetadata(
        agent_name=agent.name,
        tool_name=tool_call.name,
    )

    before_tool_guardrails = (
        guardrail
        for plugin in agent.guardrail_plugins
        for guardrail in plugin.before_tool
    )

    arguments = await apply_guardrails(
        before_tool_guardrails,
        metadata=metadata,
        payload=arguments,
    )

    validate(
        instance=arguments,
        schema=registered.tool.input_schema,
    )

    result = await registered.invoke(arguments)

    after_tool_guardrails = (
        guardrail
        for plugin in agent.guardrail_plugins
        for guardrail in plugin.after_tool
    )

    result = await apply_guardrails(
        after_tool_guardrails,
        metadata=metadata,
        payload=result,
    )

    if registered.tool.output_schema is not None:
        validate(
            instance=result,
            schema=registered.tool.output_schema,
        )

    return json.dumps(result, allow_nan=False)


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

    for sub_agent in agent.sub_agents:
        registered = _register_sub_agent(sub_agent)
        registered_tools.append(registered)

    registry: dict[str, RegisteredTool] = {}

    for registered in registered_tools:
        name = registered.tool.name

        if name in registry:
            raise ValueError(f"Duplicate tool name: {name}")

        registry[name] = registered

    return registry


def _register_local_tool(
    function: Callable[..., dict[str, object]],
) -> RegisteredTool:
    tool = create_tool(function)

    async def _invoke(arguments: dict[str, object]) -> dict[str, object]:
        # Run synchronous functions outside the event-loop thread.
        # Calling an async function here returns its corouting.
        result = await asyncio.to_thread(function, **arguments)

        if inspect.isawaitable(result):
            result = await result

        if tool.output_schema is not None:
            validate(instance=result, schema=tool.output_schema)

        return result

    return RegisteredTool(tool=tool, invoke=_invoke)


def _register_mcp_tool(
    server: Client,
    tool: Tool,
) -> RegisteredTool:
    async def _invoke(arguments: dict[str, object]) -> dict[str, object]:
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

        return payload

    return RegisteredTool(tool=tool, invoke=_invoke)


def _register_sub_agent(
    sub_agent: AgentDefinition,
) -> RegisteredTool:
    if sub_agent.sub_agents:
        raise ValueError(f"Sub-agent {sub_agent.name} cannot delegate to sub-agents.")

    output_type = sub_agent.output_type
    output_schema = dict[str, object]

    if output_type is None:
        output_schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
    else:
        output_schema = output_type.model_json_schema(
            mode="serialization",
            by_alias=True,
        )

    tool = Tool(
        name=sub_agent.name,
        description=sub_agent.description,
        input_schema={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task or text to pass to this agent.",
                },
            },
            "required": ["task"],
            "additionalProperties": False,
        },
        output_schema=output_schema,
    )

    async def _invoke(arguments: dict[str, object]) -> dict[str, object]:
        # harness.py imports this module, so defer the import until invocation.
        from .harness import AgentHarness

        task = arguments.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("The sub-agent task must be non-empty text.")

        run_identifier = uuid4().hex
        result = await AgentHarness.run(
            sub_agent,
            run_identifier=run_identifier,
            input=task,
        )

        answer = result.final_output

        if output_type is None:
            if not isinstance(answer, str):
                raise TypeError(f"Sub-agent {sub_agent.name} returned non-text output.")

            return {
                "answer": answer,
            }

        if not isinstance(answer, output_type):
            raise TypeError(
                f"Sub-agent {sub_agent.name} returned an unexpected model type."
            )

        payload = answer.model_dump(mode="json", by_alias=True)
        if not isinstance(payload, dict):
            raise TypeError(
                f"Sub-agent {sub_agent.name} must serialize to a JSON object."
            )

        return payload

    return RegisteredTool(tool=tool, invoke=_invoke)
