"""Implement an AI agent harness."""

from __future__ import annotations

from copy import deepcopy

from ._model_resolver import _get_llm
from ._tools import _get_tool_registry, _invoke_tool_call
from .models import (
    AgentDefinition,
    LlmClient,
    LlmMessage,
    RunResult,
)


class AgentHarness:
    """Execute an agent workload."""

    @staticmethod
    async def run(
        agent: AgentDefinition,
        *,
        input: str,
        llm: LlmClient | None = None,
        max_turns: int = 10,
    ) -> RunResult:
        """Run model turns and tools until a final answer is produced."""
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1.")

        if llm is None:
            llm = await _get_llm(agent)

        if llm is None:
            raise RuntimeError("LLM not available.")

        tool_registry = await _get_tool_registry(agent)
        tools = [registered.tool for registered in tool_registry.values()]
        messages = [LlmMessage(role="user", content=input)]

        for _ in range(max_turns):
            response = await llm.complete(messages=messages, tools=tools)
            messages.append(response)

            if not response.tool_calls:
                final_text = response.content

                if final_text is None or not final_text.strip():
                    raise RuntimeError("The model returned no final text.")

                history = tuple(deepcopy(messages))

                return RunResult(final_text=final_text, messages=history)

            for tool_call in response.tool_calls:
                output = await _invoke_tool_call(tool_registry, tool_call)

                messages.append(
                    LlmMessage(
                        role="tool",
                        content=output,
                        tool_call_id=tool_call.id,
                    )
                )

        raise RuntimeError("Maximum model turns exceeded.")
