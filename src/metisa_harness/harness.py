"""Implement an AI agent harness."""

from __future__ import annotations

from copy import deepcopy

from ._context_manager import FullHistoryContextBuilder
from ._model_resolver import get_llm
from ._tools import get_tool_registry, invoke_tool_call
from .models import (
    AgentDefinition,
    ContextBuilder,
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
        context_builder: ContextBuilder | None = None,
        max_turns: int = 10,
    ) -> RunResult:
        """Run model turns and tools until a final answer is produced."""
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1.")

        if llm is None:
            llm = await get_llm(agent)

        if llm is None:
            raise RuntimeError("LLM not available.")

        tool_registry = await get_tool_registry(agent)
        messages = [LlmMessage(role="user", content=input)]
        if context_builder is None:
            context_builder = FullHistoryContextBuilder()

        for _ in range(max_turns):
            model_context = context_builder.build(
                agent=agent,
                tool_registry=tool_registry,
                messages=messages,
            )
            response = await llm.complete(model_context=model_context)
            messages.append(response)

            if not response.tool_calls:
                final_text = response.content

                if final_text is None or not final_text.strip():
                    raise RuntimeError("The model returned no final text.")

                history = tuple(deepcopy(messages))

                return RunResult(final_text=final_text, messages=history)

            for tool_call in response.tool_calls:
                output = await invoke_tool_call(tool_registry, tool_call)

                messages.append(
                    LlmMessage(
                        role="tool",
                        content=output,
                        tool_call_id=tool_call.id,
                    )
                )

        raise RuntimeError("Maximum model turns exceeded.")
