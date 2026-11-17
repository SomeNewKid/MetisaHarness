"""Implement an AI agent harness."""

from __future__ import annotations

from copy import deepcopy

from ._context_manager import FullHistoryContextBuilder
from ._guardrails import apply_guardrails
from ._model_resolver import get_llm
from ._tools import get_tool_registry, invoke_tool_call
from .guardrails import GuardrailMetadata
from .models import (
    AgentDefinition,
    ContextBuilder,
    LlmClient,
    LlmMessage,
    ModelContext,
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

        guardrail_metadata = GuardrailMetadata(
            agent_name=agent.name,
        )

        before_run_guardrails = (
            guardrail
            for plugin in agent.guardrail_plugins
            for guardrail in plugin.before_run
        )

        input = await apply_guardrails(
            before_run_guardrails,
            metadata=guardrail_metadata,
            payload=input,
        )

        messages = [LlmMessage(role="user", content=input)]

        if context_builder is None:
            context_builder = FullHistoryContextBuilder()

        for _ in range(max_turns):
            model_context = context_builder.build(
                agent=agent,
                tool_registry=tool_registry,
                messages=messages,
            )

            before_model_guardrails = (
                guardrail
                for plugin in agent.guardrail_plugins
                for guardrail in plugin.before_model
            )

            model_context = await apply_guardrails(
                before_model_guardrails,
                metadata=guardrail_metadata,
                payload=model_context,
            )

            if not isinstance(model_context, ModelContext):
                raise RuntimeError("Invalid type for model context.")

            response = await llm.complete(model_context=model_context)

            after_model_guardrails = (
                guardrail
                for plugin in agent.guardrail_plugins
                for guardrail in plugin.after_model
            )

            response = await apply_guardrails(
                after_model_guardrails,
                metadata=guardrail_metadata,
                payload=response,
            )

            messages.append(response)

            if not response.tool_calls:
                final_text = response.content

                if final_text is None or not final_text.strip():
                    raise RuntimeError("The model returned no final text.")

                after_run_guardrails = (
                    guardrail
                    for plugin in agent.guardrail_plugins
                    for guardrail in plugin.after_run
                )

                final_text = await apply_guardrails(
                    after_run_guardrails,
                    metadata=guardrail_metadata,
                    payload=final_text,
                )

                history = tuple(deepcopy(messages))

                return RunResult(final_text=final_text, messages=history)

            for tool_call in response.tool_calls:
                output = await invoke_tool_call(agent, tool_registry, tool_call)

                messages.append(
                    LlmMessage(
                        role="tool",
                        content=output,
                        tool_call_id=tool_call.id,
                    )
                )

        raise RuntimeError("Maximum model turns exceeded.")
