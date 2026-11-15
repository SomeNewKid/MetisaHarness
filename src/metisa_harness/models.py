"""Data models for the Metisa Harness module."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

from mcp import Client
from mcp.types import Tool


@dataclass(frozen=True)
class AgentDefinition:
    """Define an agent."""

    name: str
    model: str
    instructions: str
    tools: list[Callable[..., object]] = field(default_factory=list)
    mcp_servers: list[Client] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Agent name is required.")
        if not self.model.strip():
            raise ValueError("Agent model is required.")
        if not self.instructions.strip():
            raise ValueError("Agent instructions are required.")


@dataclass(frozen=True)
class RunResult:
    """Contain the final answer and conversation from a completed run."""

    final_text: str
    messages: tuple[LlmMessage, ...]


@dataclass(frozen=True)
class ToolCall:
    """Represent a model's request to invoke a tool."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class LlmMessage:
    """Represent a conversation message exchanged with a model."""

    role: Literal["user", "assistant", "tool"]
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None


class LlmClient(Protocol):
    """Provide model responses."""

    async def complete(
        self,
        *,
        messages: list[LlmMessage],
        tools: list[Tool],
    ) -> LlmMessage:
        """Return one model response, possibly containing tool calls."""
        ...
