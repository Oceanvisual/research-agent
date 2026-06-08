"""Shared middleware utilities for LangChain AgentMiddleware."""

from collections.abc import Sequence
from typing import Any, cast

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest
from langchain_core.messages import SystemMessage


def append_system_prompt(request: ModelRequest, extra_prompt: str) -> ModelRequest:
    """Return a model request with extra text appended to the system message."""
    if request.system_message is not None:
        new_system_content = [
            *request.system_message.content_blocks,
            {"type": "text", "text": f"\n\n{extra_prompt}"},
        ]
    else:
        new_system_content = [{"type": "text", "text": extra_prompt}]
    new_system_message = SystemMessage(
        content=cast("list[str | dict[str, str]]", new_system_content)
    )
    return request.override(system_message=new_system_message)


class ToolsInjectionMiddleware(AgentMiddleware):
    """Registers tools with the create_agent middleware harness."""

    def __init__(self, tools: Sequence[Any]) -> None:
        """Initialize with tools to expose to the agent."""
        super().__init__()
        self.tools = list(tools)


class SearchMiddleware(ToolsInjectionMiddleware):
    """Injects web search tools into the agent via AgentMiddleware.tools."""


class FileSystemMiddleware(ToolsInjectionMiddleware):
    """Injects file operation tools into the agent via AgentMiddleware.tools."""
