"""Tests for LangChain AgentMiddleware implementations."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import SystemMessage

from app.middlewares.base import (
    FileSystemMiddleware,
    SearchMiddleware,
    append_system_prompt,
)
from app.middlewares.skills import SkillsMiddleware
from app.tools.file_ops import write_file
from app.tools.search import web_search


def test_search_middleware_registers_tools() -> None:
    """Search middleware should expose configured tools to create_agent."""
    middleware = SearchMiddleware(tools=[web_search])
    assert len(middleware.tools) == 1
    assert middleware.tools[0].name == "web_search"


def test_filesystem_middleware_registers_tools() -> None:
    """Filesystem middleware should expose configured tools to create_agent."""
    middleware = FileSystemMiddleware(tools=[write_file])
    assert len(middleware.tools) == 1
    assert middleware.tools[0].name == "write_file"


def test_append_system_prompt() -> None:
    """System prompt helper should append text blocks."""
    request = MagicMock()
    request.system_message = SystemMessage(content="Base prompt")
    request.override.return_value = "updated-request"

    result = append_system_prompt(request, "Extra instructions")

    assert result == "updated-request"
    override_kwargs = request.override.call_args.kwargs
    assert "Extra instructions" in override_kwargs["system_message"].text


@pytest.mark.asyncio
async def test_skills_middleware_async_wrap() -> None:
    """Skills middleware should append researcher prompt before model call."""
    middleware = SkillsMiddleware()
    request = MagicMock()
    request.system_message = SystemMessage(content="Base prompt")
    request.override.side_effect = lambda **kwargs: kwargs

    handler = AsyncMock(return_value="model-response")
    result = await middleware.awrap_model_call(request, handler)

    assert result == "model-response"
    handler.assert_awaited_once()
    override_kwargs = request.override.call_args.kwargs
    assert "RESEARCH PROTOCOL" in override_kwargs["system_message"].text
