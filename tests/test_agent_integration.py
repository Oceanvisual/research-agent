"""Integration tests for create_agent graph assembly."""

import pytest

from app.agent.core import get_research_agent


@pytest.fixture(autouse=True)
def _agent_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide dummy API keys so the agent graph can be compiled in CI."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
    get_research_agent.cache_clear()


def test_research_agent_graph_builds() -> None:
    """create_agent should compile a graph with model and tools nodes."""
    agent = get_research_agent()
    nodes = set(agent.get_graph().nodes.keys())
    assert "model" in nodes
    assert "tools" in nodes


def test_research_agent_exposes_middleware_tools() -> None:
    """Middleware-injected tools should be available to the tool node."""
    agent = get_research_agent()
    tool_names = set(agent.nodes["tools"].bound.tools_by_name.keys())
    assert tool_names == {"web_search", "write_file"}


def test_research_agent_requires_openrouter_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing OpenRouter key should fail fast during graph build."""
    get_research_agent.cache_clear()
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        get_research_agent()
