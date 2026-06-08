"""Tests for web search tool."""

from unittest.mock import MagicMock, patch

import pytest

from app.tools.responses import ERROR_PREFIX, OK_PREFIX
from app.tools.search import _sync_search, web_search


def test_sync_search_formats_results() -> None:
    """Successful search should return OK-prefixed formatted results."""
    mock_client = MagicMock()
    mock_client.search.return_value = {
        "results": [
            {
                "title": "Test Title",
                "content": "Snippet text",
                "url": "https://example.com/page",
            }
        ]
    }

    with patch("app.tools.search._get_tavily_client", return_value=mock_client):
        result = _sync_search("test query")

    assert result.startswith(OK_PREFIX)
    assert "Source: https://example.com/page" in result
    mock_client.search.assert_called_once_with(query="test query", max_results=3)


def test_sync_search_no_results() -> None:
    """Empty results should return ERROR-prefixed message."""
    mock_client = MagicMock()
    mock_client.search.return_value = {"results": []}

    with patch("app.tools.search._get_tavily_client", return_value=mock_client):
        result = _sync_search("empty query")

    assert result.startswith(ERROR_PREFIX)


@pytest.mark.asyncio
async def test_web_search_async_wrapper() -> None:
    """Async web_search should delegate to sync search via thread."""
    with patch("app.tools.search._sync_search", return_value=f"{OK_PREFIX}results"):
        result = await web_search.ainvoke({"query": "async test"})
    assert result == f"{OK_PREFIX}results"
