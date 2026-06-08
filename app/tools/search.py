"""Web search tool powered by Tavily."""

import asyncio
import os

from langchain_core.tools import tool
from tavily import TavilyClient

from app.tools.responses import error, ok

_client: TavilyClient | None = None


def _get_tavily_client() -> TavilyClient:
    """Return a singleton Tavily client using the API key from environment."""
    global _client
    if _client is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY is not set in environment variables.")
        _client = TavilyClient(api_key=api_key)
    return _client


def _sync_search(query: str) -> str:
    """Perform a synchronous Tavily search and format results."""
    client = _get_tavily_client()
    response = client.search(query=query, max_results=3)
    results = response.get("results", [])
    if not results:
        return error(f"no results found for query: {query}")

    formatted: list[str] = []
    for idx, item in enumerate(results, start=1):
        title = item.get("title", "Untitled")
        content = item.get("content", "")
        url = item.get("url", "")
        formatted.append(f"{idx}. {title}\n{content}\nSource: {url}")

    return ok("\n\n".join(formatted))


@tool
async def web_search(query: str) -> str:
    """Search the web for information on a given topic.

    Performs ONE search per call. You MUST call this tool 2-3 times
    with different search queries covering distinct research angles
    (core facts, recent developments, criticism/contrasts, regional context).

    Args:
        query: A focused search query string.

    Returns:
        OK-prefixed formatted search results, or ERROR-prefixed message on failure.
        Each result includes a "Source:" line with the page URL — copy these URLs exactly into the report.
    """
    try:
        return await asyncio.to_thread(_sync_search, query)
    except Exception as exc:
        return error(f"web search failed: {exc}")
