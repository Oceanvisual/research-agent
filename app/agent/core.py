"""LangGraph agent setup and research execution."""

import functools
import logging
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

import app.config  # noqa: F401 — load .env once at import
from app.agent.prompts import USER_TASK_TEMPLATE, build_base_system_prompt
from app.agent.validation import (
    extract_search_urls_from_messages,
    validate_report_file,
    validate_search_count,
)
from app.middlewares.base import FileSystemMiddleware, SearchMiddleware
from app.middlewares.skills import SkillsMiddleware
from app.tools.file_ops import REPORTS_DIR, write_file
from app.tools.responses import parse_ok
from app.tools.search import web_search

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_REPORT_FILENAME_PATTERN = re.compile(r"^research_\d+\.md$")


@dataclass(frozen=True)
class ResearchResult:
    """Result of a completed research run."""

    report_path: str
    summary: str


def _create_llm() -> ChatOpenAI:
    """Initialize LLM via OpenRouter (OpenAI-compatible API)."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set in environment variables."
        )

    model = os.getenv("LLM_MODEL", "google/gemini-2.5-flash")
    if not model.startswith(("google/", "openai/", "anthropic/", "meta-llama/")):
        model = f"google/{model}"

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=0.3,
    )


def _build_agent() -> Any:
    """Apply middleware pipeline and return the research agent graph."""
    return create_agent(
        model=_create_llm(),
        tools=[],
        system_prompt=build_base_system_prompt(),
        middleware=[
            SearchMiddleware(tools=[web_search]),
            FileSystemMiddleware(tools=[write_file]),
            SkillsMiddleware(),
        ],
    )


@functools.lru_cache(maxsize=1)
def get_research_agent() -> Any:
    """Return a cached compiled research agent graph."""
    return _build_agent()


def _is_valid_report_path(content: str) -> bool:
    """Return True if content looks like a write_file result path."""
    if not content.endswith(".md") or "research_" not in content:
        return False
    try:
        path = Path(content).resolve()
    except (OSError, ValueError):
        return False
    if not path.is_absolute():
        return False
    if _REPORT_FILENAME_PATTERN.match(path.name) is None:
        return False
    try:
        return path.is_relative_to(REPORTS_DIR.resolve())
    except (ValueError, OSError):
        return False


def _normalize_tool_path(content: str) -> str | None:
    """Extract a report path from an OK-prefixed write_file response."""
    payload = parse_ok(content)
    if payload is None:
        return None
    return payload if _is_valid_report_path(payload) else None


def _extract_report_path(messages: list[Any]) -> str | None:
    """Extract saved report path from write_file tool messages only."""
    for message in reversed(messages):
        if getattr(message, "type", None) != "tool":
            continue
        if getattr(message, "name", None) != "write_file":
            continue
        content = getattr(message, "content", "")
        if isinstance(content, str):
            path = _normalize_tool_path(content)
            if path is not None:
                return path
    return None


def _extract_agent_summary(messages: list[Any]) -> str:
    """Extract the agent's final conversational reply."""
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        content = message.content
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            text_parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            summary = "\n".join(part for part in text_parts if part.strip())
            if summary.strip():
                return summary.strip()
    return ""


def _count_tool_calls(messages: list[Any], tool_name: str) -> int:
    """Count how many times a tool was invoked in the message history."""
    count = 0
    for message in messages:
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            continue
        for call in tool_calls:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name == tool_name:
                count += 1
    return count


def _build_retry_feedback(
    *,
    report_path: str | None,
    violations: list[str],
) -> str:
    """Build a human message asking the agent to fix issues on retry."""
    parts: list[str] = []
    if report_path is None:
        parts.append(
            "Report was not saved. Call write_file with the full report body "
            "and wait for an OK-prefixed file path before your final reply."
        )
    if violations:
        parts.append("Report validation failed:")
        parts.extend(f"- {item}" for item in violations)
    parts.append("Fix the issues and call write_file again with a corrected report.")
    return "\n".join(parts)


def _build_retry_messages(topic: str, feedback: str) -> list[BaseMessage]:
    """Build a fresh message list for a retry attempt."""
    return [
        HumanMessage(content=USER_TASK_TEMPLATE.format(topic=topic)),
        HumanMessage(content=feedback),
    ]


def _remove_failed_report(report_path: str | None) -> None:
    """Delete an invalid report file before retrying."""
    if report_path is None:
        return
    try:
        Path(report_path).unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Failed to remove invalid report %s: %s", report_path, exc)


def _log_run_telemetry(
    *,
    topic: str,
    attempt: int,
    search_count: int,
    report_path: str | None,
    violations: list[str],
) -> None:
    """Log attempt-level telemetry for a research run."""
    logger.info(
        "Research attempt %s topic=%r searches=%s report_path=%r violations=%s",
        attempt,
        topic,
        search_count,
        report_path,
        len(violations),
    )
    if search_count < 2 or search_count > 3:
        logger.warning(
            "Unexpected web_search count=%s for topic=%r (expected 2-3)",
            search_count,
            topic,
        )


async def execute_research_loop(
    invoke_fn: Callable[[list[BaseMessage], str], Awaitable[dict[str, Any]]],
    topic: str,
) -> ResearchResult:
    """Run the research agent with validation and retry orchestration.

    Args:
        invoke_fn: Async callable that executes one agent turn.
        topic: Research topic from the user.

    Returns:
        Report file path and the agent's final summary message.

    Raises:
        ValueError: If the topic is empty or report cannot be validated.
        RuntimeError: If the agent execution fails.
    """
    topic = topic.strip()
    if not topic:
        raise ValueError("Research topic cannot be empty.")

    max_attempts = int(os.getenv("RESEARCH_MAX_ATTEMPTS", "2"))
    messages: list[BaseMessage] = [
        HumanMessage(content=USER_TASK_TEMPLATE.format(topic=topic))
    ]
    last_violations: list[str] = []

    for attempt in range(1, max_attempts + 1):
        try:
            result = await invoke_fn(messages, topic)
        except Exception as exc:
            raise RuntimeError(f"Agent execution failed: {exc}") from exc

        messages = result.get("messages", messages)
        search_count = _count_tool_calls(messages, "web_search")
        report_path = _extract_report_path(messages)
        allowed_urls = extract_search_urls_from_messages(messages)
        violations: list[str] = []
        if report_path:
            violations.extend(
                validate_report_file(
                    Path(report_path),
                    allowed_source_urls=allowed_urls,
                )
            )
        else:
            violations.append("Report file path was not produced by write_file")
        violations.extend(validate_search_count(search_count))

        _log_run_telemetry(
            topic=topic,
            attempt=attempt,
            search_count=search_count,
            report_path=report_path,
            violations=violations,
        )

        if report_path and not violations:
            summary = _extract_agent_summary(messages)
            return ResearchResult(report_path=report_path, summary=summary)

        last_violations = violations
        if attempt < max_attempts:
            _remove_failed_report(report_path)
            feedback = _build_retry_feedback(
                report_path=report_path,
                violations=violations,
            )
            messages = _build_retry_messages(topic, feedback)

    violation_text = "; ".join(last_violations) if last_violations else "unknown error"
    raise ValueError(f"Report validation failed after {max_attempts} attempt(s): {violation_text}")
