"""Tests for core agent helpers and orchestration."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.core import (
    ResearchResult,
    _count_tool_calls,
    _extract_report_path,
    _is_valid_report_path,
    _normalize_tool_path,
    execute_research_loop,
)
from app.tools.file_ops import REPORTS_DIR
from app.tools.responses import ok

VALID_REPORT = """# Тема исследования
> Дата исследования: 2026-06-06
Тестовая тема

## Краткое резюме
Краткое описание результатов с аналитическим выводом.

## Основные находки
- Факт один

## Анализ
Источники согласуются в базовых трендах, но расходятся в оценке масштаба эффекта.
Первый источник указывает на умеренный рост, второй — на более резкие изменения.
Это говорит о неоднородности данных и необходимости осторожных выводов.

## Выводы
- Тема демонстрирует устойчивый интерес в открытых источниках
- Для точных оценок нужны дополнительные первичные данные

## Источники
- [Пример](https://example.com/article)
"""

SEARCH_RESULT = ok(
    "1. Example\nSnippet\nSource: https://example.com/article"
)


def test_is_valid_report_path_accepts_reports_dir() -> None:
    """Paths inside reports/ should be accepted."""
    path = str((REPORTS_DIR / "research_001.md").resolve())
    assert _is_valid_report_path(path) is True


def test_is_valid_report_path_rejects_outside_reports_dir() -> None:
    """Paths outside reports/ should be rejected."""
    assert _is_valid_report_path("/tmp/research_001.md") is False


def test_normalize_tool_path_ok_prefix() -> None:
    """OK-prefixed paths should be parsed correctly."""
    path = str((REPORTS_DIR / "research_001.md").resolve())
    assert _normalize_tool_path(ok(path)) == path


def test_normalize_tool_path_rejects_unprefixed() -> None:
    """Unprefixed paths should be rejected."""
    path = str((REPORTS_DIR / "research_001.md").resolve())
    assert _normalize_tool_path(path) is None


def test_extract_report_path_from_messages() -> None:
    """Report path should be extracted from write_file tool messages."""
    path = str((REPORTS_DIR / "research_001.md").resolve())
    messages = [
        HumanMessage(content="Research topic: test"),
        AIMessage(content="", tool_calls=[{"name": "write_file", "args": {}, "id": "1"}]),
        ToolMessage(content=ok(path), tool_call_id="1", name="write_file"),
    ]
    assert _extract_report_path(messages) == path


def test_count_tool_calls() -> None:
    """Tool call counting should work across AI messages."""
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "web_search", "args": {"query": "a"}, "id": "1"},
                {"name": "web_search", "args": {"query": "b"}, "id": "2"},
            ],
        ),
        ToolMessage(content="ok", tool_call_id="1", name="web_search"),
        ToolMessage(content="ok", tool_call_id="2", name="web_search"),
    ]
    assert _count_tool_calls(messages, "web_search") == 2


def _build_success_messages(report_path: str) -> list:
    """Build a minimal successful agent message history."""
    return [
        HumanMessage(content="Research topic: test"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "web_search", "args": {"query": "a"}, "id": "1"},
                {"name": "web_search", "args": {"query": "b"}, "id": "2"},
                {"name": "write_file", "args": {}, "id": "3"},
            ],
        ),
        ToolMessage(content=SEARCH_RESULT, tool_call_id="1", name="web_search"),
        ToolMessage(content=SEARCH_RESULT, tool_call_id="2", name="web_search"),
        ToolMessage(content=ok(report_path), tool_call_id="3", name="write_file"),
        AIMessage(content="Отчёт сохранён."),
    ]


@pytest.mark.asyncio
async def test_execute_research_loop_success(tmp_path, monkeypatch) -> None:
    """Valid report with 2 searches should succeed."""
    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setattr("app.tools.file_ops.REPORTS_DIR", reports)
    monkeypatch.setattr("app.tools.file_ops._LOCK_FILE", reports / ".write.lock")
    monkeypatch.setattr("app.agent.core.REPORTS_DIR", reports)

    report_path = reports / "research_001.md"
    report_path.write_text(VALID_REPORT, encoding="utf-8")
    messages = _build_success_messages(str(report_path.resolve()))

    async def invoke_fn(_messages, _topic):
        return {"messages": messages}

    result = await execute_research_loop(invoke_fn, "test topic")
    assert isinstance(result, ResearchResult)
    assert result.report_path == str(report_path.resolve())


@pytest.mark.asyncio
async def test_execute_research_loop_rejects_single_search(tmp_path, monkeypatch) -> None:
    """A single web_search call should fail validation."""
    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setattr("app.tools.file_ops.REPORTS_DIR", reports)
    monkeypatch.setattr("app.tools.file_ops._LOCK_FILE", reports / ".write.lock")
    monkeypatch.setattr("app.agent.core.REPORTS_DIR", reports)
    monkeypatch.setenv("RESEARCH_MAX_ATTEMPTS", "1")

    report_path = reports / "research_001.md"
    report_path.write_text(VALID_REPORT, encoding="utf-8")
    messages = [
        HumanMessage(content="Research topic: test"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "web_search", "args": {"query": "a"}, "id": "1"},
                {"name": "write_file", "args": {}, "id": "2"},
            ],
        ),
        ToolMessage(content=SEARCH_RESULT, tool_call_id="1", name="web_search"),
        ToolMessage(content=ok(str(report_path.resolve())), tool_call_id="2", name="write_file"),
    ]

    async def invoke_fn(_messages, _topic):
        return {"messages": messages}

    with pytest.raises(ValueError, match="web_search calls"):
        await execute_research_loop(invoke_fn, "test topic")


@pytest.mark.asyncio
async def test_execute_research_loop_retry_uses_fresh_messages(
    tmp_path, monkeypatch
) -> None:
    """Retry should restart with topic + feedback, not full failed history."""
    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setattr("app.tools.file_ops.REPORTS_DIR", reports)
    monkeypatch.setattr("app.tools.file_ops._LOCK_FILE", reports / ".write.lock")
    monkeypatch.setattr("app.agent.core.REPORTS_DIR", reports)
    monkeypatch.setenv("RESEARCH_MAX_ATTEMPTS", "2")

    seen_lengths: list[int] = []

    async def invoke_fn(messages, _topic):
        seen_lengths.append(len(messages))
        return {"messages": messages}

    with pytest.raises(ValueError):
        await execute_research_loop(invoke_fn, "test topic")

    assert seen_lengths == [1, 2]
