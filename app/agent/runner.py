"""Async research runner with optional streaming callbacks."""

from collections.abc import Callable
from typing import Any

from langchain_core.messages import BaseMessage

from app.agent.core import ResearchResult, execute_research_loop, get_research_agent

StepCallback = Callable[[str], None]
_AGENT_CONFIG = {"recursion_limit": 50}


def _label_from_tool_call(tool_name: str) -> str:
    """Map tool name to a human-readable UI step label."""
    labels = {
        "web_search": "Поиск в интернете...",
        "write_file": "Сохранение отчёта...",
    }
    return labels.get(tool_name, f"Вызов {tool_name}...")


def _emit_new_steps(
    messages: list[BaseMessage],
    start_count: int,
    on_step: StepCallback,
    seen_labels: set[str],
) -> None:
    """Emit UI labels for new tool calls since the attempt started."""
    for message in messages[start_count:]:
        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            name = (
                call.get("name")
                if isinstance(call, dict)
                else getattr(call, "name", None)
            )
            if not name:
                continue
            label = _label_from_tool_call(name)
            if label in seen_labels:
                continue
            seen_labels.add(label)
            on_step(label)


async def run_research_async(
    topic: str,
    on_step: StepCallback | None = None,
) -> ResearchResult:
    """Run a research task asynchronously with optional step callbacks."""
    agent = get_research_agent()

    async def invoke_fn(messages: list[BaseMessage], _topic: str) -> dict[str, Any]:
        input_state = {"messages": messages}

        if on_step is None:
            return await agent.ainvoke(input_state, config=_AGENT_CONFIG)

        start_count = len(messages)
        seen_labels: set[str] = set()
        final_state: dict[str, Any] | None = None

        async for state in agent.astream(
            input_state,
            config=_AGENT_CONFIG,
            stream_mode="values",
        ):
            final_state = state
            _emit_new_steps(
                state.get("messages", []),
                start_count,
                on_step,
                seen_labels,
            )

        if final_state is None:
            return await agent.ainvoke(input_state, config=_AGENT_CONFIG)
        return final_state

    return await execute_research_loop(invoke_fn, topic)
