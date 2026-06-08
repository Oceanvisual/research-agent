"""Skills middleware — injects researcher skill into system prompt."""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse

from app.agent.prompts import build_researcher_skill_prompt
from app.middlewares.base import append_system_prompt


class SkillsMiddleware(AgentMiddleware):
    """Injects the Researcher skill prompt before each model call."""

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """Append researcher workflow instructions to the system prompt."""
        skill_prompt = build_researcher_skill_prompt()
        return handler(append_system_prompt(request, skill_prompt))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        """Async variant for ainvoke/astream execution paths."""
        skill_prompt = build_researcher_skill_prompt()
        return await handler(append_system_prompt(request, skill_prompt))
