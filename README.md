# Research Agent

AI-агент для исследования тем из интернета: web search → аналитический синтез → markdown-отчёт.  
Стек: LangGraph / LangChain, Tavily, OpenRouter, Streamlit.

## Быстрый старт

```bash
cd research_agent
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # заполнить ключи (см. ниже)
streamlit run app/main.py
```

Открой в браузере: **http://localhost:8501** (или порт, который покажет Streamlit).

## API-ключи

| Переменная | Обязательна | Где взять |
|---|---|---|
| `OPENROUTER_API_KEY` | да | [openrouter.ai](https://openrouter.ai/) → Keys |
| `TAVILY_API_KEY` | да | [tavily.com](https://tavily.com/) → API Keys |

Опционально:

| Переменная | По умолчанию | Описание |
|---|---|---|
| `LLM_MODEL` | `google/gemini-2.5-flash` | Slug модели на OpenRouter |
| `RESEARCH_MAX_ATTEMPTS` | `2` | Повторы при ошибке валидации отчёта |
| `LOG_LEVEL` | `INFO` | Уровень логирования |

Пример `.env`:

```env
OPENROUTER_API_KEY=sk-or-...
TAVILY_API_KEY=tvly-...
LLM_MODEL=google/gemini-2.5-flash
```

## Модель

По умолчанию используется **`google/gemini-2.5-flash`** через [OpenRouter](https://openrouter.ai/) (OpenAI-compatible API).

Сменить модель можно в `.env`:

```env
LLM_MODEL=anthropic/claude-sonnet-4
```

Любой slug с OpenRouter: `google/...`, `openai/...`, `anthropic/...`, `meta-llama/...`.

## Что реализовано

| Функция | Статус |
|---|---|
| Web search через Tavily (2–3 запроса с разных углов) | ★★★ |
| Аналитический отчёт: находки + анализ + выводы | ★★★ |
| Сохранение отчёта в `reports/research_NNN.md` | ★★★ |
| Валидация структуры и grounding URL по результатам поиска | ★★★ |
| Retry при невалидном отчёте | ★★☆ |
| Streamlit UI с предпросмотром отчёта | ★★☆ |
| Middleware pipeline (search, filesystem, skills) | ★★☆ |
| Диалоговый режим / follow-up вопросы | ☆☆☆ |
| Кастомные промпты через UI | ☆☆☆ |

**Минимум (MVP):** тема → 2–3 поиска → markdown-отчёт с анализом и источниками.  
**★★★** — готово и проверено; **★★☆** — работает, но без расширенного UX; **☆☆☆** — не реализовано.

## Как работает

1. Пользователь вводит тему в Streamlit.
2. Агент формулирует исследовательские вопросы и делает 2–3 `web_search`.
3. Синтезирует отчёт: резюме → факты → **анализ** → **выводы** → источники.
4. Сохраняет через `write_file`, валидирует, при ошибке — retry.

Структура отчёта:

```markdown
# Тема исследования
## Краткое резюме
## Основные находки
## Анализ
## Выводы
## Источники
```

## Тесты

```bash
pip install -r requirements-dev.txt
pytest tests/
```

## Архитектура

- **Tools**: `web_search` (Tavily, async), `write_file` (auto-increment, sandbox)
- **Agent**: `langchain.agents.create_agent` + LLM через OpenRouter
- **Middlewares**: `SearchMiddleware`, `FileSystemMiddleware`, `SkillsMiddleware`
- **Orchestration**: validation + retry в `app/agent/core.py`, runner в `app/agent/runner.py`
- **UI**: Streamlit в `app/main.py`

## Дальнейшее развитие

- **Диалоговый режим**: `checkpointer=MemorySaver()` + `thread_id` для follow-up («добавь в отчёт ещё этот факт»)
- **Кастомные промпты**: редактирование `app/agent/prompts.py` или UI для промптов
