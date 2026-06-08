"""Streamlit UI entry point for the research agent."""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

import app.config  # noqa: F401 — load .env once at import
from app.agent.runner import run_research_async
from app.config import setup_logging, validate_env
setup_logging()

st.set_page_config(page_title="Research Agent", page_icon="🔬", layout="centered")

st.title("🔬 AI Research Agent")
st.caption("Исследование темы с web search, аналитическим синтезом и отчётом")

try:
    validate_env()
except ValueError as exc:
    st.error(f"Ошибка конфигурации: {exc}")
    st.stop()

topic = st.text_input(
    "Тема исследования",
    placeholder="Например: влияние LLM на образование",
)

if st.button("Начать исследование", type="primary", disabled=not topic.strip()):
    steps: list[str] = []

    def on_step(label: str) -> None:
        steps.append(label)

    with st.status("Исследование...", expanded=True) as status:
        try:
            result = asyncio.run(run_research_async(topic, on_step=on_step))
            for step in steps:
                st.write(step)
            status.update(label="Исследование завершено", state="complete")
        except ValueError as exc:
            status.update(label="Ошибка валидации", state="error")
            st.error(f"Ошибка: {exc}")
        except RuntimeError as exc:
            status.update(label="Ошибка выполнения", state="error")
            st.error(f"Ошибка выполнения агента: {exc}")
        except Exception as exc:
            status.update(label="Непредвиденная ошибка", state="error")
            st.error(f"Непредвиденная ошибка: {exc}")
        else:
            st.success("Исследование завершено успешно!")
            if result.summary:
                st.markdown(result.summary)
            st.info(f"Отчёт сохранён: `{result.report_path}`")

            report_file = Path(result.report_path)
            if report_file.exists():
                with st.expander("Предпросмотр отчёта"):
                    st.markdown(report_file.read_text(encoding="utf-8"))
