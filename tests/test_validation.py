"""Tests for report validation."""

from pathlib import Path

from app.agent.validation import validate_report_content, validate_report_file

VALID_REPORT = """# Тема исследования
> Дата исследования: 2026-06-06
Тестовая тема

## Краткое резюме
Краткое описание результатов с аналитическим выводом по теме исследования.

## Основные находки
- Факт один
- Факт два

## Анализ
Источники согласуются в базовых трендах, но расходятся в оценке масштаба эффекта.
Первый источник указывает на умеренный рост, второй — на более резкие изменения.
Это говорит о неоднородности данных и необходимости осторожных выводов при интерпретации.

## Выводы
- Тема демонстрирует устойчивый интерес в открытых источниках
- Для точных количественных оценок нужны дополнительные первичные данные

## Источники
- [Пример](https://example.com/article)
"""


def test_validate_report_content_valid() -> None:
    """Valid report markdown should produce no violations."""
    assert validate_report_content(VALID_REPORT) == []


def test_validate_report_content_missing_headers() -> None:
    """Missing headers should be reported."""
    content = VALID_REPORT.replace("## Источники", "")
    violations = validate_report_content(content)
    assert any("Источники" in item for item in violations)


def test_validate_report_content_missing_date() -> None:
    """Missing research date should be reported."""
    content = VALID_REPORT.replace("> Дата исследования: 2026-06-06", "")
    violations = validate_report_content(content)
    assert any("Дата исследования" in item for item in violations)


def test_validate_report_content_bad_sources() -> None:
    """Sources without markdown links should fail validation."""
    content = VALID_REPORT.replace(
        "- [Пример](https://example.com/article)",
        "- example.com",
    )
    violations = validate_report_content(content)
    assert any("markdown link" in item for item in violations)


def test_validate_report_file(tmp_path: Path) -> None:
    """File-based validation should read and validate content."""
    report = tmp_path / "research_001.md"
    report.write_text(VALID_REPORT, encoding="utf-8")
    assert validate_report_file(report) == []


def test_validate_source_grounding_rejects_unknown_url() -> None:
    """Report URLs must appear in web_search output."""
    allowed = {"https://example.com/article"}
    violations = validate_report_content(
        VALID_REPORT,
        allowed_source_urls=allowed,
    )
    assert violations == []


def test_validate_source_grounding_flags_hallucinated_url() -> None:
    """Hallucinated URLs should fail grounding validation."""
    content = VALID_REPORT.replace(
        "https://example.com/article",
        "https://evil.com/fake",
    )
    violations = validate_report_content(
        content,
        allowed_source_urls={"https://example.com/article"},
    )
    assert any("not found in web_search results" in item for item in violations)


def test_validate_search_count() -> None:
    """Search count validator should enforce 2-3 calls."""
    from app.agent.validation import validate_search_count

    assert validate_search_count(2) == []
    assert validate_search_count(3) == []
    assert validate_search_count(1) != []
    assert validate_search_count(4) != []


def test_validate_analytical_sections_rejects_short_analysis() -> None:
    """Analysis section must contain substantive prose."""
    content = VALID_REPORT.replace(
        "## Анализ\n"
        "Источники согласуются в базовых трендах, но расходятся в оценке масштаба эффекта.\n"
        "Первый источник указывает на умеренный рост, второй — на более резкие изменения.\n"
        "Это говорит о неоднородности данных и необходимости осторожных выводов при интерпретации.",
        "## Анализ\nКоротко.",
    )
    violations = validate_report_content(content)
    assert any("Анализ" in item for item in violations)


def test_validate_analytical_sections_rejects_short_conclusions() -> None:
    """Conclusions section must contain clear takeaways."""
    content = VALID_REPORT.replace(
        "## Выводы\n"
        "- Тема демонстрирует устойчивый интерес в открытых источниках\n"
        "- Для точных количественных оценок нужны дополнительные первичные данные",
        "## Выводы\n- Ок",
    )
    violations = validate_report_content(content)
    assert any("Выводы" in item for item in violations)
