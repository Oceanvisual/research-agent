"""Report validation for research agent output."""

import re
from pathlib import Path
from typing import Any

from app.tools.responses import ERROR_PREFIX, parse_ok

REQUIRED_HEADERS: tuple[str, ...] = (
    "# Тема исследования",
    "## Краткое резюме",
    "## Основные находки",
    "## Анализ",
    "## Выводы",
    "## Источники",
)
DATE_PATTERN = re.compile(r"Дата исследования:\s*\d{4}-\d{2}-\d{2}")
MD_LINK_PATTERN = re.compile(r"\[[^\]]+\]\(https?://[^)]+\)")
MD_LINK_URL_PATTERN = re.compile(r"\[[^\]]+\]\((https?://[^)]+)\)")
SOURCE_LINE_PATTERN = re.compile(r"^Source:\s*(https?://\S+)", re.MULTILINE)

MIN_SEARCH_CALLS = 2
MAX_SEARCH_CALLS = 3
MIN_ANALYSIS_CHARS = 200
MIN_CONCLUSIONS_CHARS = 40


def extract_urls_from_search_output(content: str) -> set[str]:
    """Extract page URLs from formatted web_search tool output."""
    return set(SOURCE_LINE_PATTERN.findall(content))


def extract_search_urls_from_messages(messages: list[Any]) -> set[str]:
    """Collect all URLs returned by web_search tool messages."""
    urls: set[str] = set()
    for message in messages:
        if getattr(message, "type", None) != "tool":
            continue
        if getattr(message, "name", None) != "web_search":
            continue
        content = getattr(message, "content", "")
        if not isinstance(content, str) or content.startswith(ERROR_PREFIX):
            continue
        payload = parse_ok(content)
        if payload is None:
            continue
        urls.update(extract_urls_from_search_output(payload))
    return urls


def extract_report_source_urls(content: str) -> list[str]:
    """Extract URLs from markdown links in the sources section."""
    sources_idx = content.find("## Источники")
    if sources_idx < 0:
        return []
    return MD_LINK_URL_PATTERN.findall(content[sources_idx:])


def _extract_section_body(content: str, header: str, next_headers: tuple[str, ...]) -> str:
    """Return text between a section header and the next known header."""
    start = content.find(header)
    if start < 0:
        return ""
    body_start = start + len(header)
    end = len(content)
    for next_header in next_headers:
        idx = content.find(next_header, body_start)
        if idx >= 0:
            end = min(end, idx)
    return content[body_start:end].strip()


def validate_analytical_sections(content: str) -> list[str]:
    """Ensure analysis and conclusions sections contain substantive content."""
    violations: list[str] = []

    analysis_body = _extract_section_body(
        content,
        "## Анализ",
        ("## Выводы", "## Источники"),
    )
    if len(analysis_body) < MIN_ANALYSIS_CHARS:
        violations.append(
            f"Section 'Анализ' is too short ({len(analysis_body)} chars, "
            f"minimum {MIN_ANALYSIS_CHARS}). Write analytical prose, not bullet rephrasing."
        )

    conclusions_body = _extract_section_body(
        content,
        "## Выводы",
        ("## Источники",),
    )
    if len(conclusions_body) < MIN_CONCLUSIONS_CHARS:
        violations.append(
            f"Section 'Выводы' is too short ({len(conclusions_body)} chars, "
            f"minimum {MIN_CONCLUSIONS_CHARS}). Add clear research conclusions."
        )

    return violations


def validate_search_count(search_count: int) -> list[str]:
    """Validate that web_search was called within the required range."""
    if MIN_SEARCH_CALLS <= search_count <= MAX_SEARCH_CALLS:
        return []
    return [
        f"Expected {MIN_SEARCH_CALLS}-{MAX_SEARCH_CALLS} web_search calls, "
        f"got {search_count}"
    ]


def validate_source_grounding(content: str, allowed_urls: set[str]) -> list[str]:
    """Ensure report source URLs came from web_search tool output."""
    report_urls = extract_report_source_urls(content)
    if not report_urls:
        return []

    if not allowed_urls:
        return ["Sources cite URLs but no web_search results were available"]

    violations: list[str] = []
    for url in report_urls:
        if url not in allowed_urls:
            violations.append(f"Source URL not found in web_search results: {url}")
    return violations


def validate_report_content(
    content: str,
    *,
    allowed_source_urls: set[str] | None = None,
) -> list[str]:
    """Validate markdown report content against required structure.

    Args:
        content: Raw markdown report body.

    Returns:
        List of validation violation messages. Empty list means valid.
    """
    violations: list[str] = []
    if not content.strip():
        return ["Report content is empty"]

    for header in REQUIRED_HEADERS:
        if header not in content:
            violations.append(f"Missing required header: {header}")

    if not DATE_PATTERN.search(content):
        violations.append(
            "Missing research date line (Дата исследования: YYYY-MM-DD)"
        )

    sources_idx = content.find("## Источники")
    if sources_idx >= 0:
        sources_section = content[sources_idx:]
        if not MD_LINK_PATTERN.search(sources_section):
            violations.append(
                "Sources section must contain at least one markdown link [title](url)"
            )

    if allowed_source_urls is not None:
        violations.extend(validate_source_grounding(content, allowed_source_urls))

    violations.extend(validate_analytical_sections(content))

    return violations


def validate_report_file(
    path: Path,
    *,
    allowed_source_urls: set[str] | None = None,
) -> list[str]:
    """Read and validate a report file on disk.

    Args:
        path: Path to the markdown report file.

    Returns:
        List of validation violation messages. Empty list means valid.
    """
    if not path.exists():
        return [f"Report file not found: {path}"]

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"Cannot read report file: {exc}"]

    return validate_report_content(
        content,
        allowed_source_urls=allowed_source_urls,
    )
