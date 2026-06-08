"""File operation tools for saving research reports."""

import fcntl
import re
from pathlib import Path

from langchain_core.tools import tool

from app.tools.responses import error, ok

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
_LOCK_FILE = REPORTS_DIR / ".write.lock"
_FILENAME_PATTERN = re.compile(r"^research_\d+\.md$")


def _ensure_reports_dir() -> None:
    """Create the reports directory if it does not exist."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _sanitize_filename(name: str) -> str | None:
    """Return a safe basename or None if the filename is invalid."""
    stripped = name.strip()
    if not stripped:
        return None
    if ".." in stripped or stripped.startswith(("/", "\\")):
        return None
    if Path(stripped).is_absolute():
        return None
    basename = Path(stripped).name
    if not basename.endswith(".md"):
        basename = f"{basename}.md"
    if not _FILENAME_PATTERN.match(basename):
        return None
    return basename


def _next_report_filename() -> str:
    """Generate the next auto-incremented report filename under file lock."""
    _ensure_reports_dir()
    with open(_LOCK_FILE, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        existing = list(REPORTS_DIR.glob("research_*.md"))
        max_num = 0
        pattern = re.compile(r"research_(\d+)\.md$")
        for path in existing:
            match = pattern.search(path.name)
            if match:
                max_num = max(max_num, int(match.group(1)))
        return f"research_{max_num + 1:03d}.md"


def _resolve_report_path(name: str) -> Path | None:
    """Resolve a filename to an absolute path inside REPORTS_DIR."""
    file_path = (REPORTS_DIR / name).resolve()
    if not file_path.is_relative_to(REPORTS_DIR.resolve()):
        return None
    return file_path


@tool
def write_file(content: str, filename: str = "") -> str:
    """Save content to a markdown file in the reports/ directory.

    DO NOT provide a filename argument unless strictly necessary —
    the system will auto-generate one (research_001.md, research_002.md, ...).
    Put the full report body in `content` only — never in chat messages.

    Args:
        content: Full markdown report content to save.
        filename: Optional. Leave empty to let the system auto-generate a filename.

    Returns:
        OK-prefixed absolute path on success, ERROR-prefixed message on failure.
    """
    _ensure_reports_dir()

    if filename.strip():
        safe_name = _sanitize_filename(filename)
        if safe_name is None:
            return error("invalid filename")
        name = safe_name
    else:
        name = _next_report_filename()

    file_path = _resolve_report_path(name)
    if file_path is None:
        return error("invalid filename")

    try:
        file_path.write_text(content, encoding="utf-8")
    except OSError as exc:
        return error(f"failed to save file: {exc}")

    return ok(str(file_path))
