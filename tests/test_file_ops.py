"""Tests for file operation tools."""

from pathlib import Path

import pytest

from app.tools import file_ops
from app.tools.responses import ERROR_PREFIX, OK_PREFIX, parse_ok


@pytest.fixture
def reports_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point REPORTS_DIR to a temporary directory."""
    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setattr(file_ops, "REPORTS_DIR", reports)
    monkeypatch.setattr(file_ops, "_LOCK_FILE", reports / ".write.lock")
    return reports


def test_write_file_auto_increment(reports_dir: Path) -> None:
    """Auto-generated filenames should increment sequentially."""
    first = file_ops.write_file.invoke({"content": "# report 1", "filename": ""})
    second = file_ops.write_file.invoke({"content": "# report 2", "filename": ""})

    assert first.startswith(OK_PREFIX)
    assert second.startswith(OK_PREFIX)
    assert first != second
    assert (reports_dir / "research_001.md").exists()
    assert (reports_dir / "research_002.md").exists()


def test_write_file_blocks_path_traversal(reports_dir: Path) -> None:
    """Path traversal attempts should return ERROR."""
    result = file_ops.write_file.invoke(
        {"content": "# evil", "filename": "../../outside.md"}
    )
    assert result.startswith(ERROR_PREFIX)
    assert not (reports_dir.parent / "outside.md").exists()


def test_write_file_blocks_absolute_filename(reports_dir: Path) -> None:
    """Absolute filenames should be rejected."""
    result = file_ops.write_file.invoke(
        {"content": "# evil", "filename": "/tmp/evil.md"}
    )
    assert result.startswith(ERROR_PREFIX)


def test_write_file_ok_prefix(reports_dir: Path) -> None:
    """Successful writes should return OK-prefixed absolute path."""
    result = file_ops.write_file.invoke({"content": "# ok", "filename": ""})
    payload = parse_ok(result)
    assert payload is not None
    assert Path(payload).exists()
