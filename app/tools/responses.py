"""Structured tool response helpers."""

OK_PREFIX = "OK: "
ERROR_PREFIX = "ERROR: "


def ok(payload: str) -> str:
    """Format a successful tool response."""
    return f"{OK_PREFIX}{payload}"


def error(message: str) -> str:
    """Format a failed tool response."""
    return f"{ERROR_PREFIX}{message}"


def parse_ok(content: str) -> str | None:
    """Extract payload from an OK-prefixed tool response."""
    if content.startswith(OK_PREFIX):
        return content[len(OK_PREFIX) :]
    return None
