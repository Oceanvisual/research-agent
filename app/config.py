"""Application configuration, logging, and environment validation."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def setup_logging() -> None:
    """Configure stdlib logging from LOG_LEVEL env var."""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def validate_env() -> None:
    """Validate required environment variables at startup.

    Raises:
        ValueError: If required API keys are missing.
    """
    missing: list[str] = []
    if not os.getenv("OPENROUTER_API_KEY"):
        missing.append("OPENROUTER_API_KEY")
    if not os.getenv("TAVILY_API_KEY"):
        missing.append("TAVILY_API_KEY")
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}"
        )
