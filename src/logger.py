"""
Logging configuration for the RAG-based AI assistant.
Uses loguru for structured logging.
"""

import os
import sys
from pathlib import Path

from loguru import logger

# Optional dotenv loader; not required for environments without .env
try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover - optional dependency

    def load_dotenv(*_args, **_kwargs):  # pylint: disable=missing-function-docstring
        """Stub for load_dotenv when python-dotenv is not installed."""
        return False


# Load .env file to ensure LOG_LEVEL is set before logger initialization
env_file = Path(__file__).parent.parent / ".env"
load_dotenv(env_file)

# Configure logger destination paths
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "rag_assistant.log")

# Get log level from environment variable (default: INFO)
LOG_LEVEL = (os.getenv("LOG_LEVEL") or os.environ.get("LOG_LEVEL") or "INFO").upper()

# Validate log level
valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
if LOG_LEVEL not in valid_levels:
    LOG_LEVEL = "INFO"

# Configure loguru
logger.remove()

CONSOLE_FORMAT = (
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

logger.add(
    sys.stderr,
    format=CONSOLE_FORMAT,
    colorize=True,
    level=LOG_LEVEL,
    enqueue=False,
)

FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
    + "{name}:{function}:{line} - {message}"
)

logger.add(
    LOG_FILE,
    format=FILE_FORMAT,
    rotation="10 MB",
    retention="7 days",
    level=LOG_LEVEL,
    enqueue=False,
)

# Log initialization info
logger.debug(f"Logger initialized with LOG_LEVEL: {LOG_LEVEL}")
logger.debug(f"Log files location: {LOG_DIR}")

__all__ = ["logger"]
