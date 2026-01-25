"""
Logging configuration for the RAG-based AI assistant.
Uses loguru when available for structured logging; falls back to the
standard library logging module if loguru is not installed.
"""

import os
import sys
from pathlib import Path

# Optional import of loguru; provide a stdlib fallback when missing.
try:
    from loguru import logger as LOGURU_LOGGER  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    LOGURU_LOGGER = None

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
DEBUG_FILE = os.path.join(LOG_DIR, "debug.log")

# Get log level from environment variable (default: INFO)
LOG_LEVEL = (os.getenv("LOG_LEVEL") or os.environ.get("LOG_LEVEL") or "INFO").upper()

# Validate log level
valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
if LOG_LEVEL not in valid_levels:
    LOG_LEVEL = "INFO"

# If loguru is available, configure it; otherwise use stdlib logging
if LOGURU_LOGGER is not None:
    logger = LOGURU_LOGGER

    # Remove default handlers and add ours
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
        "{name}:{function}:{line} - {message}"
    )

    logger.add(
        LOG_FILE,
        format=FILE_FORMAT,
        rotation="10 MB",
        retention="7 days",
        level=LOG_LEVEL,
        enqueue=False,
    )

    if LOG_LEVEL == "DEBUG":
        logger.add(
            DEBUG_FILE,
            format=CONSOLE_FORMAT,
            colorize=True,
            rotation="5 MB",
            retention="3 days",
            level="DEBUG",
            enqueue=False,
        )
else:
    # stdlib logging fallback
    import logging
    from logging.handlers import RotatingFileHandler

    logger = logging.getLogger("rag_assistant")
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # Clear existing handlers
    logger.handlers = []

    console_fmt = logging.Formatter(
        "%(levelname)-8s | %(name)s:%(funcName)s:%(lineno)s - %(message)s"
    )
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=7
    )
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)s - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    file_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    logger.addHandler(file_handler)

    if LOG_LEVEL == "DEBUG":
        debug_handler = RotatingFileHandler(
            DEBUG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(console_fmt)
        logger.addHandler(debug_handler)

# Log initialization info
try:
    logger.debug(f"Logger initialized with LOG_LEVEL: {LOG_LEVEL}")
    logger.debug(f"Log files location: {LOG_DIR}")
except (AttributeError, TypeError):  # pylint: disable=broad-exception-caught
    # If the logger implementation doesn't support the same API, ignore
    pass

__all__ = ["logger"]
