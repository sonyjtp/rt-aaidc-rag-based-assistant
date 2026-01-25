"""
Logging configuration for the RAG-based AI assistant.
Uses loguru for clean, structured logging.
"""

import os
import sys
from pathlib import Path
from loguru import logger

# Load .env file to ensure LOG_LEVEL is set before logger initialization
from dotenv import load_dotenv
env_file = Path(__file__).parent.parent / ".env"
load_dotenv(env_file)

# Configure logger
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "rag_assistant.log")
DEBUG_FILE = os.path.join(LOG_DIR, "debug.log")

# Get log level from environment variable (default: INFO)
# Try multiple ways to ensure we get the value
LOG_LEVEL = (
    os.getenv("LOG_LEVEL") or
    os.environ.get("LOG_LEVEL") or
    "INFO"
).upper()

# Validate log level
valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
if LOG_LEVEL not in valid_levels:
    LOG_LEVEL = "INFO"

# Remove default handler
logger.remove()

# Add console handler to stderr with immediate flush
logger.add(
    sys.stderr,
    format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
    level=LOG_LEVEL,
    enqueue=False  # Disable queue to ensure immediate output
)

# Add main file handler with rotation
logger.add(
    LOG_FILE,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    rotation="10 MB",
    retention="7 days",
    level=LOG_LEVEL,
    enqueue=False  # Disable queue for immediate writes
)

# Add dedicated debug file handler (always DEBUG level for troubleshooting)
if LOG_LEVEL == "DEBUG":
    logger.add(
        DEBUG_FILE,
        format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,  # Enable colors in the file
        rotation="5 MB",
        retention="3 days",
        level="DEBUG",
        enqueue=False  # Disable queue for immediate writes
    )

# Log initialization info
logger.debug(f"Logger initialized with LOG_LEVEL: {LOG_LEVEL}")
logger.debug(f"Log files location: {LOG_DIR}")

__all__ = ["logger"]















